"""Hybrid retriever: fuses BM25 and vector search via Reciprocal Rank Fusion.

Every retrieval is entitlement-filtered. The vector branch pre-filters inside
the store; the lexical branch keeps one BM25 index *per tenant* (so another
tenant's documents can neither appear nor skew IDF) and applies the same ACL
rule before ranking, then re-checks hits against the store's current payload.
"""

from __future__ import annotations

import logging
from typing import Any

from online_pipeline.auth.acl import is_authorized
from online_pipeline.auth.principal import Principal

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Combine BM25 and vector search scores using Reciprocal Rank Fusion (RRF).

    Args:
        vector_db: A :class:`~vector_db.client.VectorDBClient` (used for
            the vector branch and to re-validate lexical hits).
        corpus: In-memory document list for BM25 (may be ``None`` to skip).
        bm25_weight: Weight given to BM25 scores in the fusion step.
        vector_weight: Weight given to vector scores in the fusion step.
        rrf_k: RRF rank constant (default 60).
    """

    def __init__(
        self,
        vector_db: Any | None = None,
        corpus: list[dict[str, Any]] | None = None,
        bm25_weight: float = 0.3,
        vector_weight: float = 0.7,
        rrf_k: int = 60,
    ) -> None:
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        self.rrf_k = rrf_k

        self._vector_db = vector_db
        self._vector_retriever = None
        self._bm25_by_tenant: dict[str, Any] = {}

        if vector_db is not None:
            from online_pipeline.retrieval.vector_retriever import VectorRetriever
            self._vector_retriever = VectorRetriever(vector_db)

        if corpus:
            self.load_corpus(corpus)

    # ── public API ───────────────────────────────────────────────────────────

    def load_corpus(self, corpus: list[dict[str, Any]]) -> None:
        """Build (or rebuild) one BM25 index per tenant over *corpus*.

        Documents without a ``tenant_id`` are not indexed at all (default deny).
        The lexical branch is only live once this has been called with a
        non-empty corpus; until then ``retrieve`` is dense-only. Call it again
        after ingesting to bring new chunks into lexical range.

        Note: indexes are held in memory and score every document of the
        caller's tenant per query. That is fine up to tens of thousands of
        chunks; past that the lexical branch belongs in the store as sparse
        vectors.
        """
        by_tenant: dict[str, list[dict[str, Any]]] = {}
        skipped = 0
        for doc in corpus:
            tenant = (doc.get("metadata") or {}).get("tenant_id")
            if not isinstance(tenant, str) or not tenant:
                skipped += 1
                continue
            by_tenant.setdefault(tenant, []).append(doc)
        if skipped:
            logger.warning("BM25: skipped %d chunk(s) with no tenant_id.", skipped)

        from online_pipeline.retrieval.bm25_retriever import BM25Retriever

        indexes: dict[str, Any] = {}
        for tenant, docs in by_tenant.items():
            try:
                indexes[tenant] = BM25Retriever(docs)
            except Exception as exc:  # noqa: BLE001
                # A missing rank-bm25 install must not take the dense branch down.
                logger.warning("BM25 index unavailable (%s); retrieval is dense-only.", exc)
                indexes = {}
                break
        # Swap in one assignment so a concurrent query sees the old or new set,
        # never a half-built one.
        self._bm25_by_tenant = indexes
        if indexes:
            logger.info("BM25 indexes built for %d tenant(s).", len(indexes))

    @property
    def lexical_enabled(self) -> bool:
        """True when the BM25 branch is live and will contribute to fusion."""
        return bool(self._bm25_by_tenant)

    def retrieve(
        self,
        query_text: str,
        query_vector: list[float],
        principal: Principal,
        top_k: int = 20,
        source_in: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Run hybrid retrieval for *principal* and return fused, de-duplicated results.

        Args:
            query_text: Rewritten query text (for BM25).
            query_vector: Dense embedding of the query (for vector search).
            principal: The caller; only chunks they may read are returned.
            top_k: Number of results to return after fusion.
            source_in: Optional sources to narrow to. Never widens access.

        Returns:
            List of result dicts sorted by descending RRF score.
        """
        bm25_results: list[dict[str, Any]] = []
        vector_results: list[dict[str, Any]] = []

        index = self._bm25_by_tenant.get(principal.tenant_id)
        if index is not None:
            hits = index.retrieve(
                query_text,
                top_k=top_k,
                predicate=lambda doc: self._permitted(doc, principal, source_in),
            )
            bm25_results = self._revalidate(hits, principal, source_in)

        # An empty vector is a lexical-only query, not a zero-dimension search:
        # forwarding it would have the store reject the whole request.
        if self._vector_retriever is not None and query_vector:
            vector_results = self._vector_retriever.retrieve(
                query_vector, principal, top_k=top_k, source_in=source_in
            )

        return self._rrf_fuse(bm25_results, vector_results, top_k)

    # ── private helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _permitted(
        doc: dict[str, Any], principal: Principal, source_in: list[str] | None
    ) -> bool:
        metadata = doc.get("metadata") or {}
        if not is_authorized(metadata, principal):
            return False
        return not source_in or metadata.get("source") in source_in

    def _revalidate(
        self,
        hits: list[dict[str, Any]],
        principal: Principal,
        source_in: list[str] | None,
    ) -> list[dict[str, Any]]:
        """Re-check lexical hits against the store's *current* payloads.

        The BM25 index is a snapshot. Without this, a document whose ACL was
        just narrowed or that was just deleted would keep matching until the
        next rebuild. Fails closed: if the store cannot be read, the lexical
        branch contributes nothing for this request.
        """
        if self._vector_db is None or not hits:
            return hits
        ids = [h["id"] for h in hits if h.get("id")]
        try:
            fresh = self._vector_db.fetch_payloads(ids)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not re-validate lexical hits (%s); dropping them.", exc)
            return []

        checked = []
        for hit in hits:
            payload = fresh.get(hit.get("id", ""))
            if payload is None:  # deleted since the index was built, or no id
                continue
            payload = dict(payload)
            payload.pop("text", None)
            current = dict(hit, metadata=payload)
            if self._permitted(current, principal, source_in):
                checked.append(current)
        return checked

    def _rrf_fuse(
        self,
        bm25_results: list[dict[str, Any]],
        vector_results: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Merge two ranked lists using Reciprocal Rank Fusion."""
        scores: dict[str, float] = {}
        docs: dict[str, dict[str, Any]] = {}

        for rank, doc in enumerate(bm25_results, start=1):
            key = self._doc_key(doc)
            scores[key] = scores.get(key, 0.0) + self.bm25_weight / (self.rrf_k + rank)
            self._keep(docs, key, doc)

        for rank, doc in enumerate(vector_results, start=1):
            key = self._doc_key(doc)
            scores[key] = scores.get(key, 0.0) + self.vector_weight / (self.rrf_k + rank)
            self._keep(docs, key, doc)

        sorted_keys = sorted(scores, key=lambda k: scores[k], reverse=True)[:top_k]
        results = []
        for key in sorted_keys:
            doc = dict(docs[key])
            doc["hybrid_score"] = scores[key]
            results.append(doc)
        return results

    @staticmethod
    def _keep(
        docs: dict[str, dict[str, Any]],
        key: str,
        doc: dict[str, Any],
    ) -> None:
        """Record *doc* under *key*, merging rather than overwriting.

        A chunk found by both branches arrives twice. The first sighting is the
        better-ranked one, so it wins the payload; the second only contributes
        whichever branch score the first was missing.
        """
        existing = docs.get(key)
        if existing is None:
            docs[key] = dict(doc)
            return
        for score_key in ("bm25_score", "vector_score", "web_score"):
            if score_key in doc and score_key not in existing:
                existing[score_key] = doc[score_key]

    @staticmethod
    def _doc_key(doc: dict[str, Any]) -> str:
        """Identify a single *chunk*.

        Every chunk cut from one document carries that document's ``doc_id``, so
        keying on ``doc_id`` alone silently folds a whole document's chunks into
        one entry and discards the rest. ``chunk_index`` is what separates them.
        """
        meta = doc.get("metadata", {})
        doc_id = meta.get("doc_id") or doc.get("text", "")[:64]
        chunk_index = meta.get("chunk_index")
        if chunk_index is None:
            return str(doc_id)
        return f"{doc_id}:{chunk_index}"
