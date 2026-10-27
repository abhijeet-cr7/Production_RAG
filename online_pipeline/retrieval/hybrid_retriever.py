"""Hybrid retriever: fuses BM25 and vector search via Reciprocal Rank Fusion."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Combine BM25 and vector search scores using Reciprocal Rank Fusion (RRF).

    Args:
        vector_db: A :class:`~vector_db.client.VectorDBClient` (used for
            the vector branch).
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

        self._vector_retriever = None
        self._bm25_retriever = None

        if vector_db is not None:
            from online_pipeline.retrieval.vector_retriever import VectorRetriever
            self._vector_retriever = VectorRetriever(vector_db)

        if corpus:
            self.load_corpus(corpus)

    # ── public API ───────────────────────────────────────────────────────────

    def load_corpus(self, corpus: list[dict[str, Any]]) -> None:
        """Build (or rebuild) the BM25 index over *corpus*.

        The lexical branch is only live once this has been called with a
        non-empty corpus; until then ``retrieve`` is dense-only. Call it again
        after ingesting to bring new chunks into lexical range.

        Note: this index is held in memory and scores every document per query.
        That is fine up to tens of thousands of chunks; past that the lexical
        branch belongs in the store as sparse vectors.
        """
        if not corpus:
            self._bm25_retriever = None
            return
        from online_pipeline.retrieval.bm25_retriever import BM25Retriever

        try:
            self._bm25_retriever = BM25Retriever(corpus)
            logger.info("BM25 index built over %d chunks.", len(corpus))
        except Exception as exc:  # noqa: BLE001
            # A missing rank-bm25 install must not take the dense branch down.
            logger.warning("BM25 index unavailable (%s); retrieval is dense-only.", exc)
            self._bm25_retriever = None

    @property
    def lexical_enabled(self) -> bool:
        """True when the BM25 branch is live and will contribute to fusion."""
        return self._bm25_retriever is not None

    def retrieve(
        self,
        query_text: str,
        query_vector: list[float],
        top_k: int = 20,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Run hybrid retrieval and return fused, de-duplicated results.

        Args:
            query_text: Rewritten query text (for BM25).
            query_vector: Dense embedding of the query (for vector search).
            top_k: Number of results to return after fusion.
            metadata_filter: Optional payload filter for the vector branch.

        Returns:
            List of result dicts sorted by descending RRF score.
        """
        bm25_results: list[dict[str, Any]] = []
        vector_results: list[dict[str, Any]] = []

        if self._bm25_retriever is not None:
            bm25_results = self._bm25_retriever.retrieve(query_text, top_k=top_k)

        # An empty vector is a lexical-only query, not a zero-dimension search:
        # forwarding it would have the store reject the whole request.
        if self._vector_retriever is not None and query_vector:
            vector_results = self._vector_retriever.retrieve(
                query_vector, top_k=top_k, metadata_filter=metadata_filter
            )

        return self._rrf_fuse(bm25_results, vector_results, top_k)

    # ── private helpers ──────────────────────────────────────────────────────

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
