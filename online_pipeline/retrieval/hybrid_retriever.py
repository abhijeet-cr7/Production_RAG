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
            from online_pipeline.retrieval.bm25_retriever import BM25Retriever
            self._bm25_retriever = BM25Retriever(corpus)

    # ── public API ───────────────────────────────────────────────────────────

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

        if self._vector_retriever is not None:
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
            docs[key] = doc

        for rank, doc in enumerate(vector_results, start=1):
            key = self._doc_key(doc)
            scores[key] = scores.get(key, 0.0) + self.vector_weight / (self.rrf_k + rank)
            docs[key] = doc

        sorted_keys = sorted(scores, key=lambda k: scores[k], reverse=True)[:top_k]
        results = []
        for key in sorted_keys:
            doc = dict(docs[key])
            doc["hybrid_score"] = scores[key]
            results.append(doc)
        return results

    @staticmethod
    def _doc_key(doc: dict[str, Any]) -> str:
        meta = doc.get("metadata", {})
        return meta.get("doc_id") or doc.get("text", "")[:64]
