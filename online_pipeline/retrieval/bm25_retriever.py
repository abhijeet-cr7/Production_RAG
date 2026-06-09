"""BM25 retriever: sparse lexical retrieval over the document corpus."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class BM25Retriever:
    """Retrieve documents using BM25 ranking.

    Args:
        corpus: List of dicts, each with at minimum a ``text`` key.
    """

    def __init__(self, corpus: list[dict[str, Any]]) -> None:
        self.corpus = corpus
        self._bm25 = self._build_index(corpus)

    # ── public API ───────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 20) -> list[dict[str, Any]]:
        """Return the top-*k* documents scored by BM25.

        Args:
            query: The tokenised search query.
            top_k: Maximum number of results to return.

        Returns:
            List of corpus dicts enriched with a ``bm25_score`` key,
            sorted descending by score.
        """
        if self._bm25 is None:
            return []

        tokens = query.lower().split()
        scores: list[float] = self._bm25.get_scores(tokens).tolist()

        ranked = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )[:top_k]

        results = []
        for idx, score in ranked:
            if score <= 0:
                continue
            doc = dict(self.corpus[idx])
            doc["bm25_score"] = score
            results.append(doc)

        return results

    # ── private helpers ──────────────────────────────────────────────────────

    def _build_index(self, corpus: list[dict[str, Any]]) -> Any:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise ImportError("rank-bm25 is required for BM25Retriever.") from exc

        tokenised = [doc["text"].lower().split() for doc in corpus]
        return BM25Okapi(tokenised)
