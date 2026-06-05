"""Cross-encoder reranker: re-scores retrieved candidates for final ordering."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    """Re-score candidate documents with a cross-encoder model.

    Falls back to the original retrieval order when the model is unavailable.

    Args:
        model_name: HuggingFace cross-encoder model identifier.
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL) -> None:
        self.model_name = model_name
        self._model = self._load_model()

    # ── public API ───────────────────────────────────────────────────────────

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Return *top_k* candidates sorted by cross-encoder relevance score.

        Args:
            query: The (rewritten) user query.
            candidates: Documents from the hybrid retriever.
            top_k: Number of results to return.

        Returns:
            List of candidate dicts enriched with a ``rerank_score`` key,
            sorted descending.
        """
        if not candidates:
            return []
        if self._model is None:
            return candidates[:top_k]

        pairs = [(query, c["text"]) for c in candidates]
        try:
            scores: list[float] = self._model.predict(pairs).tolist()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Reranking failed (%s); using retrieval order.", exc)
            return candidates[:top_k]

        ranked = sorted(
            zip(candidates, scores), key=lambda x: x[1], reverse=True
        )[:top_k]

        results = []
        for doc, score in ranked:
            doc = dict(doc)
            doc["rerank_score"] = score
            results.append(doc)
        return results

    # ── private helpers ──────────────────────────────────────────────────────

    def _load_model(self) -> Any:
        try:
            from sentence_transformers import CrossEncoder
            return CrossEncoder(self.model_name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load reranker model (%s).", exc)
            return None
