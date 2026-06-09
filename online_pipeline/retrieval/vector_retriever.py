"""Vector retriever: dense ANN search against the Qdrant collection."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class VectorRetriever:
    """Retrieve documents from a Qdrant collection via vector similarity.

    Args:
        vector_db: A :class:`~vector_db.client.VectorDBClient`.
    """

    def __init__(self, vector_db: Any) -> None:
        self.vector_db = vector_db

    # ── public API ───────────────────────────────────────────────────────────

    def retrieve(
        self,
        query_vector: list[float],
        top_k: int = 20,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Return *top_k* nearest neighbours for *query_vector*.

        Args:
            query_vector: Dense embedding of the query.
            top_k: Number of results to return.
            metadata_filter: Optional payload filter forwarded to Qdrant.

        Returns:
            List of result dicts with keys ``text``, ``metadata``, and
            ``vector_score``.
        """
        results = self.vector_db.search(
            vector=query_vector,
            top_k=top_k,
            metadata_filter=metadata_filter,
        )
        for r in results:
            r.setdefault("vector_score", r.pop("score", 0.0))
        return results
