"""Vector retriever: dense ANN search against the Qdrant collection."""

from __future__ import annotations

import logging
from typing import Any

from online_pipeline.auth.principal import Principal

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
        principal: Principal,
        top_k: int = 20,
        source_in: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Return *top_k* nearest neighbours for *query_vector*.

        Args:
            query_vector: Dense embedding of the query.
            top_k: Number of results to return.
            principal: Caller whose entitlements filter the search (required).
            source_in: Optional sources to narrow to; never widens access.

        Returns:
            List of result dicts with keys ``text``, ``metadata``, and
            ``vector_score``.
        """
        results = self.vector_db.search(
            vector=query_vector,
            top_k=top_k,
            principal=principal,
            source_in=source_in,
        )
        for r in results:
            r.setdefault("vector_score", r.pop("score", 0.0))
        return results
