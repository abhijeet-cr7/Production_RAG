"""Qdrant vector database client.

Wraps the Qdrant Python SDK to provide a simple upsert / search interface
used by both the offline embedding workers and the online retrieval pipeline.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from config.settings import settings

logger = logging.getLogger(__name__)


class VectorDBClient:
    """Thin wrapper around the Qdrant client.

    Args:
        url: Qdrant service URL.
        collection: Name of the collection to operate on.
        api_key: Optional Qdrant API key.
        dimension: Embedding vector dimension.
    """

    def __init__(
        self,
        url: str | None = None,
        collection: str | None = None,
        api_key: str | None = None,
        dimension: int | None = None,
    ) -> None:
        self.url = url or settings.qdrant_url
        self.collection = collection or settings.qdrant_collection
        self.api_key = api_key or settings.qdrant_api_key or None
        self.dimension = dimension or settings.embedding_dimension
        self._client = self._connect()

    # ── public API ───────────────────────────────────────────────────────────

    def ensure_collection(self) -> None:
        """Create the Qdrant collection if it does not already exist."""
        from qdrant_client.models import Distance, VectorParams

        existing = {c.name for c in self._client.get_collections().collections}
        if self.collection not in existing:
            self._client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=self.dimension, distance=Distance.COSINE
                ),
            )
            logger.info("Created Qdrant collection '%s'.", self.collection)

    def upsert(self, points: list[dict[str, Any]]) -> None:
        """Upsert *points* into the collection.

        Each point dict must have:
            - ``vector`` (list[float])
            - ``payload`` (dict)  — metadata attached to the point
            - ``text`` (str)      — stored inside the payload as ``text``
            - ``id`` (str, optional) — defaults to a new UUID
        """
        from qdrant_client.models import PointStruct

        qdrant_points = []
        for p in points:
            payload = dict(p.get("payload", {}))
            payload["text"] = p.get("text", "")
            qdrant_points.append(
                PointStruct(
                    id=str(p.get("id") or uuid.uuid4()),
                    vector=p["vector"],
                    payload=payload,
                )
            )
        self._client.upsert(
            collection_name=self.collection, points=qdrant_points
        )

    def search(
        self,
        vector: list[float],
        top_k: int = 20,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Return the *top_k* nearest neighbours for *vector*.

        Args:
            vector: Query embedding.
            top_k: Number of results.
            metadata_filter: Optional Qdrant filter dict (e.g.
                ``{"must": [{"key": "source", "match": {"value": "x"}}]}``).

        Returns:
            List of dicts with keys ``text``, ``metadata``, and ``score``.
        """
        from qdrant_client.models import Filter

        qdrant_filter = Filter(**metadata_filter) if metadata_filter else None

        hits = self._client.search(
            collection_name=self.collection,
            query_vector=vector,
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
        )
        results = []
        for hit in hits:
            payload = dict(hit.payload or {})
            text = payload.pop("text", "")
            results.append(
                {"text": text, "metadata": payload, "score": hit.score}
            )
        return results

    # ── private helpers ──────────────────────────────────────────────────────

    def _connect(self) -> Any:
        try:
            from qdrant_client import QdrantClient

            return QdrantClient(
                url=self.url,
                api_key=self.api_key,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to connect to Qdrant: %s", exc)
            raise
