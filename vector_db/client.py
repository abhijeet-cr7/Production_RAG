"""Qdrant vector database client.

Wraps the Qdrant Python SDK to provide a simple upsert / search interface
used by both the offline embedding workers and the online retrieval pipeline.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from config.settings import settings
from online_pipeline.auth.acl import build_qdrant_filter
from online_pipeline.auth.principal import Principal

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
        self._ensure_payload_indexes()

    # Fields every entitlement filter touches. Indexing them keeps filtered
    # search fast and is idempotent, so it also upgrades existing collections.
    _PAYLOAD_INDEXES = {
        "tenant_id": "KEYWORD",
        "owner_id": "KEYWORD",
        "acl_groups": "KEYWORD",
        "doc_id": "KEYWORD",
        "source": "KEYWORD",
        "classification": "INTEGER",
        "acl_public": "BOOL",
    }

    def _ensure_payload_indexes(self) -> None:
        from qdrant_client.models import PayloadSchemaType

        for field, schema in self._PAYLOAD_INDEXES.items():
            try:
                self._client.create_payload_index(
                    collection_name=self.collection,
                    field_name=field,
                    field_schema=getattr(PayloadSchemaType, schema),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not create payload index '%s': %s", field, exc)

    def collection_for_tenant(self, tenant_id: str) -> str:
        """Collection that holds *tenant_id*'s data.

        One shared collection today, isolated by the ``tenant_id`` payload
        filter. Returning a per-tenant name here (and provisioning it) is the
        single change needed to move to collection-per-tenant isolation.
        """
        return self.collection

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
        top_k: int,
        principal: Principal,
        source_in: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Return the *top_k* nearest neighbours *principal* is entitled to read.

        The entitlement filter is built here from the principal, never accepted
        from a caller, and ``principal`` has no default: no code path can run an
        unfiltered search. ``source_in`` can only narrow the result further.

        Returns:
            List of dicts with keys ``text``, ``metadata``, and ``score``.
        """
        hits = self._client.search(
            collection_name=self.collection_for_tenant(principal.tenant_id),
            query_vector=vector,
            limit=top_k,
            query_filter=build_qdrant_filter(principal, source_in),
            with_payload=True,
        )
        results = []
        for hit in hits:
            payload = dict(hit.payload or {})
            text = payload.pop("text", "")
            results.append(
                {"id": str(hit.id), "text": text, "metadata": payload, "score": hit.score}
            )
        return results

    def fetch_payloads(self, point_ids: list[str]) -> dict[str, dict[str, Any]]:
        """Current payload for each existing point; deleted points are absent.

        The lexical index is a snapshot, so its hits are re-checked against this
        authoritative copy: a revoked ACL or a deleted document stops matching
        immediately instead of after the next index rebuild.
        """
        if not point_ids:
            return {}
        points = self._client.retrieve(
            collection_name=self.collection,
            ids=point_ids,
            with_payload=True,
            with_vectors=False,
        )
        return {str(p.id): dict(p.payload or {}) for p in points}

    def scroll_all(self, limit: int = 50_000, page_size: int = 256) -> list[dict[str, Any]]:
        """Read up to *limit* stored chunks back out of the collection.

        BM25 needs the lexical corpus in memory, and the vector store is the
        only place the chunk text lives. Paged so a large collection does not
        arrive in one response.

        Returns:
            List of dicts with keys ``id``, ``text`` and ``metadata``.
        """
        records: list[dict[str, Any]] = []
        offset: Any = None
        while len(records) < limit:
            batch, offset = self._client.scroll(
                collection_name=self.collection,
                limit=min(page_size, limit - len(records)),
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in batch:
                payload = dict(point.payload or {})
                text = payload.pop("text", "")
                if text:
                    records.append({"id": str(point.id), "text": text, "metadata": payload})
            if offset is None or not batch:
                break
        return records

    # ── private helpers ──────────────────────────────────────────────────────

    def _connect(self) -> Any:
        try:
            from qdrant_client import QdrantClient

            if self.url == ":memory:":  # in-process engine, used by tests
                return QdrantClient(":memory:")
            return QdrantClient(
                url=self.url,
                api_key=self.api_key,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to connect to Qdrant: %s", exc)
            raise
