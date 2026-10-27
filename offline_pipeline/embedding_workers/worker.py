"""Embedding worker: consumes chunks, generates embeddings, writes to vector DB."""

from __future__ import annotations

import logging
import uuid
from typing import Any

logger = logging.getLogger(__name__)


class EmbeddingWorker:
    """Pull chunks from Kafka, embed them, and upsert into the vector store.

    Args:
        consumer: A :class:`~offline_pipeline.kafka.consumer.ChunkConsumer`.
        embedder: Any object with an ``embed(texts)`` method returning a
            list of float vectors.
        vector_db: A :class:`~vector_db.client.VectorDBClient`.
        batch_size: Number of chunks to embed and upsert in a single batch.
    """

    def __init__(
        self,
        consumer: Any,
        embedder: Any,
        vector_db: Any,
        batch_size: int = 32,
    ) -> None:
        self.consumer = consumer
        self.embedder = embedder
        self.vector_db = vector_db
        self.batch_size = batch_size

    # ── public API ───────────────────────────────────────────────────────────

    def run(self) -> None:
        """Start the worker loop (blocking).

        Reads chunks in batches, embeds them, and upserts into the vector DB.
        """
        batch: list[dict[str, Any]] = []
        try:
            for chunk in self.consumer.consume():
                batch.append(chunk)
                if len(batch) >= self.batch_size:
                    self._process_batch(batch)
                    batch = []
        finally:
            # Without this the last partial batch is held forever: a document
            # whose chunk count is not a multiple of batch_size never lands.
            if batch:
                self._process_batch(batch)

    def _process_batch(self, batch: list[dict[str, Any]]) -> None:
        texts = [c["text"] for c in batch]
        try:
            vectors = self.embedder.embed(texts)
        except Exception as exc:  # noqa: BLE001
            logger.error("Embedding failed for batch: %s", exc)
            return

        points = [
            {
                "id": self._point_id(chunk["metadata"], i),
                "vector": vector,
                "payload": chunk["metadata"],
                "text": chunk["text"],
            }
            for i, (chunk, vector) in enumerate(zip(batch, vectors))
        ]
        try:
            self.vector_db.upsert(points)
            logger.info("Upserted %d points to vector DB.", len(points))
        except Exception as exc:  # noqa: BLE001
            logger.error("Vector DB upsert failed: %s", exc)

    @staticmethod
    def _point_id(metadata: dict[str, Any], fallback_index: int) -> str:
        """Return a stable, per-chunk point ID.

        Keyed on ``doc_id`` alone every chunk of a document collides on one ID
        and each upsert overwrites the last, so only the final chunk survives.
        Mixing in ``chunk_index`` keeps them distinct, and UUID5 keeps the ID
        deterministic so re-ingesting a document updates in place instead of
        duplicating. Matches the scheme used by the synchronous ingest route.
        """
        doc_id = metadata.get("doc_id", "")
        chunk_index = metadata.get("chunk_index", fallback_index)
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}_{chunk_index}"))
