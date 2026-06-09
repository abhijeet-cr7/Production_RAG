"""Embedding worker: consumes chunks, generates embeddings, writes to vector DB."""

from __future__ import annotations

import logging
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
        for chunk in self.consumer.consume():
            batch.append(chunk)
            if len(batch) >= self.batch_size:
                self._process_batch(batch)
                batch = []

    def _process_batch(self, batch: list[dict[str, Any]]) -> None:
        texts = [c["text"] for c in batch]
        try:
            vectors = self.embedder.embed(texts)
        except Exception as exc:  # noqa: BLE001
            logger.error("Embedding failed for batch: %s", exc)
            return

        points = [
            {
                "id": chunk["metadata"].get("doc_id", str(i)),
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
