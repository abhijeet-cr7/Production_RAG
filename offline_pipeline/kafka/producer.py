"""Kafka producer: publishes document chunks to the configured topic."""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class ChunkProducer:
    """Publish chunk dicts to a Kafka topic.

    Args:
        bootstrap_servers: Comma-separated Kafka broker addresses.
        topic: Destination topic name.
    """

    def __init__(self, bootstrap_servers: str, topic: str) -> None:
        self.topic = topic
        self._producer = self._build_producer(bootstrap_servers)

    # ── public API ───────────────────────────────────────────────────────────

    def send(self, chunk: dict[str, Any]) -> None:
        """Serialise *chunk* to JSON and publish it to the Kafka topic.

        Args:
            chunk: A chunk dict produced by :class:`~offline_pipeline.chunkers.text_chunker.TextChunker`.
        """
        payload = json.dumps(chunk, ensure_ascii=False).encode("utf-8")
        self._producer.produce(
            self.topic,
            value=payload,
            callback=self._delivery_callback,
        )
        self._producer.poll(0)

    def flush(self) -> None:
        """Block until all pending messages are delivered."""
        self._producer.flush()

    # ── private helpers ──────────────────────────────────────────────────────

    def _build_producer(self, bootstrap_servers: str) -> Any:
        try:
            from confluent_kafka import Producer
        except ImportError as exc:
            raise ImportError(
                "confluent-kafka is required for the Kafka producer."
            ) from exc

        return Producer({"bootstrap.servers": bootstrap_servers})

    @staticmethod
    def _delivery_callback(err: Any, msg: Any) -> None:
        if err:
            logger.error("Kafka delivery failed: %s", err)
        else:
            logger.debug(
                "Message delivered to %s [%s]", msg.topic(), msg.partition()
            )
