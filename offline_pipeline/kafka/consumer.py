"""Kafka consumer: reads chunks from the topic and yields them for embedding."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from typing import Any

logger = logging.getLogger(__name__)


class ChunkConsumer:
    """Consume chunk dicts from a Kafka topic.

    Args:
        bootstrap_servers: Comma-separated Kafka broker addresses.
        topic: Source topic name.
        group_id: Consumer group identifier.
        poll_timeout: Seconds to block waiting for a message.
    """

    def __init__(
        self,
        bootstrap_servers: str,
        topic: str,
        group_id: str,
        poll_timeout: float = 1.0,
    ) -> None:
        self.topic = topic
        self.poll_timeout = poll_timeout
        self._consumer = self._build_consumer(bootstrap_servers, group_id, topic)

    # ── public API ───────────────────────────────────────────────────────────

    def consume(self) -> Iterator[dict[str, Any]]:
        """Continuously yield deserialized chunk dicts from the topic.

        This is a blocking infinite iterator; run it in a dedicated thread
        or process.
        """
        try:
            while True:
                msg = self._consumer.poll(self.poll_timeout)
                if msg is None:
                    continue
                if msg.error():
                    logger.error("Consumer error: %s", msg.error())
                    continue
                try:
                    yield json.loads(msg.value().decode("utf-8"))
                except json.JSONDecodeError as exc:
                    logger.warning("Failed to decode message: %s", exc)
        finally:
            self._consumer.close()

    # ── private helpers ──────────────────────────────────────────────────────

    def _build_consumer(
        self, bootstrap_servers: str, group_id: str, topic: str
    ) -> Any:
        try:
            from confluent_kafka import Consumer
        except ImportError as exc:
            raise ImportError(
                "confluent-kafka is required for the Kafka consumer."
            ) from exc

        consumer = Consumer(
            {
                "bootstrap.servers": bootstrap_servers,
                "group.id": group_id,
                "auto.offset.reset": "earliest",
            }
        )
        consumer.subscribe([topic])
        return consumer
