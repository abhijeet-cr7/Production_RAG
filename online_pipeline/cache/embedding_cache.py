"""Redis-backed embedding cache.

Caches query embedding vectors in Redis to avoid redundant API calls.
The cache key is derived from the query text; values are stored as
JSON-serialised float lists.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def embed_text(text: str) -> list[float]:
    """Embed *text* using the configured OpenAI embedding model.

    This is a module-level convenience used by the gateway and tests.
    """
    from config.settings import settings
    from openai import OpenAI

    client = OpenAI()
    response = client.embeddings.create(
        model=settings.embedding_model,
        input=text,
    )
    return response.data[0].embedding


class EmbeddingCache:
    """Store and retrieve embedding vectors in Redis.

    Args:
        redis_url: Redis connection URL (e.g. ``redis://localhost:6379/0``).
        ttl: Time-to-live for cached entries in seconds.
    """

    def __init__(self, redis_url: str, ttl: int = 86400) -> None:
        self.ttl = ttl
        self._redis = self._connect(redis_url)

    # ── public API ───────────────────────────────────────────────────────────

    def get(self, query: str) -> list[float] | None:
        """Return the cached embedding for *query*, or ``None`` if absent."""
        if self._redis is None:
            return None
        key = self._make_key(query)
        raw = self._redis.get(key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    def set(self, query: str, embedding: list[float]) -> None:
        """Cache *embedding* for *query* with the configured TTL."""
        if self._redis is None:
            return
        key = self._make_key(query)
        self._redis.setex(key, self.ttl, json.dumps(embedding))

    # ── private helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _make_key(query: str) -> str:
        digest = hashlib.sha256(query.encode()).hexdigest()
        return f"emb:{digest}"

    @staticmethod
    def _connect(redis_url: str) -> Any:
        try:
            import redis

            client = redis.from_url(redis_url, decode_responses=True)
            client.ping()
            return client
        except Exception as exc:  # noqa: BLE001
            logger.warning("Redis unavailable (%s); cache disabled.", exc)
            return None
