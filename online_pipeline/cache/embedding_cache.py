"""Redis-backed embedding cache.

Caches query embedding vectors in Redis to avoid redundant API calls.
The cache key is derived from the embedding model identity + query text;
values are stored as JSON-serialised float lists.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)

# Embedding backends are expensive to construct: a local SentenceTransformer
# reads its weights from disk, and the hosted SDKs open a connection pool.
# Build each one once per process and reuse it for the life of the process.
_BACKEND_LOCK = threading.Lock()
_BACKENDS: dict[str, Any] = {}


def _backend(key: str, factory: Any) -> Any:
    """Return the process-wide backend for *key*, constructing it at most once."""
    backend = _BACKENDS.get(key)
    if backend is not None:
        return backend
    with _BACKEND_LOCK:
        # Re-check inside the lock: another thread may have won the race.
        backend = _BACKENDS.get(key)
        if backend is None:
            backend = factory()
            _BACKENDS[key] = backend
    return backend


def reset_embedding_backends() -> None:
    """Drop every cached backend. Intended for tests and model hot-swaps."""
    with _BACKEND_LOCK:
        _BACKENDS.clear()


def embed_texts(texts: list[str], is_query: bool = False) -> list[list[float]]:
    """Embed a batch of *texts* using the configured embedding provider.

    Batching matters on the ingestion path: one call for a whole document is
    orders of magnitude cheaper than one call per chunk, for local models and
    hosted APIs alike.

    Args:
        texts: Texts to embed.
        is_query: True when embedding a search query rather than a document.
            Providers that distinguish the two score noticeably better when
            told which side of the pair they are encoding.

    Returns:
        One vector per input text, in the same order.
    """
    from config.settings import settings

    if not texts:
        return []

    provider = settings.embedding_provider

    if provider == "openai":
        from openai import OpenAI

        client = _backend("openai", lambda: OpenAI(api_key=settings.openai_api_key))
        response = client.embeddings.create(model=settings.embedding_model, input=texts)
        # The API does not guarantee ordering; sort by the echoed index.
        return [item.embedding for item in sorted(response.data, key=lambda d: d.index)]

    if provider == "cohere":
        import cohere

        client = _backend("cohere", lambda: cohere.ClientV2(api_key=settings.cohere_api_key))
        response = client.embed(
            texts=texts,
            model=settings.embedding_model,
            input_type="search_query" if is_query else "search_document",
            embedding_types=["float"],
        )
        return list(response.embeddings.float)

    if provider == "gemini":
        import google.generativeai as genai

        _backend("gemini", lambda: genai.configure(api_key=settings.gemini_api_key) or True)
        response = genai.embed_content(
            model=f"models/{settings.embedding_model}",
            content=texts,
            task_type="retrieval_query" if is_query else "retrieval_document",
        )
        embeddings = response["embedding"]
        # A single-item batch may come back unwrapped.
        if embeddings and not isinstance(embeddings[0], list):
            embeddings = [embeddings]
        return list(embeddings)

    # Default: sentence-transformers (local, free).
    def _load() -> Any:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(settings.embedding_model)

    model = _backend(f"st:{settings.embedding_model}", _load)
    vectors = model.encode(texts, batch_size=32, show_progress_bar=False)
    return [vector.tolist() for vector in vectors]


def embed_text(text: str) -> list[float]:
    """Embed a single search query.

    Respects ``settings.embedding_provider``:
    - ``"sentence-transformers"`` — local, no API key required (default)
    - ``"openai"``  — requires ``OPENAI_API_KEY``
    - ``"cohere"``  — requires ``COHERE_API_KEY``
    - ``"gemini"``  — requires ``GEMINI_API_KEY``
    """
    return embed_texts([text], is_query=True)[0]


class EmbeddingCache:
    """Store and retrieve embedding vectors in Redis.

    Args:
        redis_url: Redis connection URL (e.g. ``redis://localhost:6379/0``).
        ttl: Time-to-live for cached entries in seconds.
    """

    def __init__(
        self,
        redis_url: str,
        ttl: int = 86400,
        model_id: str | None = None,
    ) -> None:
        self.ttl = ttl
        self.model_id = model_id or self._default_model_id()
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

    def _make_key(self, query: str) -> str:
        digest = hashlib.sha256(query.encode()).hexdigest()
        return f"emb:{self.model_id}:{digest}"

    @staticmethod
    def _default_model_id() -> str:
        from config.settings import settings

        # Scope cache entries by embedding space; swapping models/providers
        # naturally creates a separate namespace to avoid stale vector reuse.
        raw = f"{settings.embedding_provider}:{settings.embedding_model}"
        return "".join(ch if ch.isalnum() or ch in "-._:" else "_" for ch in raw)

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
