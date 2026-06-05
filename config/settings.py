"""Centralised application settings backed by environment variables."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── LLM ─────────────────────────────────────────────────────────────────
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    # ── Embeddings ───────────────────────────────────────────────────────────
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536

    # ── Kafka ────────────────────────────────────────────────────────────────
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_topic_chunks: str = "rag.chunks"
    kafka_consumer_group: str = "embedding-workers"

    # ── Redis ────────────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"
    redis_embedding_cache_ttl: int = 86400  # seconds

    # ── Vector DB (Qdrant) ───────────────────────────────────────────────────
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "production_rag"
    qdrant_api_key: str = ""

    # ── API Gateway ──────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # ── Retrieval ────────────────────────────────────────────────────────────
    top_k_retrieval: int = 20
    top_k_rerank: int = 5
    bm25_weight: float = 0.3
    vector_weight: float = 0.7

    # ── Chunking ─────────────────────────────────────────────────────────────
    chunk_size: int = 512
    chunk_overlap: int = 64


settings = Settings()
