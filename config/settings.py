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
    # Free alternatives
    groq_api_key: str = ""          # https://console.groq.com
    gemini_api_key: str = ""         # https://aistudio.google.com
    cohere_api_key: str = ""         # https://dashboard.cohere.com
    mistral_api_key: str = ""        # https://console.mistral.ai
    llm_routing_mode: str = "balanced"  # "cost_optimized" | "balanced" | "quality"
    query_rewrite_provider: str = "groq"
    query_rewrite_model: str = "openai/gpt-oss-20b"  # llama-3.1-8b-instant was decommissioned
    answer_generation_provider: str = "groq"
    answer_generation_model: str = ""
    llm_fallback_providers: str = "gemini,openai,anthropic"

    # ── Embeddings ───────────────────────────────────────────────────────────
    # provider: "sentence-transformers" (local/free) | "openai" | "cohere" | "gemini"
    embedding_provider: str = "sentence-transformers"
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384

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

    # BM25 lexical branch. The index is in-memory and rebuilt from the vector
    # store, so cap how much of the corpus it will pull in.
    bm25_enabled: bool = True
    bm25_max_corpus_chunks: int = 50_000

    # ── Chunking ─────────────────────────────────────────────────────────────
    chunking_strategy: str = "auto"  # "auto" | "fixed_token" | "recursive"
    chunk_size: int = 512
    chunk_overlap: int = 64

    # ── Web search (Tavily) ──────────────────────────────────────────────────
    tavily_api_key: str = ""          # https://app.tavily.com
    web_search_enabled: bool = True   # allow falling back to the web
    web_search_max_results: int = 5
    # Local hits below this count trigger a web-search top-up.
    web_search_min_local_results: int = 3
    # Cross-encoder logit below which the best local hit counts as irrelevant.
    web_search_min_relevance_score: float = 0.0
    tavily_search_depth: str = "basic"  # "basic" | "advanced"

    # ── Generation grading (hallucination + usefulness check) ───────────────
    answer_grading_enabled: bool = True
    grader_provider: str = "groq"
    grader_model: str = "openai/gpt-oss-20b"
    # Total generation attempts per request: first draft + stricter regenerations.
    max_generation_attempts: int = 2

    # ── Ingestion safety ─────────────────────────────────────────────────────
    ingest_max_upload_bytes: int = 25 * 1024 * 1024
    ingest_max_fetch_bytes: int = 10 * 1024 * 1024
    ingest_max_redirects: int = 3
    ingest_fetch_timeout_seconds: float = 30.0
    ingest_allow_http: bool = False          # dev only; production is https-only
    ingest_rate_limit_per_minute: int = 20   # URL/API ingests per user, per process
    # Untrusted files are parsed in a resource-limited subprocess.
    parser_timeout_seconds: int = 120
    parser_cpu_seconds: int = 110
    parser_max_memory_mb: int = 1024
    pdf_max_pages: int = 500
    pdf_ocr_max_pages: int = 20
    extracted_text_max_chars: int = 5_000_000

    # ── Authentication (Okta) ────────────────────────────────────────────────
    app_env: str = "dev"              # "dev" | "staging" | "prod"
    # "okta" validates Okta-issued RS256 access tokens; "dev_token" (HS256) and
    # "disabled" are honoured only when app_env == "dev".
    auth_mode: str = "okta"           # "okta" | "dev_token" | "disabled"
    okta_issuer: str = ""             # e.g. https://acme.okta.com/oauth2/default
    okta_audience: str = "api://default"
    okta_jwks_url: str = ""           # defaults to <okta_issuer>/v1/keys
    okta_scopes_claim: str = "scp"
    okta_tenant_claim: str = "tenant_id"      # custom claim on the auth server
    okta_groups_claim: str = "groups"         # custom claim on the auth server
    okta_clearance_claim: str = "clearance"   # custom claim, integer 0-3
    jwt_leeway_seconds: int = 30
    dev_jwt_secret: str = ""          # HS256 secret, dev_token mode only (>= 32 chars)
    tenant_policies_path: str = "config/tenant_policies.yaml"

    # ── Audit ────────────────────────────────────────────────────────────────
    audit_log_path: str = "logs/audit.jsonl"
    # Keys the hash chain and query fingerprints. Required outside dev.
    audit_hmac_key: str = ""
    # True: a request fails (503) when its audit event cannot be written.
    audit_fail_closed: bool = True

    # ── Observability (LangSmith) ────────────────────────────────────────────
    langsmith_tracing: bool = False
    langsmith_api_key: str = ""       # https://smith.langchain.com
    langsmith_project: str = "production-rag"
    langsmith_endpoint: str = "https://api.smith.langchain.com"


settings = Settings()
