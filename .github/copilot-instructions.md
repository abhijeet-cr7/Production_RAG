# GitHub Copilot Instructions — Production RAG System

This file gives Copilot (and any agent) full context on the codebase: architecture, code flow, every module's purpose, all public functions, and key variables.

---

## 1. High-Level Architecture

Two coordinated pipelines share config, the vector DB client, and the embedding function.

```
┌─────────────────────────────────────────────────────────┐
│                    OFFLINE PIPELINE                     │
│  File/URL/API → Extract → Clean → Chunk → Kafka →      │
│  EmbeddingWorker → embed_text() → VectorDBClient.upsert │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                    ONLINE PIPELINE                      │
│  HTTP POST /query                                       │
│    → QueryRewriter.rewrite()                            │
│    → EmbeddingCache.get() / embed_text()                │
│    → HybridRetriever.retrieve()  (BM25 + Qdrant)        │
│    → Reranker.rerank()                                  │
│    → ContextBuilder.build()                             │
│    → LLMClient.generate()                               │
│    → QueryResponse                                      │
│                                                         │
│  HTTP POST /ingest                                      │
│    → DocumentExtractor.extract()                        │
│    → TextCleaner.clean()                                │
│    → MetadataExtractor.enrich()                         │
│    → TextChunker.chunk()                                │
│    → embed_text() per chunk                             │
│    → VectorDBClient.upsert()                            │
│    → IngestResponse                                     │
└─────────────────────────────────────────────────────────┘
```

---

## 2. Infrastructure (docker-compose.yml)

| Service | Image | Port | Purpose |
|---|---|---|---|
| zookeeper | confluentinc/cp-zookeeper:7.6.1 | 2181 | Kafka coordinator |
| kafka | confluentinc/cp-kafka:7.6.1 | 9092 | Message queue for chunks |
| redis | redis:7.2-alpine | 6379 | Embedding cache (TTL-backed) |
| qdrant | qdrant/qdrant:v1.9.2 | 6333, 6334 | Vector store (cosine similarity) |

---

## 3. Configuration — `config/settings.py`

Single `Settings` class (pydantic-settings, reads from `.env`).

| Variable | Type | Default | Purpose |
|---|---|---|---|
| `openai_api_key` | str | `""` | OpenAI (paid) |
| `anthropic_api_key` | str | `""` | Anthropic (paid) |
| `groq_api_key` | str | `""` | Groq free LLM — **primary LLM provider** |
| `gemini_api_key` | str | `""` | Google Gemini free LLM / embeddings |
| `cohere_api_key` | str | `""` | Cohere free LLM / embeddings |
| `mistral_api_key` | str | `""` | Mistral free LLM |
| `embedding_provider` | str | `"sentence-transformers"` | Embedding backend |
| `embedding_model` | str | `"all-MiniLM-L6-v2"` | Model name passed to provider |
| `embedding_dimension` | int | `384` | Vector dimension (must match Qdrant collection) |
| `kafka_bootstrap_servers` | str | `"localhost:9092"` | Kafka address |
| `kafka_topic_chunks` | str | `"rag.chunks"` | Topic where chunks are published |
| `kafka_consumer_group` | str | `"embedding-workers"` | Kafka consumer group ID |
| `redis_url` | str | `"redis://localhost:6379/0"` | Redis connection URL |
| `redis_embedding_cache_ttl` | int | `86400` | Cache TTL in seconds (1 day) |
| `qdrant_url` | str | `"http://localhost:6333"` | Qdrant HTTP URL |
| `qdrant_collection` | str | `"production_rag"` | Qdrant collection name |
| `qdrant_api_key` | str | `""` | Qdrant auth (empty = no auth) |
| `api_host` | str | `"0.0.0.0"` | FastAPI bind address |
| `api_port` | int | `8000` | FastAPI port |
| `top_k_retrieval` | int | `20` | Candidates fetched before reranking |
| `top_k_rerank` | int | `5` | Final results returned after reranking |
| `bm25_weight` | float | `0.3` | BM25 contribution in RRF fusion |
| `vector_weight` | float | `0.7` | Vector search contribution in RRF fusion |
| `chunk_size` | int | `512` | Max tokens per chunk |
| `chunk_overlap` | int | `64` | Token overlap between consecutive chunks |

Import via: `from config.settings import settings`

---

## 4. Offline Pipeline

### 4a. Extractors

#### `offline_pipeline/extractors/document_extractor.py` — `DocumentExtractor`
Extracts text from local files. Supports `.pdf`, `.docx`, `.txt`.

| Method | Args | Returns | Notes |
|---|---|---|---|
| `extract(file_path)` | `str \| Path` | `{"text": str, "metadata": dict}` | Raises `FileNotFoundError`, `ValueError` for unsupported types. Uses `pypdf` for PDF, `python-docx` for DOCX. Falls back to Tesseract OCR for image-only PDF pages. |

Metadata keys returned: `source` (file path), `file_type`, `page_count` (PDF), `title` (DOCX).

#### `offline_pipeline/extractors/web_extractor.py` — `WebExtractor`
Scrapes a URL using `requests` + `BeautifulSoup`. Strips `<script>`, `<style>`, `<nav>`, `<footer>`, `<header>` tags.

| Method | Args | Returns |
|---|---|---|
| `extract(url)` | `str` | `{"text": str, "metadata": {"source", "file_type": "web", "title", "content_type"}}` |

#### `offline_pipeline/extractors/api_extractor.py` — `APIExtractor`
Fetches JSON from a REST endpoint via `httpx`.

| Method | Args | Returns |
|---|---|---|
| `extract(url, params, text_fields)` | `url: str`, `params: dict \| None`, `text_fields: list[str] \| None` | `{"text": str (JSON-serialised), "metadata": {"source", "file_type": "api"}}` |

`text_fields`: if provided, only those top-level JSON keys are serialised as text.

---

### 4b. Preprocessors

#### `offline_pipeline/preprocessors/cleaner.py` — `TextCleaner`
Normalises extracted text.

| Method | Notes |
|---|---|
| `clean(text: str) → str` | Runs: Unicode NFKC normalisation → control char removal (keeps `\n`, `\t`) → collapse 3+ blank lines to 2 → collapse multiple spaces |

#### `offline_pipeline/preprocessors/metadata_extractor.py` — `MetadataExtractor`
Enriches the document dict with derived metadata.

| Method | Args | Returns | Added Keys |
|---|---|---|---|
| `enrich(document: dict) → dict` | `{"text": str, "metadata": dict}` | Same shape with enriched metadata | `doc_id` (SHA-256 of text), `char_count`, `word_count`, `ingested_at` (UTC ISO-8601) |

---

### 4c. Chunker

#### `offline_pipeline/chunkers/text_chunker.py` — `TextChunker`

| Constructor arg | Default | Purpose |
|---|---|---|
| `chunk_size` | `512` | Max tokens per chunk (from settings) |
| `chunk_overlap` | `64` | Overlap tokens between chunks (from settings) |

Uses `tiktoken` (`cl100k_base` encoding) for token counting; falls back to whitespace splitting if tiktoken is unavailable.

| Method | Args | Returns |
|---|---|---|
| `chunk(text, metadata)` | `text: str`, `metadata: dict \| None` | `list[{"text": str, "metadata": dict, "chunk_index": int}]` |

Each chunk's metadata inherits the parent doc metadata plus `chunk_index` and `token_count`.

---

### 4d. Kafka

#### `offline_pipeline/kafka/producer.py` — `ChunkProducer`
Publishes chunk dicts as JSON to a Kafka topic.

| Method | Notes |
|---|---|
| `send(chunk: dict)` | Serialises to UTF-8 JSON, calls `producer.produce()` |
| `flush()` | Blocks until all pending messages are delivered |

Constructor: `ChunkProducer(bootstrap_servers: str, topic: str)`

#### `offline_pipeline/kafka/consumer.py` — `ChunkConsumer`
Consumes chunk messages from Kafka.

| Method | Returns |
|---|---|
| `consume() → Iterator[dict]` | Yields deserialized chunk dicts |

---

### 4e. Embedding Worker

#### `offline_pipeline/embedding_workers/worker.py` — `EmbeddingWorker`
Pulls chunks from Kafka, embeds them, upserts into Qdrant.

Constructor:
```python
EmbeddingWorker(consumer, embedder, vector_db, batch_size=32)
```
- `embedder`: any object with `embed(texts: list[str]) → list[list[float]]`
- `vector_db`: `VectorDBClient`

| Method | Notes |
|---|---|
| `run()` | Blocking loop. Batches chunks, calls `_process_batch` |
| `_process_batch(batch)` | Calls `embedder.embed()`, constructs point dicts, calls `vector_db.upsert()` |

---

## 5. Online Pipeline

### 5a. API Gateway — `online_pipeline/api/gateway.py`

FastAPI app with `lifespan` context that initialises all singletons at startup.

**Singletons (module-level globals):**
- `_vector_db: VectorDBClient` — shared Qdrant client; also calls `ensure_collection()` at startup
- `_rewriter: QueryRewriter`
- `_cache: EmbeddingCache`
- `_retriever: HybridRetriever(vector_db=_vector_db)`
- `_reranker: Reranker`
- `_context_builder: ContextBuilder`
- `_llm: LLMClient`

**Endpoints:**

| Method | Path | Request | Response | Description |
|---|---|---|---|---|
| GET | `/health` | — | `{"status": "ok"}` | Liveness check |
| POST | `/ingest` | `multipart/form-data` — `file` field (.pdf/.docx/.txt) | `IngestResponse` | Full offline pipeline in-process |
| POST | `/query` | `QueryRequest` JSON | `QueryResponse` JSON | Full online RAG pipeline |

**`QueryRequest` fields:**
- `query: str` — user question
- `metadata_filter: dict | None` — Qdrant payload filter (optional)
- `top_k: int` — number of final results (default: `settings.top_k_rerank`)

**`QueryResponse` fields:**
- `answer: str` — LLM-generated answer
- `sources: list[dict]` — metadata of reranked chunks used

**`IngestResponse` fields:**
- `filename: str`
- `chunks_indexed: int`
- `doc_id: str` — SHA-256 of document text

**`_ALLOWED_EXTENSIONS`**: `{".pdf", ".docx", ".txt"}`

---

### 5b. Query Rewriter — `online_pipeline/query_rewrite/rewriter.py`

`QueryRewriter` rewrites raw user queries into retrieval-optimised search queries using an LLM.

| Constructor arg | Default | Purpose |
|---|---|---|
| `model` | `"llama-3.1-8b-instant"` | Groq model used for rewriting |

| Method | Returns | Fallback |
|---|---|---|
| `rewrite(query: str) → str` | Rewritten query | Returns original `query` on any exception |

Uses **Groq** (`settings.groq_api_key`). System prompt instructs the model to return a single concise search query.

---

### 5c. Embedding Cache — `online_pipeline/cache/embedding_cache.py`

#### Module-level function: `embed_text(text: str) → list[float]`
Routes embedding to the configured provider via `settings.embedding_provider`:

| Provider value | Library used | Key required |
|---|---|---|
| `"sentence-transformers"` (default) | `sentence_transformers.SentenceTransformer` | None (local) |
| `"openai"` | `openai.OpenAI` | `OPENAI_API_KEY` |
| `"cohere"` | `cohere.ClientV2` | `COHERE_API_KEY` |
| `"gemini"` | `google.generativeai` | `GEMINI_API_KEY` |

#### `EmbeddingCache`
Redis-backed cache keyed by SHA-256 of the query text. Stores embeddings as JSON-serialised float lists.

Constructor: `EmbeddingCache(redis_url: str, ttl: int = 86400)`

| Method | Returns |
|---|---|
| `get(query: str) → list[float] \| None` | Cached vector or `None` |
| `set(query: str, embedding: list[float])` | Stores with TTL |

Cache key: `SHA-256(query.encode())` as hex string.

---

### 5d. Retrieval

#### `online_pipeline/retrieval/hybrid_retriever.py` — `HybridRetriever`
Fuses BM25 and vector search using **Reciprocal Rank Fusion (RRF)**.

Constructor:
```python
HybridRetriever(vector_db=None, corpus=None, bm25_weight=0.3, vector_weight=0.7, rrf_k=60)
```
- `corpus`: in-memory list of docs for BM25; `None` skips BM25
- `rrf_k`: RRF rank constant (higher = flatter score distribution)

| Method | Args | Returns |
|---|---|---|
| `retrieve(query_text, query_vector, top_k, metadata_filter)` | text + dense vector | `list[dict]` sorted by descending RRF score |

#### `online_pipeline/retrieval/vector_retriever.py` — `VectorRetriever`
Wraps `VectorDBClient.search()`.

| Method | Returns |
|---|---|
| `retrieve(query_vector, top_k, metadata_filter) → list[dict]` | Renames `score` → `vector_score` |

#### `online_pipeline/retrieval/bm25_retriever.py` — `BM25Retriever`
In-memory BM25 over a corpus using `rank-bm25`.

---

### 5e. Reranker — `online_pipeline/reranker/reranker.py`

`Reranker` uses a HuggingFace cross-encoder to re-score candidates.

| Constructor arg | Default |
|---|---|
| `model_name` | `"cross-encoder/ms-marco-MiniLM-L-6-v2"` |

| Method | Args | Returns | Fallback |
|---|---|---|---|
| `rerank(query, candidates, top_k)` | str, list[dict], int | `list[dict]` with `rerank_score` added, sorted desc | Returns `candidates[:top_k]` if model unavailable |

---

### 5f. Context Builder — `online_pipeline/context_builder/builder.py`

`ContextBuilder` assembles reranked chunks into an LLM prompt string.

Constructor: `ContextBuilder(max_tokens=3000, separator="\n\n---\n\n")`

| Method | Returns |
|---|---|
| `build(chunks: list[dict]) → str` | Numbered blocks `[1] Source: ...\n<text>` joined by separator, budget-capped |

Token budget uses whitespace word count (approximate). Stops adding chunks when budget exhausted.

---

### 5g. LLM Client — `online_pipeline/llm/llm_client.py`

`LLMClient` dispatches generation to the selected provider.

Constructor:
```python
LLMClient(provider="groq", model=None, temperature=0.2, max_tokens=1024)
```

If `model` is `None`, defaults from `_DEFAULT_MODELS` dict are used:

| Provider | Default model |
|---|---|
| `groq` | `llama-3.1-8b-instant` |
| `gemini` | `gemini-1.5-flash` |
| `cohere` | `command-r` |
| `mistral` | `mistral-small-latest` |
| `openai` | `gpt-4o-mini` |
| `anthropic` | `claude-3-haiku-20240307` |

| Method | Returns |
|---|---|
| `generate(query, context) → str` | Grounded answer string |

Private methods: `_groq`, `_gemini`, `_cohere`, `_mistral`, `_openai`, `_anthropic` — each takes `user_message: str` and returns `str`.

System prompt instructs the model to answer only from provided context and cite source numbers.

---

## 6. Vector DB Client — `vector_db/client.py`

`VectorDBClient` wraps the Qdrant Python SDK.

Constructor:
```python
VectorDBClient(url=None, collection=None, api_key=None, dimension=None)
```
All args fall back to `settings.*` values.

| Method | Notes |
|---|---|
| `ensure_collection()` | Creates Qdrant collection with cosine distance if absent |
| `upsert(points: list[dict])` | Each point: `{id, vector, payload, text}`. `text` is stored inside payload. IDs default to `uuid4()` |
| `search(vector, top_k, metadata_filter) → list[dict]` | Returns `[{"text", "metadata", "score"}]`. `metadata_filter` is a raw Qdrant filter dict |

---

## 7. Code Conventions

- All modules use `from __future__ import annotations`
- Settings always imported as `from config.settings import settings` (singleton)
- Heavy imports (SDKs) are deferred inside methods to keep startup fast
- Exceptions in non-critical paths are caught, logged with `logger.warning/error`, and degraded gracefully (e.g. rewriter falls back to original query, reranker falls back to retrieval order)
- Point IDs in Qdrant are strings: `"{doc_id}_{chunk_index}"` for ingested docs
- `embed_text()` is the single shared embedding function used by both `/ingest` and `/query`

---

## 8. Running the System

```bash
# 1. Start infrastructure
docker compose up -d

# 2. Install deps
pip install -r requirements.txt

# 3. Start API (default: http://localhost:8000)
uvicorn online_pipeline.api.gateway:app --host 0.0.0.0 --port 8000 --reload

# 4. Ingest a document
curl -X POST http://localhost:8000/ingest \
  -F "file=@/path/to/document.pdf"

# 5. Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is this document about?"}'

# 6. Swagger UI
open http://localhost:8000/docs

# 7. Qdrant dashboard
open http://localhost:6333/dashboard
```

---

## 9. Environment Variables (`.env`)

```
GROQ_API_KEY=          # primary LLM provider (free)
GEMINI_API_KEY=        # alternative free LLM / embeddings
COHERE_API_KEY=        # alternative free LLM / embeddings
MISTRAL_API_KEY=       # alternative free LLM
EMBEDDING_PROVIDER=sentence-transformers   # local, no key needed
EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384
```
