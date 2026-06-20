# Production RAG System

A production-grade Retrieval-Augmented Generation (RAG) system with coordinated ingestion, retrieval, chunking orchestration and LLM routing.

---

## Architecture

### Offline / Ingestion Orchestration

```
File / URL / API
      |
      v
Extractor Router
      |
      +--> DocumentExtractor  (PDF / DOCX / TXT, OCR fallback for image PDFs)
      +--> WebExtractor       (HTML pages)
      +--> APIExtractor       (JSON APIs)
      |
      v
Text Cleaner
      |
      v
Metadata Enrichment
      |
      v
Chunking Strategy Router
      |
      +--> auto
      |     +--> recursive      (structured text, markdown/web-like content)
      |     +--> fixed_token    (flat or unknown text fallback)
      +--> fixed_token          (token window with overlap)
      +--> recursive            (paragraph / line / sentence / word boundaries)
      |
      v
Chunk Quality Metadata
      |
      +--> chunk_index
      +--> token_count
      +--> chunking_strategy
      |
      v
Embedding
      |
      v
Vector DB Upsert
```

### Online / Query Orchestration

```
User Query
      |
      v
API Gateway
      |
      v
LLM Router: query_rewrite
      |
      +--> primary provider/model
      +--> fallback providers
      |
      v
Embedding Cache  <---- Redis Cache Layer
      |
      v
Query Embedding
      |
      v
Hybrid Retrieval
(BM25 + Vector Search)
      |
      v
Metadata Filter
      |
      v
Reranker
      |
      v
Context Builder
      |
      v
LLM Router: rag_answer
      |
      +--> cost_optimized / balanced / quality routing mode
      +--> primary provider/model
      +--> fallback providers
      |
      v
Response
```

### Router Responsibilities

| Router | Module | Purpose |
|---|---|---|
| LLMRouter | `online_pipeline/llm/router.py` | Chooses a provider/model for query rewriting and answer generation, then falls back to backup providers if a call fails. |
| ChunkingStrategyRouter | `offline_pipeline/chunkers/router.py` | Chooses fixed-token or recursive chunking manually or automatically from document structure and metadata. |

### LLM Routing Defaults

| Task | Default Provider | Default Model | Notes |
|---|---|---|---|
| Query rewrite | Groq | `llama-3.1-8b-instant` | Fast, low-temperature rewrite for retrieval. |
| RAG answer | Groq | Provider default | Grounded answer generation from retrieved context. |
| Fallback order | Gemini, OpenAI, Anthropic | Provider default | Used when the primary provider raises an error. |

### Chunking Strategy Defaults

| Strategy | When Used | Behavior |
|---|---|---|
| `auto` | Default | Selects `recursive` for structured text and `fixed_token` for flat/unknown text. |
| `fixed_token` | Baseline / predictable ingestion | Splits into token windows with overlap. Default: 512 tokens, 64 overlap. |
| `recursive` | Structured prose, markdown/web-like text | Splits by paragraphs, lines, sentences and words before falling back to token windows. |

---

## Project Structure

```
Production_RAG/
├── config/                        # Centralised configuration (env-backed)
├── offline_pipeline/
│   ├── extractors/                # Document, web and API extraction
│   ├── preprocessors/             # Text cleaning and metadata tagging
│   ├── chunkers/                  # Chunking strategies and strategy router
│   ├── kafka/                     # Kafka producer / consumer
│   └── embedding_workers/         # Batch embedding generation
├── online_pipeline/
│   ├── api/                       # FastAPI gateway
│   ├── query_rewrite/             # LLM-based query rewriting
│   ├── cache/                     # Redis embedding cache
│   ├── retrieval/                 # BM25, vector and hybrid retrieval
│   ├── reranker/                  # Cross-encoder reranking
│   ├── context_builder/           # Context assembly
│   └── llm/                       # LLM client abstraction and model router
├── vector_db/                     # Vector database client
├── tests/                         # Unit and integration tests
├── docker-compose.yml             # Local dev stack (Kafka, Redis, Qdrant)
├── requirements.txt
└── .env.example
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your API keys and service URLs
```

### 3. Start infrastructure services

```bash
docker compose up -d
```

### 4. Run tests

```bash
pytest tests/ -v
```

---

## Key Technologies

| Component | Technology |
|---|---|
| Message queue | Apache Kafka |
| Cache | Redis |
| Vector store | Qdrant |
| Embeddings | OpenAI / Sentence-Transformers |
| BM25 retrieval | rank-bm25 |
| Reranker | cross-encoder (sentence-transformers) |
| LLM routing | Groq / Gemini / Cohere / Mistral / OpenAI / Anthropic |
| API | FastAPI |

## Routing Configuration

```bash
# LLM router
LLM_ROUTING_MODE=balanced              # cost_optimized | balanced | quality
QUERY_REWRITE_PROVIDER=groq
QUERY_REWRITE_MODEL=llama-3.1-8b-instant
ANSWER_GENERATION_PROVIDER=groq
ANSWER_GENERATION_MODEL=
LLM_FALLBACK_PROVIDERS=gemini,openai,anthropic

# Chunking router
CHUNKING_STRATEGY=auto                 # auto | fixed_token | recursive
CHUNK_SIZE=512
CHUNK_OVERLAP=64
```

### Embedding Cache Safety

Embedding cache keys are namespaced by embedding identity (`provider:model`).
That prevents stale vectors from one embedding space being reused after model
or provider changes. If you change embedding provider/model, cached entries are
automatically isolated under a new namespace.
