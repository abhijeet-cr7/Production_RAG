# Production RAG System

A production-grade Retrieval-Augmented Generation (RAG) system with two coordinated pipelines.

---

## Architecture

### Offline Pipeline

```
Documents / Websites / APIs
           │
           ▼
    Extraction / OCR
           │
           ▼
  Cleaning + Metadata
           │
           ▼
      Chunking
           │
           ▼
        Kafka
           │
           ▼
  Embedding Workers
           │
           ▼
      Vector DB
```

### Online Pipeline

```
User Query
      │
      ▼
API Gateway
      │
      ▼
Query Rewrite
      │
      ▼
Embedding Cache  ◄──── Redis Cache Layer
      │
      ▼
Query Embedding
      │
      ▼
Hybrid Retrieval
(BM25 + Vector Search)
      │
      ▼
Metadata Filter
      │
      ▼
Reranker
      │
      ▼
Context Builder
      │
      ▼
LLM
      │
      ▼
Response
```

---

## Project Structure

```
Production_RAG/
├── config/                        # Centralised configuration (env-backed)
├── offline_pipeline/
│   ├── extractors/                # Document, web and API extraction
│   ├── preprocessors/             # Text cleaning and metadata tagging
│   ├── chunkers/                  # Chunking strategies
│   ├── kafka/                     # Kafka producer / consumer
│   └── embedding_workers/         # Batch embedding generation
├── online_pipeline/
│   ├── api/                       # FastAPI gateway
│   ├── query_rewrite/             # LLM-based query rewriting
│   ├── cache/                     # Redis embedding cache
│   ├── retrieval/                 # BM25, vector and hybrid retrieval
│   ├── reranker/                  # Cross-encoder reranking
│   ├── context_builder/           # Context assembly
│   └── llm/                       # LLM client abstraction
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
| LLM | OpenAI GPT / Anthropic Claude |
| API | FastAPI |
