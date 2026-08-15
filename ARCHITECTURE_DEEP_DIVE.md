# Production RAG — Architecture Deep Dive

A complete, chronological walkthrough of every code file, what it does, how data flows through it, and **why** each architectural decision exists.

---

## Table of Contents

1. [The Big Picture](#the-big-picture)
2. [Shared Foundation](#shared-foundation)
3. [Offline Pipeline — Ingestion](#offline-pipeline--ingestion)
   - [Step 1 — Extraction](#step-1--extraction)
   - [Step 2 — Text Cleaning](#step-2--text-cleaning)
   - [Step 3 — Metadata Enrichment](#step-3--metadata-enrichment)
   - [Step 4 — Chunking Strategy Router](#step-4--chunking-strategy-router)
   - [Step 5 — Kafka (Optional Async Path)](#step-5--kafka-optional-async-path)
   - [Step 6 — Embedding Worker](#step-6--embedding-worker)
   - [Step 7 — Vector DB Upsert](#step-7--vector-db-upsert)
4. [Online Pipeline — Query & Answer](#online-pipeline--query--answer)
   - [Step 1 — HTTP Request hits the Gateway](#step-1--http-request-hits-the-gateway)
   - [Step 2 — LLM Router selects a provider](#step-2--llm-router-selects-a-provider)
   - [Step 3 — Query Rewriting](#step-3--query-rewriting)
   - [Step 4 — Embedding Cache + Embed](#step-4--embedding-cache--embed)
   - [Step 5 — Hybrid Retrieval](#step-5--hybrid-retrieval)
   - [Step 6 — Reranking](#step-6--reranking)
   - [Step 7 — Context Building](#step-7--context-building)
   - [Step 8 — LLM Answer Generation](#step-8--llm-answer-generation)
5. [Infrastructure Services](#infrastructure-services)
6. [Why This Architecture?](#why-this-architecture)

---

## The Big Picture

This system has **two completely separate pipelines** that share infrastructure:

```
┌───────────────────────────────────────────────────────────────┐
│  OFFLINE PIPELINE  (writes to the knowledge base)             │
│                                                               │
│  Document/URL/API → Extract → Clean → Enrich → Chunk →        │
│  [Kafka queue] → Embed → VectorDB.upsert()                    │
└───────────────────────────────────────────────────────────────┘

                        ┌─────────────┐
                        │  Qdrant DB  │  ← shared store
                        └─────────────┘

┌───────────────────────────────────────────────────────────────┐
│  ONLINE PIPELINE  (reads from the knowledge base to answer)   │
│                                                               │
│  HTTP /query → Rewrite → [Redis cache] → Embed →             │
│  HybridRetrieve → Rerank → BuildContext → LLM → Answer       │
└───────────────────────────────────────────────────────────────┘
```

The **Offline Pipeline** is a *write path*: it transforms raw documents into searchable vectors stored in Qdrant.

The **Online Pipeline** is a *read path*: it takes a user question, finds the most relevant chunks, and generates a grounded answer.

---

## Shared Foundation

### `config/settings.py` — The Single Source of Truth

**File:** `config/settings.py`  
**Class:** `Settings` (inherits `pydantic_settings.BaseSettings`)

This is the **first file loaded by every other module**. It reads all configuration from a `.env` file and exposes them as typed Python attributes. Nothing is hardcoded anywhere else in the codebase — every module imports `from config.settings import settings`.

```python
settings = Settings()  # singleton at module level
```

**Why pydantic-settings?**
- Validates types at startup (you get a crash early, not at runtime)
- `.env` file support out of the box
- `extra="ignore"` means unused env vars don't break anything

**Key groups of settings:**

| Group | What it controls |
|---|---|
| LLM API keys | Which paid/free LLM providers can be used |
| `embedding_provider` + `embedding_model` | Which model generates the vectors |
| `embedding_dimension` | Must match the Qdrant collection's vector size |
| Kafka settings | Message broker for async ingestion |
| Redis settings | Embedding cache TTL and URL |
| Qdrant settings | Where vectors are stored |
| `top_k_retrieval` / `top_k_rerank` | Pipeline funnel sizes |
| `chunking_strategy` / `chunk_size` | How documents are split |

---

## Offline Pipeline — Ingestion

The offline pipeline is triggered via `POST /ingest` on the API gateway. In a scaled deployment, it can also be run fully asynchronously through Kafka. Here is the complete step-by-step flow:

```
UploadedFile (PDF/DOCX/TXT)
        │
        ▼
  DocumentExtractor.extract()          ← offline_pipeline/extractors/document_extractor.py
        │ {"text": str, "metadata": {source, file_type, page_count}}
        ▼
  TextCleaner.clean()                  ← offline_pipeline/preprocessors/cleaner.py
        │ normalised text
        ▼
  MetadataEnricher.enrich()           ← offline_pipeline/preprocessors/metadata_enricher.py
        │ metadata += {doc_id, char_count, word_count, ingested_at}
        ▼
  ChunkingStrategyRouter.chunk()       ← offline_pipeline/chunkers/router.py
        │   ┌── TextChunker (fixed_token)        ← chunkers/text_chunker.py
        │   └── RecursiveTextChunker (recursive) ← chunkers/recursive_chunker.py
        │ list of {text, metadata{chunk_index, token_count, chunking_strategy}}
        ▼
  [Optional: ChunkProducer → Kafka → ChunkConsumer → EmbeddingWorker]
        │                            ← kafka/producer.py, consumer.py, embedding_workers/worker.py
        ▼
  embed_text(chunk.text)              ← online_pipeline/cache/embedding_cache.py
        │ list[float] (384-dim default)
        ▼
  VectorDBClient.upsert()             ← vector_db/client.py
        │ point = {id: uuid5, vector, payload: metadata, text}
        ▼
      Qdrant
```

---

### Step 1 — Extraction

**File:** `offline_pipeline/extractors/document_extractor.py`  
**Class:** `DocumentExtractor`

This is the entry point for all document ingestion. It reads a file from disk and turns it into a Python dict with `text` and `metadata`.

```
file_path (.pdf / .docx / .txt)
        │
        ├── .pdf  → pypdf.PdfReader → extract_text() per page
        │             └── blank page? → Tesseract OCR fallback (_ocr_page)
        ├── .docx → python-docx Document → join all paragraph.text
        └── .txt  → Path.read_text(encoding="utf-8")
        │
        ▼
{"text": "full document text...", "metadata": {"source": path, "file_type": "pdf", "page_count": N}}
```

**Why Tesseract OCR fallback?**  
Many PDFs (scanned documents, report exports) have image-based pages where `pypdf` extracts nothing. The fallback renders the page as an image and runs OCR to recover the text. Without this, scanned documents would silently produce empty chunks.

**Other extractors (not used in the `/ingest` route, available for CLI/batch use):**
- `web_extractor.py` — `WebExtractor`: uses `requests` + `BeautifulSoup` to scrape a URL. Strips nav/footer/script/style tags to get clean body text.
- `api_extractor.py` — `APIExtractor`: uses `httpx` to hit a JSON REST endpoint. Can optionally pick only specific fields from the JSON response.

---

### Step 2 — Text Cleaning

**File:** `offline_pipeline/preprocessors/cleaner.py`  
**Class:** `TextCleaner`

Raw extracted text is messy. This step normalises it in three passes before any other processing touches it.

```
raw text
    │
    ├── NFKC Unicode normalisation   → "ﬁle" becomes "file", "½" stays "½"
    ├── Control char removal         → strips \x00–\x1f EXCEPT \n and \t
    └── Whitespace collapse          → 3+ blank lines → 2 blank lines
                                     → multiple spaces on one line → single space
    │
    ▼
clean, normalised text
```

**Why NFKC specifically?**  
Many PDF extractors produce ligature characters (`ﬁ`, `ﬂ`), compatibility forms of digits or punctuation, and full-width characters. NFKC decomposes these into their canonical ASCII equivalents. Without it, token counts are inflated and BM25 keyword matches fail because `ﬁle` ≠ `file`.

**Why preserve `\n` and `\t`?**  
These carry structural meaning (paragraph breaks, table alignment). Stripping them would destroy the signals used by the `RecursiveTextChunker` to find natural split boundaries.

---

### Step 3 — Metadata Enrichment

**File:** `offline_pipeline/preprocessors/metadata_enricher.py`
**Class:** `MetadataEnricher`

After cleaning, the doc dict goes through `enrich()` which derives four extra metadata fields and attaches them.

```python
metadata["doc_id"]      = hashlib.sha256(text.encode()).hexdigest()
metadata["char_count"]  = len(text)
metadata["word_count"]  = len(text.split())
metadata["ingested_at"] = datetime.now(tz=timezone.utc).isoformat()
```

**Why SHA-256 as the doc ID?**  
The hash is deterministic: the same document always gets the same ID. This enables **idempotent re-ingestion** — if you re-ingest a document, the Qdrant point IDs (derived from `doc_id + chunk_index`) will be the same, and `upsert` will overwrite rather than duplicate. It also uniquely identifies a document without needing an external database sequence.

---

### Step 4 — Chunking Strategy Router

**File:** `offline_pipeline/chunkers/router.py`  
**Class:** `ChunkingStrategyRouter`

LLMs have a fixed context window, so long documents must be split into chunks. The router decides *how* to split based on the document's structure.

```
ChunkingStrategyRouter.chunk(text, metadata)
        │
        ├── strategy="fixed_token"  → TextChunker
        ├── strategy="recursive"    → RecursiveTextChunker
        └── strategy="auto" (default)
              │
              ├── file_type in {md, markdown, html, web}? → recursive
              ├── text contains "\n\n" or "#" headers?   → recursive
              └── otherwise                              → fixed_token
```

#### `TextChunker` — `offline_pipeline/chunkers/text_chunker.py`

The baseline chunker. Tokenises the entire text with `tiktoken` (`cl100k_base` encoding, same as OpenAI models) and slices it into windows:

```
tokens = tiktoken.encode(text)

window:  [0 ──────── 511]           chunk 0 (512 tokens)
              [448 ──────── 959]     chunk 1 (overlap = 64 tokens)
                    [896 ──────── 1407]  chunk 2
...
```

**Why overlap?**  
If a sentence spans a chunk boundary, both chunks contain it. This means a query about that sentence will match at least one chunk reliably. Without overlap, boundary sentences would be permanently split and often missed.

**Why tiktoken?**  
It counts the same tokens the LLM itself counts, giving you accurate context-window math. Falls back to whitespace splitting when tiktoken is not installed.

#### `RecursiveTextChunker` — `offline_pipeline/chunkers/recursive_chunker.py`

A smarter chunker for structured text. Instead of blindly slicing at token boundaries, it tries to split at natural text boundaries in this order:

```
1. \n\n  (paragraph breaks)
2. \n    (line breaks)
3. ". "  (sentence endings)
4. " "   (word boundaries)
5. token fallback (delegates to TextChunker)
```

Each piece is recursively split further if it still exceeds `chunk_size`. Pieces that fit within the budget are merged back together with overlap between chunks.

**Why two chunkers?**  
Flat unstructured text (e.g., a raw transcript) has no meaningful boundaries to split on — fixed_token gives you predictable uniform chunks. Structured prose (a Wikipedia article, markdown docs, a web page) has explicit paragraph and section structure — splitting at those boundaries keeps semantically coherent units together, improving retrieval precision.

---

### Step 5 — Kafka (Optional Async Path)

**Files:** `offline_pipeline/kafka/producer.py`, `offline_pipeline/kafka/consumer.py`  
**Classes:** `ChunkProducer`, `ChunkConsumer`

In the synchronous `/ingest` API route, chunks go directly to embedding. For high-throughput batch ingestion, chunks are published to Kafka and consumed asynchronously by an `EmbeddingWorker`.

```
ChunkProducer.send(chunk)
    → confluent_kafka.Producer.produce(topic="rag.chunks", value=json.dumps(chunk))
    → producer.flush()

                            Kafka topic: rag.chunks
                                     │
                        ChunkConsumer.consume()
                            → yields deserialized chunk dicts
```

**Why Kafka?**  
- **Backpressure**: If embedding (which calls an API or loads a GPU model) is slower than document ingestion, the queue absorbs the difference.
- **Durability**: Messages persist on disk. A crashed embedding worker can restart and continue from its last committed offset.
- **Decoupling**: Multiple embedding workers can consume the same topic in parallel (same `group.id` = load balanced), or different workers for different tasks can consume the same messages independently.

---

### Step 6 — Embedding Worker

**File:** `offline_pipeline/embedding_workers/worker.py`  
**Class:** `EmbeddingWorker`

Used in the Kafka async path. Batches chunks from the consumer and embeds them in groups for efficiency.

```
for chunk in consumer.consume():          # blocking generator
    batch.append(chunk)
    if len(batch) >= batch_size (32):
        texts = [c["text"] for c in batch]
        vectors = embedder.embed(texts)    # batch API call
        points = [{id, vector, payload, text} for each]
        vector_db.upsert(points)
        batch = []
```

**Why batch size 32?**  
Most embedding APIs (OpenAI, Cohere) and `SentenceTransformer.encode()` are significantly faster per item when called in batches rather than one at a time. 32 is a safe default that works well on CPU and GPU alike.

---

### Step 7 — Vector DB Upsert

**File:** `vector_db/client.py`  
**Class:** `VectorDBClient`

The final step of ingestion — writing the vector and its associated payload into Qdrant.

```python
# In /ingest route (sync path)
point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}_{chunk_index}"))

_vector_db.upsert([{
    "id": point_id,
    "vector": embed_text(chunk["text"]),   # 384-dim float list
    "payload": chunk["metadata"],          # all metadata fields
    "text": chunk["text"],                 # stored inside payload
}])
```

`ensure_collection()` is called at startup and creates the Qdrant collection with **cosine distance** if it does not already exist.

**Why cosine distance?**  
Sentence-transformer vectors and OpenAI embeddings are L2-normalised, meaning their dot product equals their cosine similarity. Cosine distance ignores vector magnitude and measures the angle between vectors, making it robust to varying text lengths.

**Why UUID5 from `doc_id + chunk_index`?**  
UUID5 is deterministic (same inputs → same UUID). Re-ingesting the same document produces the same point IDs, so Qdrant performs an update rather than creating duplicates.

---

## Online Pipeline — Query & Answer

This pipeline is triggered by `POST /query`. Every component is initialised once at startup inside the `lifespan` context manager and reused across all requests.

```
POST /query {"query": "...", "top_k": 5}
        │
        ▼
  gateway.py: query()
        │
        ├── 1. LLMRouter.rewrite_query(query)       ← LLM query rewrite
        ├── 2. EmbeddingCache.get(rewritten)         ← Redis cache check
        │       └── miss → embed_text(rewritten)     ← compute + cache
        ├── 3. HybridRetriever.retrieve(...)         ← BM25 + Vector → RRF fusion
        ├── 4. Reranker.rerank(...)                  ← cross-encoder scoring
        ├── 5. ContextBuilder.build(reranked)        ← format chunks for LLM
        └── 6. LLMRouter.generate_answer(...)        ← final LLM call
        │
        ▼
{"answer": "...", "sources": [...metadata...]}
```

---

### Step 1 — HTTP Request hits the Gateway

**File:** `online_pipeline/api/gateway.py`  
**Framework:** FastAPI

The gateway is the only public-facing entry point. At startup, the `lifespan` context manager initialises all singletons exactly once:

```python
@asynccontextmanager
async def lifespan(app):
    _vector_db      = VectorDBClient()         # connects to Qdrant
    _vector_db.ensure_collection()             # creates collection if absent
    _llm_router     = LLMRouter()              # reads routing config
    _cache          = EmbeddingCache(...)      # connects to Redis
    _retriever      = HybridRetriever(...)     # sets up BM25 + vector
    _reranker       = Reranker()               # loads cross-encoder model
    _context_builder = ContextBuilder()
    yield
    # cleanup would go here
```

**Why lifespan and not module-level globals?**  
`lifespan` runs after the event loop starts. This ensures async-unfriendly startup work (model loading, network connections) happens in the right context and only once, not on every import.

**Request model validation:**
```python
class QueryRequest(BaseModel):
    query: str
    metadata_filter: dict | None = None   # passed to Qdrant for filtered search
    top_k: int = settings.top_k_rerank    # default 5
    chat_history: list[dict] = []         # multi-turn conversation support
```

---

### Step 2 — LLM Router selects a provider

**File:** `online_pipeline/llm/router.py`  
**Class:** `LLMRouter`

Before any LLM call is made, the router decides which provider and model to use, and which fallbacks to try if the primary fails.

```
LLMRouter(routing_mode="balanced")
    │
    ├── _route_for("query_rewrite")
    │       → ModelRoute(provider="groq", model="llama-3.1-8b-instant", temperature=0.0)
    │
    └── _route_for("rag_answer")
            → routing_mode="cost_optimized" → Groq (free)
            → routing_mode="balanced"       → settings.answer_generation_provider
            → routing_mode="quality"        → Anthropic (most capable)

Fallback chain: primary → gemini → openai → anthropic
    → each is tried in sequence; first success wins
    → all fail → RuntimeError
```

**Why a router instead of directly calling an LLM?**  
Provider APIs go down, rate limits are hit, and API keys expire. The router abstracts these away: callers simply request `generate_answer(...)` and get a response without knowing which provider delivered it. The fallback chain means the system stays available even when the primary provider has an outage.

**Three routing modes:**

| Mode | Use Case | Provider Choice |
|---|---|---|
| `cost_optimized` | High-volume production | Groq (free tier) |
| `balanced` | Default | Whatever is configured |
| `quality` | Research / accuracy-critical | Anthropic Claude |

---

### Step 3 — Query Rewriting

**File:** `online_pipeline/query_rewrite/rewriter.py`  
**Class:** `QueryRewriter`

User queries are often conversational and ambiguous. This step transforms them into precise retrieval-optimised search queries using an LLM.

```
User: "what did they say about climate?"
            │
            ▼
  QueryRewriter.rewrite()
    → LLMClient(provider="groq", temperature=0.0)
    → system: "Rewrite as a single concise self-contained search query."
    → user:   "what did they say about climate?"
            │
            ▼
  "climate change findings and statements"
```

**Why `temperature=0.0`?**  
Rewriting is not a creative task. You want the most deterministic, focused output — no randomness. Higher temperature would produce varied rewrites that could harm retrieval consistency.

**Why rewrite at all?**  
- "what did they say" — who is "they"? The retriever doesn't know.
- "climate" — too vague. "climate change", "climate policy", "climate data" are all different.
- Rewriting removes pronouns, adds domain context, and removes filler words that confuse embedding models.

The rewriter **falls back to the original query** on any exception, so a failed rewrite never blocks a user's question from being answered.

---

### Step 4 — Embedding Cache + Embed

**File:** `online_pipeline/cache/embedding_cache.py`  
**Functions/Classes:** `embed_text()`, `EmbeddingCache`

#### `embed_text(text)` — the shared embedding function

Used by **both** the ingest and query paths. Routes to the configured provider:

```
settings.embedding_provider
    │
    ├── "sentence-transformers"  → SentenceTransformer(model).encode(text)  [local]
    ├── "openai"                 → client.embeddings.create(model, input=text)
    ├── "cohere"                 → client.embed(texts=[text], input_type="search_query")
    └── "gemini"                 → genai.embed_content(model, content=text, task_type="retrieval_query")
    │
    ▼
list[float]  (e.g. 384 dims for all-MiniLM-L6-v2)
```

**Critical**: The same `embedding_provider` and `embedding_model` must be used for both ingestion and queries. If you index with Sentence Transformers and query with OpenAI, the vectors live in different geometric spaces and similarity search is meaningless.

#### `EmbeddingCache` — Redis-backed cache

```
query = "climate change findings"

key = "emb:{provider}:{model}:{sha256(query)}"
    e.g. "emb:sentence_transformers:all_MiniLM_L6_v2:a3f2..."

Redis.get(key)
    ├── hit  → return cached list[float]   (no API call, instant)
    └── miss → embed_text(query) → Redis.setex(key, TTL=86400, value)
```

**Why namespace the key by `provider:model`?**  
If you change the embedding model (e.g., from `all-MiniLM-L6-v2` to `text-embedding-3-small`), old cached vectors would be returned for new queries. They'd be in the wrong vector space — lookups would return garbage. Namespacing by the embedding identity makes this impossible: old entries are simply unreachable under the new key prefix.

**Why Redis for caching?**  
Embedding API calls cost money (and time). The same question ("what is RAG?") asked by 100 users should only call the embedding API once. Redis keeps cached vectors in memory with a configurable TTL and survives API gateway restarts.

---

### Step 5 — Hybrid Retrieval

**File:** `online_pipeline/retrieval/hybrid_retriever.py`  
**Class:** `HybridRetriever`

This is the retrieval core. It runs **two independent searches** and fuses their results.

```
query_text  = "climate change findings"    ← for BM25 (keyword)
query_vector = [0.12, -0.04, ...]          ← for Qdrant (semantic)

BM25Retriever.retrieve(query_text, top_k=20)
    → rank-bm25 scores over in-memory corpus
    → ranked list of docs by TF-IDF-like score

VectorRetriever.retrieve(query_vector, top_k=20, metadata_filter)
    → Qdrant ANN search (cosine similarity)
    → ranked list of docs by cosine score

HybridRetriever._rrf_fuse(bm25_results, vector_results, top_k=20)
    │
    ▼
RRF Score Formula:
  score(doc) = bm25_weight  / (rrf_k + rank_in_bm25)
             + vector_weight / (rrf_k + rank_in_vector)

  where rrf_k = 60, bm25_weight = 0.3, vector_weight = 0.7
    │
    ▼
top-20 de-duplicated results sorted by descending hybrid_score
```

**Why both BM25 and vector search?**

| Search Type | Strength | Weakness |
|---|---|---|
| BM25 (keyword) | Exact term matching, rare proper nouns, codes | No semantic understanding |
| Vector (semantic) | Understands meaning and synonyms | Can miss exact keyword matches |

Together they cover each other's blind spots. A query for "LSTM" (exact term) will score high in BM25. A query for "how do recurrent networks remember sequences?" will score high in semantic search. Fusion captures both.

**Why Reciprocal Rank Fusion (RRF) instead of a weighted sum of raw scores?**  
BM25 scores and cosine similarities live on completely different scales. You cannot add them directly. RRF converts both to ranks first, then uses the formula `1/(k + rank)` which is scale-independent and robust to outliers in raw scores.

**Why `bm25_weight=0.3` and `vector_weight=0.7`?**  
Modern embedding models are very strong at semantic matching, so they get more weight. BM25 is a useful secondary signal but shouldn't dominate.

---

### Step 6 — Reranking

**File:** `online_pipeline/reranker/reranker.py`  
**Class:** `Reranker`

After retrieval, the top-20 candidates are re-scored with a much more expensive but accurate model.

```
query = "climate change findings"
candidates = [doc1, doc2, ..., doc20]

CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2").predict([
    (query, doc1.text),
    (query, doc2.text),
    ...
    (query, doc20.text),
])
    → [0.92, 0.34, 0.87, ...]   float relevance scores

sorted by score descending → top 5
    → each doc gets "rerank_score" field added
    → return top_k (default 5) docs
```

**Why a two-stage retrieve-then-rerank pattern?**

Retrieval (BM25 + vector) operates at scale. Running a cross-encoder on all documents would be prohibitively slow. The retriever acts as a **fast filter** that narrows the candidate pool from thousands to 20. The reranker then applies a more accurate (but slow) cross-attention model to those 20 — a number small enough to be fast.

**Why a cross-encoder instead of just trusting the vector scores?**  
Bi-encoders (used for embedding) encode query and document **independently** and compare them. Cross-encoders process the query and document **together** in a single forward pass, allowing full attention between all tokens. This gives much more accurate relevance scores but cannot scale to the full document corpus.

**Graceful fallback:**  
If the cross-encoder model fails to load (no `sentence_transformers` installed, GPU OOM), the reranker returns `candidates[:top_k]` in original retrieval order. The system degrades gracefully rather than crashing.

---

### Step 7 — Context Building

**File:** `online_pipeline/context_builder/builder.py`  
**Class:** `ContextBuilder`

The reranked chunks are formatted into a human-readable (and LLM-parseable) context string.

```python
# For each chunk (up to token budget of 3000 words):
block = f"[{i}] Source: {chunk.metadata.source}\n{chunk.text}"

# Joined with:
separator = "\n\n---\n\n"
```

Output example:
```
[1] Source: report_2024.pdf
Climate change data from 2024 shows a 1.5°C increase...

---

[2] Source: ipcc_summary.pdf
The IPCC report concludes that human activities are the...
```

**Why number the sources `[1]`, `[2]`?**  
The LLM system prompt instructs it to cite source numbers in the answer. This connects the answer claims back to specific chunks whose metadata is returned as `sources` in the API response. Users can verify which document each claim came from.

**Why a token budget?**  
LLMs have context window limits. Adding unlimited chunks would exceed the limit and either error out or force truncation of the question itself. The budget ensures the most relevant chunks fit within a safe window size.

---

### Step 8 — LLM Answer Generation

**File:** `online_pipeline/llm/llm_client.py`  
**Class:** `LLMClient`

The final step: give the LLM the question, the conversation history, and the context, and get a grounded answer.

```python
user_message = f"""
Conversation so far:
{history_text}

Context:
[1] Source: report_2024.pdf
Climate change data...

---

[2] Source: ipcc_summary.pdf
The IPCC report...

Latest user question: What do reports say about temperature increases?
"""

system_prompt = (
    "You are a helpful assistant. "
    "Answer ONLY from the provided context. "
    "If the context does not contain enough information, say so. "
    "Cite source numbers [1], [2] where relevant."
)

LLMClient(provider="groq", model="llama-3.1-8b-instant").complete(system_prompt, user_message)
```

**Why `answer ONLY from context`?**  
This prevents the LLM from hallucinating facts from its training data that aren't in the indexed documents. The RAG pattern's entire value is grounding answers in *your* documents, not general internet knowledge.

**Why pass `chat_history`?**  
Multi-turn conversations need context. Without history, a follow-up question "what about the 2023 data?" has no referent. The history is formatted as `User: ...\nAssistant: ...` blocks prepended to the context.

**Provider dispatch table:**
```python
dispatch = {
    "groq":      self._groq,
    "gemini":    self._gemini,
    "cohere":    self._cohere,
    "mistral":   self._mistral,
    "anthropic": self._anthropic,
    # default:   self._openai
}
```

Each provider method wraps the respective SDK's chat completion API. All providers receive the same `system_prompt` and `user_message`; they differ only in the SDK call structure.

---

## Infrastructure Services

These four services (`docker-compose.yml`) must be running before starting the API:

| Service | Port | Purpose in This System |
|---|---|---|
| **Zookeeper** | 2181 | Kafka's metadata coordinator. Manages broker state and topic/partition metadata. Required by Kafka. |
| **Kafka** | 9092 | The async message queue. Holds chunks between the ingestion pipeline and the embedding workers. Survives worker crashes. |
| **Redis** | 6379 | The embedding vector cache. Prevents redundant embedding API calls. Stores vectors as JSON with a 24-hour TTL. |
| **Qdrant** | 6333 | The vector database. Stores all embedded chunks and provides ANN (Approximate Nearest Neighbor) search. Qdrant dashboard: `http://localhost:6333/dashboard`. |

---

## Why This Architecture?

### Separation of Online and Offline Paths

Ingestion is slow (PDF parsing, OCR, chunking, embedding API calls). Queries must be fast (sub-second response). Keeping them on separate paths means a large batch ingestion job never slows down query response times.

### Kafka as an Ingestion Buffer

Without Kafka, a spike in document uploads would queue up behind the embedding API rate limit and block the HTTP server. Kafka decouples the upload acknowledgement from the actual embedding work. The HTTP server accepts documents instantly; the embedding worker processes them at a sustainable rate.

### Two-Stage Retrieval (Retrieve → Rerank)

Running a cross-encoder on every document in Qdrant would take minutes. Running vector search to get the top 20, then reranking those 20 with a cross-encoder, takes milliseconds for retrieval and ~100ms for reranking — acceptable for a live query.

### Hybrid BM25 + Vector Search

Neither approach dominates in all cases. Combining them with RRF fusion consistently outperforms either alone on retrieval benchmarks. The 0.3/0.7 split reflects the empirical observation that semantic search is more generally useful but exact-match capability of BM25 fills important gaps.

### Redis Embedding Cache

Embedding the same query string twice is wasteful. Common questions ("what is this project about?") would hit the embedding API on every single request. Redis makes the second and all subsequent requests for the same query essentially free.

### Embedding Cache Namespaced by Model Identity

A subtle but critical safety feature. If you swap from `all-MiniLM-L6-v2` (384-dim) to `text-embedding-3-small` (1536-dim), cached vectors from the old model would be served for new queries. They represent completely different geometric spaces. Namespacing by `provider:model` means cache misses are automatic when the model changes, so stale vectors can never corrupt search results.

### `embed_text()` as a Single Shared Function

Both the ingestion path and the query path call the same `embed_text()` function. This is not an accident — it is a guarantee that the same embedding model is always used for both, which is a correctness requirement for vector similarity search to work.

### Pydantic Settings at the Root

Every tunable parameter lives in `config/settings.py`. No hardcoded values exist in business logic. This means you can switch from Groq to OpenAI, change the chunk size, or swap the embedding model entirely via a single `.env` file change with zero code modifications.
