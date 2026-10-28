# Production RAG — Architecture Deep Dive

A walkthrough of every part of the system, how data flows through it, and **why** each decision exists. The system answers questions about financial documents (filings, earnings releases, XBRL data) for several tenants that must never see each other's data, with every answer checked before it is shown.

---

## What Changed Recently

Read this first if you are coming back to the project. Everything below is on branch `feat/access-control`, built in phases that each ended with a green test run. Several phases were also *mutation-checked*: a protection was deliberately broken to confirm a test fails (the surviving case is listed under [Known limitations](#known-limitations)).

| Phase | What it adds | Where |
|---|---|---|
| **Grading** | A grader checks each answer is grounded and useful; one bounded retry; web top-up | `online_pipeline/grading/` |
| **P0 Auth** | Okta (OIDC/JWT) login, scopes, per-tenant policy ceiling, fail-closed startup | `online_pipeline/auth/` |
| **P1 Audit** | Tamper-evident, hash-chained log of who did what | `online_pipeline/audit/` |
| **P2 ACLs** | Every chunk carries tenant/owner/groups/classification; ids can't collide across tenants | `online_pipeline/auth/acl.py` |
| **P3 Entitled retrieval** | The caller's access rule is applied inside the vector store *and* re-checked in Python; BM25 is per tenant | `retrieval/`, `vector_db/` |
| **P4 Safe ingestion** | SSRF-proof fetching, hidden-text stripping, injection quarantine, parsing in a sandbox subprocess | `offline_pipeline/safety/` |
| **P5 Provenance** | Server-computed trust tier, content hash, raw-original store, supersession | `online_pipeline/provenance/` |
| **P6 Tables & XBRL** | Tables kept whole; inline XBRL facts parsed exactly; table-aware chunking | `offline_pipeline/tables/` |
| **P7 Answer safety** | Citation + figure verification, output sanitising, advice refusal, per-tenant policies | `online_pipeline/answer_safety/` |
| **P8 Lifecycle** | List/patch/delete documents, legal hold, retention purge, legacy backfill, audit read | `online_pipeline/lifecycle.py` |
| **Docker** | One-command stack with a port check; UI behind nginx | `docker-compose.yml`, `scripts/docker-up.sh` |

The earlier LangGraph "agentic RAG" ideas were applied like this: hallucination grading and answer grading **yes** (one combined call, bounded), web search fallback **already existed** (now also triggered after grading and limited to verified publishers when a tenant requires it), per-document LLM relevance grading **no** (the cross-encoder reranker already scores every chunk), question router **no** (the corpus is open-ended).

A bug review of an earlier branch (`fix/rag-pipeline-defects`) was merged in: the embedding model is loaded once instead of per call, RRF fusion no longer collapses a document's chunks into one, the Kafka worker no longer overwrites a document with itself, BM25 actually runs, and `/query` no longer blocks the event loop.

---

## Table of Contents

1. [The Big Picture](#the-big-picture)
2. [Security model](#security-model)
3. [Shared Foundation](#shared-foundation)
4. [Offline Pipeline — Ingestion](#offline-pipeline--ingestion)
5. [Online Pipeline — Query & Answer](#online-pipeline--query--answer)
6. [Document management and lifecycle](#document-management-and-lifecycle)
7. [Infrastructure Services](#infrastructure-services)
8. [Operations runbook](#operations-runbook)
9. [Known limitations](#known-limitations)
10. [Testing and verification](#testing-and-verification)
11. [Why This Architecture?](#why-this-architecture)

---

## The Big Picture

Two pipelines share one store, and a thin security layer wraps both.

```
        Okta token ──► authenticate ──► scopes ──► tenant policy ceiling ──► audit event
                          │
┌─────────────────────────┴─────────────────────────────────────────────┐
│ OFFLINE (write path)                                                  │
│                                                                       │
│ file / URL / API                                                      │
│   → size cap + type check                                             │
│   → SafeFetcher (SSRF-proof) ......... for URL/API                    │
│   → sandboxed parser subprocess (PDF/DOCX/HTML/XBRL/TXT)              │
│   → hidden-text strip → injection screen (quarantine)                 │
│   → blocks (prose / tables / XBRL facts)                              │
│   → table-aware chunker → embed → upsert with ACL + provenance        │
└───────────────────────────────────────────────────────────────────────┘

                     ┌───────────────────────┐
                     │ Qdrant (per-chunk ACL) │   raw originals: content-addressed store
                     └───────────────────────┘

┌───────────────────────────────────────────────────────────────────────┐
│ ONLINE (read path)                                                    │
│                                                                       │
│ /query → advice refusal → rewrite → embed → ENTITLED retrieval        │
│   (vector pre-filter + per-tenant BM25 + Python guard) → rerank       │
│   → trust filter → [web top-up] → context → LLM (hardened prompt)     │
│   → grade + verify figures → [retry / web] → policy (warn/abstain)    │
│   → sanitise output → answer + sources + verification                 │
└───────────────────────────────────────────────────────────────────────┘
```

The **offline pipeline** turns untrusted documents into searchable, access-controlled, provenance-stamped chunks. The **online pipeline** finds only what the caller may read, drafts an answer, and refuses to show figures it cannot trace to a cited source.

---

## Security model

### Who can do what

A request carries an Okta **access token**. It is validated (RS256 only, issuer, audience, expiry; keys from Okta's JWKS) and turned into a `Principal`: user, **tenant**, groups, clearance (0–3) and scopes. A token with no tenant is refused; there is no default tenant.

| Scope | Allows |
|---|---|
| `rag:query` | `POST /query` |
| `rag:ingest` | uploads, `/ingest/url`, `/ingest/api`, manage *your own* documents |
| `rag:web` | the web-search fallback |
| `rag:admin` | quarantine review, legal hold, manage any document in your tenant |
| `rag:audit` | `GET /audit` (your tenant's events only) |

**Tenant policy is a ceiling.** `config/tenant_policies.yaml` sets, per tenant, whether web search or URL ingest is allowed at all, the URL allow-list, the highest classification, whether only verified sources may feed answers, and the warn/abstain rules. The effective permission is *tenant policy AND user scope AND request flag*; a request can only opt **out**. An unknown tenant gets the most restrictive policy.

### Document access rule

A chunk is readable iff **all** hold: same tenant · `classification ≤ caller's clearance` · (caller is the owner, or the chunk is public to the tenant, or they share a group) · not quarantined · not superseded. A chunk missing any ACL field is denied. This rule is implemented twice and the two must agree:

1. as a Qdrant filter built **server-side** from the token (so top-k fills with allowed chunks), and
2. as `is_authorized()` in Python, run on every candidate list after retrieval and again after any web top-up.

Anything the second layer drops means the first was bypassed, so it is recorded as an `acl.guard_dropped` audit event. The caller can no longer send a filter: `QueryRequest` rejects unknown fields.

### Threats and the control for each

| Threat | Control | Code |
|---|---|---|
| Another tenant reads my documents | ACL pre-filter + guard + per-tenant BM25 + store re-check of BM25 hits | `auth/acl.py`, `retrieval/` |
| Identical upload from another user overwrites my ACL | `doc_id` hashes tenant + owner + content | `auth/acl.py` |
| Forged / expired / key-confusion token | RS256 pinned, `alg=none` rejected, issuer/audience/expiry required | `auth/tokens.py` |
| SSRF via `/ingest/url` | https only, allow-list, every DNS answer must be public, connect to the validated IP, per-hop redirect checks | `safety/net.py` |
| Huge or lying download | decoded-size cap, deadline, declared type must match the bytes | `safety/net.py` |
| Malicious PDF (JavaScript, launch, embedded files, bombs) | object-graph inspection, bounded decompression, parsing in a limited subprocess | `safety/pdf_guard.py`, `sandbox.py` |
| Malicious DOCX / XML (macros, remote templates, XXE, zip bombs) | part/relationship inspection, hardened XML parser, ratio limits | `safety/docx_guard.py`, `xml_guard.py` |
| Hidden-text prompt injection | hidden elements/characters stripped, invisible Unicode removed, injection screen → quarantine | `safety/html_sanitizer.py`, `injection.py` |
| Poisoned document steers the model | context and history wrapped in delimiters it cannot close; prompt says they are data | `llm/llm_client.py` |
| Model leaks data through a link or image | output sanitiser removes images, HTML and URLs that are not the answer's own sources | `answer_safety/sanitize.py` |
| Wrong or invented figure | every figure checked against cited sources: value, period, currency, sign | `answer_safety/verify.py` |
| Unverifiable source feeds an answer | trust tier computed from the host; tenants can require verified sources | `provenance/trust.py` |
| Someone alters the audit trail | HMAC hash chain; `scripts/verify_audit_chain.py` | `audit/logger.py` |
| Secrets in the parser | parser runs with a scrubbed environment and no network credentials | `safety/sandbox.py` |
| Confidential text leaves via tracing | LangSmith runs hide prompts and answers by default | `observability/tracing.py` |

### Audit

Every query, ingest, ACL change, delete, policy denial and auth failure becomes one event in an append-only JSONL file. Each event stores the previous event's hash and `hash = HMAC(key, previous_hash + body)`, so editing, removing or reordering an event breaks the chain and nobody without the key can forge one. Events hold **ids and keyed fingerprints, never document text, queries or answers**. With `AUDIT_FAIL_CLOSED=true` a request whose event cannot be written fails with 503.

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
| `chunking_strategy` / `chunk_size` | How documents are split (`auto`, `fixed_token`, `recursive`, `table_aware`) |
| `app_env`, `auth_mode`, `okta_*` | Environment and Okta token validation (see [Security model](#security-model)) |
| `tenant_policies_path` | The per-tenant policy file |
| `audit_log_path`, `audit_hmac_key`, `audit_fail_closed` | The tamper-evident audit log |
| `ingest_*`, `parser_*`, `pdf_*` | Upload/fetch size caps, rate limit, sandbox CPU/memory/time limits |
| `verified_publishers`, `raw_store_*` | Which hosts count as verified; where originals are kept |
| `require_citations`, `answer_verification_enabled` | Answer verification switches |
| `retention_days` | Used by `scripts/purge_expired.py` |
| `tavily_api_key`, `web_search_*` | Web fallback and its weak-recall thresholds |
| `answer_grading_enabled`, `grader_*`, `max_generation_attempts` | Hallucination / usefulness grading loop (the warn/abstain rules are per tenant, in the policy file) |
| `langsmith_*` | Tracing |

---

## Offline Pipeline — Ingestion

Documents arrive three ways: `POST /ingest` (upload one or more PDF, DOCX, TXT, HTML or XBRL files), `POST /ingest/url`, and `POST /ingest/api`. All three share one rule: **the input is hostile until proven otherwise**, and all three end in the same indexing step.

```
upload ──► size cap (streamed) ─► type must match bytes ─┐
URL/API ─► SafeFetcher ──────────────────────────────────┤
                                                          ▼
                       sandbox subprocess: inspect + parse (PDF/DOCX/HTML/XBRL/TXT)
                                                          │  text + blocks (prose, tables, XBRL facts)
                                                          ▼
                       clean (invisible Unicode) → injection screen ──► quarantine if flagged
                                                          │
                                                          ▼
                       enrich (tenant+owner scoped doc_id) → stamp ACL + provenance
                                                          │
                                                          ▼
                       chunk (table-aware when tables/facts exist) → embed (batched) → upsert
```

Before anything runs, the request's ACL is validated: tenant and owner come from the token only, a caller can share only with groups they belong to, and cannot classify above their own clearance or the tenant's ceiling. A bad ACL rejects the whole request.

---

### Step 1 — Safe extraction

**Files:** `offline_pipeline/safety/` (`net.py`, `sandbox.py`, `extract_worker.py`, `pdf_guard.py`, `docx_guard.py`, `markup_guard.py`, `xml_guard.py`, `upload_guard.py`, `html_sanitizer.py`)

**Fetching (`SafeFetcher`).** URL and API ingests are checked *before* a connection opens and again on every redirect:

```
https only · no credentials in the URL · port 443 · host on the tenant allow-list
   → resolve DNS → EVERY address must be globally routable
        (no loopback, private, link-local/cloud-metadata, CGNAT, multicast, IPv4-mapped, 6to4, NAT64)
   → connect to the validated IP (TLS still verifies the real host name)   ← defeats DNS rebinding
   → follow at most 3 redirects, each re-validated
   → stream the body; stop at the byte cap on the DECODED size (gzip bombs)
   → declared Content-Type must be allow-listed AND match what the bytes are
```

**Parsing (the sandbox).** Every file and every downloaded page is parsed by `extract_worker.py` in a **separate process** that limits itself first (CPU seconds, memory, file size, no core dumps), runs with a **scrubbed environment** (no API keys), and prints one JSON line. The parent kills it on timeout or if its memory grows past the limit. A parser exploit, a hang or a bomb can cost one disposable process, never the API.

**Inspecting before extracting:**

| Format | Rejected when |
|---|---|
| PDF | encrypted · too many pages · JavaScript, launch/submit/import actions, embedded files, rich media, XFA, automatic actions (found by walking the object graph, including objects packed in object streams and nested dictionaries) · a stream that expands past its ceiling or an image with absurd dimensions (measured with a *bounded* decompressor, so a bomb is refused in milliseconds) |
| DOCX | macros, embedded OLE/ActiveX, remote templates or frames · path traversal · too many parts · suspicious compression ratio · any XML part with a DTD or entities (XXE, billion laughs) |
| HTML | sanitised, not rejected: scripts, frames, forms, comments removed, and **hidden elements** (inline-style hiding, `hidden`, `aria-hidden`, off-screen, zero-size, white-on-white, "visually hidden" classes) dropped |
| XML / XBRL | any DTD or entity declaration · wrong root |
| Any | size cap exceeded while streaming · bytes do not match the extension |

PDF text also drops characters a human could not see (1pt, off-page, white on white unless over a dark box) and counts them; a hit is a signal, not a rejection.

Files are accepted only as `.pdf .docx .txt .html .htm .xml .xbrl`. The name shown as the source is the sanitised file name; a server temp path never appears.

---

### Step 2 — Text cleaning and the injection screen

**Files:** `offline_pipeline/preprocessors/cleaner.py`, `offline_pipeline/safety/injection.py`

The cleaner does three things: NFKC Unicode normalisation, removal of **invisible characters** (zero-width, bidi overrides, soft hyphen and the Unicode "tag" block that can smuggle readable ASCII past a reviewer), and whitespace collapse. Table cells are cleaned cell by cell so column structure survives.

Then the **injection screen** looks for text aimed at the model rather than the reader ("ignore all previous instructions", role markers like `<|im_start|>`, tool-call syntax, "send the conversation to https://…", markdown images that carry data out). Findings have a severity:

- **high** → the whole document is stored but **quarantined**: invisible to every retrieval until a tenant admin releases it (`POST /admin/quarantine/{doc_id}/release`);
- medium/low (a `System:` line, a long encoded blob, hidden content that was removed) → recorded in `screen_flags` only.

It is a heuristic: it catches blunt attacks, and the prompt delimiters and output sanitiser exist for the ones it misses. A test corpus of ordinary financial sentences guards against false positives.

---

### Step 3 — Metadata enrichment

**File:** `offline_pipeline/preprocessors/metadata_enricher.py`

```python
metadata["doc_id"]      = sha256(len-prefixed tenant_id + owner_id + text)
metadata["char_count"]  = len(text)
metadata["word_count"]  = len(text.split())
metadata["ingested_at"] = datetime.now(tz=timezone.utc).isoformat()
```

**Why include tenant and owner in the hash?** A hash of the content alone gave two tenants who uploaded the same file the *same* point ids, so the second upload silently overwrote the first one's access rules. Hashing tenant and owner too means identical files stay separate, while the same owner re-uploading the same file still updates in place (idempotent). Length-prefixing stops `("ab","c")` and `("a","bc")` colliding.

Every chunk is then stamped by the gateway (never by a client, extractor or chunker):

| Group | Fields |
|---|---|
| ACL | `tenant_id` `owner_id` `acl_groups` `acl_public` `classification` |
| Provenance | `source_type` `origin` `content_sha256` `trust_tier` `publisher` `ingested_by` `fetched_at` `external_id` `as_of` `integrity_verified` `supersedes` |
| State | `quarantined` `superseded` `screen_flags` |

**Provenance is computed on the server.** `trust_tier` is `verified_primary` only for https content from a publisher on the server-side list (default `sec.gov`), `internal` for an authenticated upload, `unverified` for any other host. A client may declare only `as_of`, `expected_sha256` (a mismatch rejects the document) and `supersedes`. For SEC URLs the accession number is parsed into `external_id`. The exact original bytes are kept in a content-addressed, per-tenant store under their hash (`provenance/rawstore.py`), so a figure can be traced to the file it came from.

---

### Step 4 — Chunking Strategy Router

**File:** `offline_pipeline/chunkers/router.py`  
**Class:** `ChunkingStrategyRouter`

LLMs have a fixed context window, so long documents must be split into chunks. The router decides *how* to split based on the document's structure.

```
ChunkingStrategyRouter.chunk(text, metadata, blocks=None)
        │
        ├── strategy="fixed_token"  → TextChunker
        ├── strategy="recursive"    → RecursiveTextChunker
        ├── strategy="table_aware"  → BlockChunker (needs blocks; else falls back to recursive)
        └── strategy="auto" (default)
              │
              ├── blocks contain a table or XBRL facts?   → table_aware
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

#### `TableChunker` and `BlockChunker` — `offline_pipeline/tables/`, `offline_pipeline/chunkers/block_chunker.py`

When a document actually contains tables or XBRL facts, `auto` selects **`table_aware`** chunking; any other document chunks exactly as before (a test pins this byte-for-byte). Extractors no longer return one flat string: they return ordered **blocks** (headings, prose, tables, fact groups), because flattening destroys tables.

```
HTML:  <table> → grid (colspan/rowspan expanded, header rows, caption and "(In millions)" from the line above)
PDF:   ruled tables via pdfplumber · borderless numeric statements via our own detector
       (label then numbers only, right edges lining up into columns; digit-heavy prose fails both tests)
DOCX:  w:tbl in document order; merged cells appear once; headings from styles
XBRL:  ix:nonFraction → exact value × 10^scale, sign, unit, period, dimensions; hidden free text ignored
```

`TableChunker` rules: **never split a row**; every chunk repeats the caption, the units line and the column headers; an over-wide row is split by column group with its row label kept in each part; footnotes stay attached. Each chunk has two forms: the *embedded text* is linearised (`Net sales: FY2024 = 391,035; FY2023 = 383,285`), which matches queries far better than pipes, and the payload keeps the original Markdown plus `numerics`, the parsed cells with exact values (`(1,234)` → −1234, scale applied once, percentages never scaled). Fact chunks carry the same `numerics` from XBRL. Those structured numbers are what answer verification checks against.

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

```python
point_id = uuid5(NAMESPACE_URL, f"{doc_id}_{chunk_index}")      # doc_id is tenant+owner scoped
upsert([{"id": point_id, "vector": ..., "payload": acl + provenance + state + chunk metadata, "text": ...}])
```

`ensure_collection()` runs at startup: it creates the collection (cosine distance) and **payload indexes** on every field a filter touches (`tenant_id`, `owner_id`, `acl_groups`, `classification`, `acl_public`, `quarantined`, `superseded`, `doc_id`, `source`, `content_sha256`). It is idempotent, so it upgrades existing collections.

`collection_for_tenant(tenant_id)` is the single place that decides which collection holds a tenant's data. Today it returns one shared collection, isolated by the tenant filter; returning a per-tenant name (and provisioning it) is the whole change needed for collection-per-tenant isolation.

`search()` **requires a principal** and builds the entitlement filter itself; there is no way to run an unfiltered search. The Kafka worker refuses any chunk that has no ACL fields.

---

## Online Pipeline — Query & Answer

Triggered by `POST /query`. Components are created once in `lifespan`. The whole synchronous pipeline runs in a worker thread so concurrent queries overlap.

```
POST /query {"query": "...", "top_k": 5, "source_in": [...]?, "web_search": bool?}
        │
        ▼  authenticate · scope rag:query · tenant policy
  1. investment-advice request?  ──► declined (no retrieval, no model call)
  2. rewrite → embed (Redis cache)
  3. ENTITLED retrieval: vector pre-filter + per-tenant BM25 (re-checked in the store) → RRF
  4. guard: ACL re-check + trust filter  ──► audit anomaly if the pre-filter was bypassed
  5. rerank (cross-encoder)
  6. weak recall? → web search (tenant ceiling ∧ rag:web ∧ not opted out; verified publishers only if required)
  7. context = the chunks that fit the budget; [n] is position n
  8. generate (hardened prompt)
  9. evaluate: LLM grade (grounded? useful?) + deterministic verification (citations, figures)
        └─ failing → regenerate with specific feedback (bounded) → web top-up once → re-evaluate
 10. policy: warn or abstain, per tenant      11. sanitise output      12. disclaimer
        │
        ▼
{"answer", "sources" (public fields only), "web_search_used", "grounded", "answer_useful",
 "generation_attempts", "trust_summary", "warnings", "verification", "needs_review"}
```

---

### Step 1 — HTTP Request hits the Gateway

**File:** `online_pipeline/api/gateway.py`
**Framework:** FastAPI

At startup `lifespan` validates the configuration and **refuses to start** if it is unsafe (`AUTH_MODE=disabled` or `dev_token` outside `APP_ENV=dev`, an `http://` Okta issuer, no audit key or Qdrant key in production, a dev secret under 32 characters). It then loads the tenant policies, writes a `service.start` audit event (so a broken audit log stops the service at boot), and creates the singletons.

Every route except `/health` depends on `require_scope(...)`: it validates the token, checks the scope, and records `auth.fail` / `authz.fail` events (token text is never logged).

```python
class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")   # a caller cannot send a filter of their own
    query: str
    source_in: list[str] | None = None          # narrows within what you may read, never widens
    top_k: int = settings.top_k_rerank
    chat_history: list[dict] = []
    web_search: bool | None = None              # can only opt OUT of web search
```

**Endpoints**

| Method & path | Scope | Purpose |
|---|---|---|
| `GET /health` | none | liveness, and whether lexical retrieval is live |
| `POST /query` | `rag:query` | the pipeline above |
| `POST /ingest` | `rag:ingest` | upload files (optional `acl_groups`, `acl_public`, `classification`, `as_of`, `expected_sha256`, `supersedes`) |
| `POST /ingest/url` · `/ingest/api` | `rag:ingest` | fetch through `SafeFetcher`; also need the tenant's `allow_url_ingest` and allow-list |
| `GET /documents` | `rag:ingest` | your documents with status (`?all=true` needs `rag:admin`) |
| `PATCH /documents/{id}/acl` | `rag:ingest` (owner) / admin | change who can read it; effective on the next query |
| `DELETE /documents/{id}` | owner / admin | delete chunks and the original (unless another document uses the bytes); blocked by legal hold |
| `POST /admin/documents/{id}/legal-hold` | `rag:admin` | place or lift a hold |
| `GET /admin/quarantine` · `POST /admin/quarantine/{id}/release` | `rag:admin` | review documents the injection screen held back |
| `GET /audit` | `rag:audit` | your tenant's audit events, paged by `after_seq` |

A document you cannot manage is reported as **404, not 403**, so other people's documents are not disclosed.

---

### Step 2 — LLM Router selects a provider

**File:** `online_pipeline/llm/router.py`  
**Class:** `LLMRouter`

Before any LLM call is made, the router decides which provider and model to use, and which fallbacks to try if the primary fails.

```
LLMRouter(routing_mode="balanced")
    │
    ├── _route_for("query_rewrite")
    │       → ModelRoute(provider="groq", model="openai/gpt-oss-20b", temperature=0.0)
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

VectorRetriever.retrieve(query_vector, principal, top_k=20, source_in)   # principal is mandatory
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

#### Entitlement filtering and the lexical branch

The vector branch passes the caller's rule to Qdrant as a filter (tenant · clearance · owner/public/group · not quarantined · not superseded). The lexical branch (**BM25**) keeps **one index per tenant**, so another tenant's text can neither appear nor skew term statistics; the same rule is applied *before* ranking so `top_k` fills with permitted chunks, and every BM25 hit is re-checked against the store's *current* payload, so a revoked ACL or deleted document stops matching immediately rather than after the next index rebuild (if the store cannot be read, the lexical branch contributes nothing for that request). The index is rebuilt in a worker thread after each ingest.

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

### Step 6b — Web Search Fallback

**Files:** `online_pipeline/retrieval/web_search.py` (`TavilyWebSearch`), `gateway._answer_query`

Two triggers: local recall is **weak** (fewer than `web_search_min_local_results` candidates, or the best cross-encoder score is below `web_search_min_relevance_score`), or after generation the answer failed grading or verification and the web has not been tried.

It runs only if **tenant policy allows it AND the user holds `rag:web` AND the request did not opt out**. A blocked search that was wanted becomes a `policy.deny` audit event. When a tenant requires verified sources, Tavily is called with `include_domains` set to the verified publishers. Web hits are stamped with an explicit public-to-this-tenant ACL and a **host-based trust tier** (never anything the page claims), then reranked together with local chunks and pass through the same guard. Chat history is never sent to the web.

---

### Step 7 — Context Building

**File:** `online_pipeline/context_builder/builder.py`

`select(chunks)` decides which chunks fit the token budget, and `build()` formats exactly those as `[1] Source: …`, `[2] …`. `select()` is the **numbering authority**: the verifier checks `[n]` against the same list, so a citation can never point at a chunk the model was not shown. Table and fact chunks are budgeted by characters (`len/3`) rather than words, because numbers tokenise badly ("391,035" is several tokens), so a table can no longer overrun the model's budget.

---

### Step 8 — LLM Answer Generation

**File:** `online_pipeline/llm/llm_client.py`

The prompt treats everything retrieved as **data, not instructions**:

```
<conversation> … </conversation>
<retrieved_context> [1] Source: … </retrieved_context>
Latest user question: …                    (+ "Reviewer feedback: …" on a retry)
```

The system prompt says text inside the tags may try to change the rules, reveal the prompt, call tools or emit links/images, and must never be followed; that figures must be quoted exactly with their period and units; that every statement cites `[n]`; and that derived figures must be accompanied by their inputs. Untrusted text is **defanged** before it is inserted, so a chunk containing `</retrieved_context>` cannot close the data block.

Providers, routing and fallbacks are unchanged (see Step 2). `answer ONLY from context` is still the first rule: RAG is only worth anything if answers come from *your* documents.

---

### Step 9 — Evaluation: grading and figure verification

**Files:** `online_pipeline/grading/grader.py`, `online_pipeline/answer_safety/verify.py`

Each draft is evaluated two ways.

**The semantic grader** (one LLM call, JSON) returns `grounded` (every claim supported by the context) and `answers_question` (a bare "the context lacks this" does *not* answer it). It fails open: if every grader provider is down the answer is returned ungraded.

**The deterministic verifier** needs no model:

1. every `[n]` must exist among the chunks actually in the prompt;
2. every sentence containing a figure must carry a citation;
3. every figure must match a value in a **cited** chunk, scale-aware and within the precision the answer itself states (`$391.0 billion` matches 391,035 million, not 391,535);
4. the **period** must agree (a 2023 value quoted for 2024 fails) using the table column or the XBRL period;
5. the **currency** must agree;
6. a stated **direction or sign** (`fell 2%`, `-2%`) must match the source's, whether the source is a table sign or prose ("declined 3%");
7. a figure not in the sources is accepted only as a **derived** figure: a change %, difference, sum or ratio of figures in the same sentence, tolerating the rounding of those inputs (`up 2.0% from $383.3B to $391.0B`); `9.9%` for that sentence fails.

Labels and trivia are ignored (years, `FY2024`, `Q3`, dates, `Item 7`, list numbers, bare small integers). An honest "I can't find that" is never penalised.

```
draft → evaluate → failing and attempts < MAX?  ──► regenerate WITH specific feedback ("the figure '$391.5 billion'
                                                     does not appear in the cited sources …") → evaluate
        still failing and web not tried and allowed? ──► web top-up → regenerate once → evaluate
```

Then the tenant policy applies (see Step 10). The verifier checks **numbers, not meaning**; see [Known limitations](#known-limitations).

---

### Step 10 — Policy, output safety and review flag

**Files:** `gateway._apply_answer_policies`, `answer_safety/sanitize.py`, `answer_safety/advice.py`

- **Warn or abstain**, per tenant, separately for ungrounded text and for unverifiable figures. *Warn* prepends a note naming the figures; *abstain* replaces the answer with a refusal. Financial tenants default to abstain.
- **Sanitise**: the answer never keeps a markdown image, raw HTML, a `javascript:`/`data:` URI, or a URL whose host is not one of the answer's own sources. This closes the usual way a poisoned document leaks data (an image URL carrying the secret).
- **Investment advice** requests ("should I buy…") are declined before retrieval when the tenant policy says so. Questions *about* ratings or targets in a filing are not affected.
- **Disclaimer** from the tenant policy is appended.
- **`needs_review`** is true when restricted (classification ≥ 2) sources were used, a figure failed verification, or the answer was ungrounded. `warnings` lists `unverified_sources_used`, `figures_unverified`, `ungrounded`, `output_sanitized`.
- `sources` contain only public fields (origin, trust tier, hash, period…), **never** owner, groups or tenant.

The audit event for the query records counts and flags (how many figures verified, which policy applied), not the figures or the text.

---

### Observability (LangSmith)

**File:** `online_pipeline/observability/tracing.py`

Every stage (`query_rewrite`, `embed_query`, `hybrid_retrieve`, `rerank`, `tavily_web_search`, `build_context`, `generate_answer`, `grade_generation`, `verify_answer`) is traced under one `rag_query` run. Tracing activates only with `LANGSMITH_TRACING=true` and a key. By default **prompts, retrieved text and answers are hidden** (`LANGSMITH_HIDE_INPUTS/OUTPUTS`), leaving names, timings and ids, because those would otherwise leave for a third party.

---

## Document management and lifecycle

**Files:** `online_pipeline/lifecycle.py`, `scripts/backfill_acl.py`, `scripts/purge_expired.py`

The API endpoints and the maintenance scripts share the same functions, so they cannot disagree.

- **Change who can read a document** (`PATCH …/acl`): takes effect on the next query (the lexical branch re-checks the store). Owners can *add* only groups they belong to (groups already on the document are kept) and classify up to their own clearance; an admin may use any group; **nobody can exceed the tenant's classification ceiling**. The change is audited with before and after.
- **Delete**: removes the chunks, and the stored original only if no other document of the tenant uses the same bytes.
- **Legal hold**: a held document cannot be deleted by anyone, nor by the retention job.
- **Supersession**: uploading with `supersedes=<doc_id>` hides the replaced version once the successor is live. Only the owner or an admin may supersede; a quarantined successor does not hide the old version.
- **Retention**: `scripts/purge_expired.py --days N` (dry-run by default) deletes documents older than N days, skips held ones, never guesses at documents with no timestamp, and audits each deletion.
- **Legacy data**: points indexed before access control have no ACL, so they are invisible to everyone. `scripts/backfill_acl.py --tenant T --owner U --apply` gives them an owner (private to that owner, internal, `unverified`); points that already have an ACL are never touched, so re-running is safe.

---

## Infrastructure Services

`docker-compose.yml` runs the whole application (project name `production-rag`):

| Service | Host port (default) | Role |
|---|---|---|
| **api** | 8000 | the FastAPI gateway, built from `Dockerfile` (non-root, models baked in, healthcheck) |
| **frontend** | 8080 | the UI; nginx serves the build and proxies `/query /ingest /health /admin` to `api`, so the browser only talks to one origin |
| **qdrant** | 6333, 6334 | vector database (`/dashboard` on 6333) |
| **redis** | 6379 | embedding cache |
| zookeeper, kafka | 2181, 9092 | **optional**, profile `kafka`: asynchronous ingestion |

Every host port is an environment variable (`API_PORT`, `FRONTEND_PORT`, `QDRANT_PORT`, …). `scripts/docker-up.sh` checks each one **before starting anything**, reports who is holding a busy port (a Docker container or a local process) and either picks the next free port (default) or stops (`--strict`). Ports held by this stack's own containers are not conflicts. Volumes: `qdrant_storage`, `audit_logs`, `hf_cache`. See the README for commands.

---

## Operations runbook

| Task | Command |
|---|---|
| Start everything (checks ports) | `./scripts/docker-up.sh` (`--dev-auth` for local tokens, `--kafka` for Kafka, `--check-only` to just check ports) |
| Mint a dev token | `docker compose exec api python scripts/make_dev_token.py --tenant dev --groups finance --clearance 3 --scopes rag:query,rag:ingest,rag:web,rag:admin` |
| Verify the audit log was not tampered with | `python scripts/verify_audit_chain.py [path]` (exit 1 and the first bad `seq` if so) |
| Give pre-ACL data an owner | `python scripts/backfill_acl.py --tenant legacy --owner admin@acme.com` then `--apply` |
| Apply retention | `python scripts/purge_expired.py --days 2555` then `--apply` |
| Review quarantined documents | `GET /admin/quarantine`, then `POST /admin/quarantine/{doc_id}/release` |
| Edit tenant policy | edit `config/tenant_policies.yaml` (mounted into the container), restart `api` |

**Okta setup.** On the Okta authorization server: set the audience (`OKTA_AUDIENCE`), define the scopes above, and add custom claims `tenant_id` (a single string), `groups` and `clearance` (an integer 0–3) to the access token. A token without a single `tenant_id` is rejected. Set `OKTA_ISSUER` to the server's https URL; keys are fetched from its JWKS endpoint.

---

## Known limitations

These are deliberate boundaries or things not yet built, stated so nobody relies on more than is there.

- **The verifier checks numbers, not meaning.** "Net sales rose 2.0% from $391.0B to $383.3B" contradicts itself, yet each figure matches a real source value, so it passes; a correct number attached to the wrong claim passes too. The semantic grader is the layer for that, and it is an LLM.
- **No stated period means nothing to contradict**: a figure quoted without a year can match the wrong year's column.
- **The injection screen is a heuristic.** It stops blunt attacks; the prompt delimiters, output sanitiser and quarantine are further layers, not guarantees. Text hidden by an *external* stylesheet is invisible to the HTML sanitiser.
- **PDF tables**: ruled tables and borderless numeric statements are handled; a statement with a trailing text column (e.g. a "Status" column) is kept as prose rather than risk mis-parsing digit-heavy prose; tables that continue across pages are not merged; scanned pages get OCR text but no table structure; PDFs have no heading detection (HTML and DOCX do).
- **Invisible-text filtering in PDFs** does not see "render mode 3" text (legitimate OCR layers use it too).
- **The sandbox limits memory and CPU and scrubs the environment, but does not block the network** (macOS offers no simple way); it relies on the parser needing none. On macOS `RLIMIT_AS` is not enforced, so a polling watchdog backs it up.
- **Rate limiting is in process**: with several workers the effective limit is multiplied.
- **The BM25 index is in memory** and rebuilt after ingest; fine to tens of thousands of chunks, after which it belongs in the store as sparse vectors. Newly ingested chunks reach the lexical branch after the rebuild (the vector branch sees them at once).
- **Audit tail truncation is not detectable** from the file alone; anchor the latest hash externally if that matters.
- **Restricted data still goes to whichever LLM provider you configure.** Entitlement filtering does not make third-party model use acceptable for restricted classes; that needs privately hosted models and per-classification egress rules (not built).
- **Compliance**: these are *controls* (access control, tamper-evident audit, retention, erasure, legal hold). Mapping them to FINRA/SEC/SOC 2/GDPR is for your compliance team; nothing here is a certification.
- The UI has a token box for development; a real Okta sign-in (PKCE) flow is not included.
- One mutation survived on this interpreter: removing the explicit IPv4-mapped-address unwrap changes nothing on Python 3.13+, whose standard library already handles it. It still matters on the Python 3.12 Docker image.

---

## Testing and verification

`pytest tests/` runs the whole suite (about 930 tests, ~80 s; the sandbox, PDF and XML tests are the slow ones). Highlights:

- **Isolation matrix**: 300 random (document, principal) pairs against Qdrant's real filter engine; the store filter and the Python guard must agree on every one.
- **Hostile inputs** are built at test time: PDFs with JavaScript, launch actions, embedded files, a 200 MB zlib bomb, cyclic page trees; DOCX with macros, XXE, zip bombs; SSRF targets, DNS rebinding, redirect chains, gzip bombs.
- **Sandbox mechanics**: infinite loops, memory hogs, crashes, segfaults and garbage output are all contained and reported.
- **Backward compatibility**: documents without tables chunk byte-identically to before.
- **Mutation checks**: for each phase a protection was deliberately broken, one at a time (≈70 mutants), and the suite had to fail. Doing this found real bugs, among them a path-traversal in the raw store (tenant id `..`), a verifier that was sign-blind, month-name matching that swallowed `decreased 1.8%`, and a "to … from" detector that could not span a decimal point.

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

### Defence in depth, with two independent implementations

The access rule exists as a store filter and as a Python predicate; the injection defence is a screen, a data-delimited prompt and an output sanitiser; the number check is a grader and a verifier. Each layer is allowed to be imperfect because the next one does not share its blind spot, and a disagreement between layers is itself an audited signal.

### Fail closed, with one deliberate exception

Unknown tenant → most restrictive policy. Missing ACL → invisible. Missing audit log → no service. Unparseable file → rejected. Unreadable store during a BM25 re-check → the lexical branch returns nothing. The one fail-*open* is the semantic grader: if every grader provider is down the answer is returned ungraded and flagged, because refusing all answers during an LLM outage is a worse failure than an unchecked one — and the deterministic verifier still runs.

### Why parse in a separate process

A PDF or DOCX parser is a large amount of C-adjacent code reading attacker-controlled bytes. In-process, one bad file can hang or crash the API or exhaust its memory. In a throwaway process with its own CPU/memory/time limits and no secrets in its environment, the worst case is one rejected upload.

### Why store the original bytes under their hash

An answer is only as trustworthy as your ability to show what it came from. The chunk carries the file's SHA-256 and the raw-store holds the file under that hash, so any figure can be traced to the exact bytes, and a retention or deletion request can remove them.
