---
title: Secure financial RAG — access control, audit, provenance, safe ingestion, table-aware parsing, answer safety
status: DRAFT            # v2 — awaiting approval; nothing below is implemented
risk-tier: R4            # a defect leaks one tenant's data, ingests hostile content, or states a wrong financial figure
size: L (program)        # ship as one branch + PR per phase
supersedes: access-control-entitlements.md (v1)
---

# Secure financial RAG — program plan (v2)

v1 covered access control only. This version adds what was decided afterwards:

- per-user scopes **plus** a per-tenant ceiling,
- audit and compliance,
- source provenance for authenticity,
- sanitised, SSRF-safe ingestion (`/ingest/url` and the other ingest paths),
- a table-aware chunker for earnings tables and XBRL,
- finance-grade answer safety.

## 0. Assumptions (not yet confirmed — correct me, they are cheap to change now)

| Decision | Assumed default | Why |
|---|---|---|
| Identity provider | Generic OIDC/JWT via JWKS (works with Okta, Entra, Keycloak, Auth0) | Config-only; no vendor code. Dev uses a local HS256 token. |
| Tenancy | One collection + `tenant_id` payload filter, behind a `collection_for(principal)` seam | Cheapest now; collection-per-tenant later is a config change. |
| Classification | 0 public · 1 internal · 2 confidential · 3 restricted | Common scheme. |
| Existing indexed data | Backfilled into a `legacy` tenant, owner = admin, **default-deny until an admin assigns it** | Safe by default. |
| Compliance posture | Provide the *controls* (tamper-evident audit, retention, erasure). **Not** a certification: FINRA/SEC/SOC 2 mapping needs your compliance team. | I can't certify anything. |

## 1. Verified findings in the current code (`feat/generation-grading`)

| # | Finding | Where |
|---|---|---|
| F1 | No authentication on any route | `gateway.py` |
| F2 | `metadata_filter` is client-supplied and passed to `Filter(**…)` — a caller picks their own filter | `QueryRequest`, `vector_db/client.py` |
| F3 | Chunks carry no tenant/owner/group/classification | `_index_document` |
| F4 | `doc_id = sha256(text)`; point id from `doc_id+chunk_index` → identical uploads from two tenants **share point ids** and overwrite each other's payload | `metadata_enricher.py`, `gateway.py` |
| F5 | `HybridRetriever(vector_db=…)` has no corpus → **BM25 is not active**; the architecture doc says it is. BM25 also has no filter hook | `gateway.py:104` |
| F6 | `WebExtractor` does `requests.get(url, timeout)`: follows redirects, no scheme/host/IP check, no size cap, no content-type check → SSRF, memory exhaustion | `web_extractor.py` |
| F7 | Scraped/ingested text goes straight into the LLM prompt → **indirect prompt injection** (hidden text, "ignore previous instructions", exfil links) | extractors → `ContextBuilder` → `LLMClient` |
| F8 | `get_text(separator="\n")` puts every table cell on its own line; `TextCleaner` collapses runs of spaces — together they **destroy table structure** | `web_extractor.py`, `cleaner.py` |
| F9 | `pypdf` is text-only; no table extraction, no `lxml`/`pdfplumber`/`defusedxml` in requirements; no XBRL support | `document_extractor.py`, `requirements.txt` |
| F10 | `ContextBuilder` budgets by word count and can cut between chunks; fine for prose, harmful for tables | `builder.py` |
| F11 | The grader is an LLM judging an LLM; no deterministic check that a number in the answer exists in the sources | `grading/grader.py` |
| F12 | Full query/context/answer go to LangSmith when tracing is on; web search and graders send content to third parties | `tracing.py`, `web_search.py` |
| F13 | No record of who asked, ingested, changed ACLs or triggered web search | — |

## 2. Core design

### 2.1 Principal
JWT claims → `Principal(user_id, tenant_id, groups, clearance, scopes)`. `current_principal` dependency on every route except `/health`. `AUTH_MODE=disabled` honoured only when `APP_ENV=dev`; otherwise startup aborts (fail closed). 401 for bad/missing/expired token, 403 for missing scope.

Scopes: `rag:query`, `rag:ingest`, `rag:web`, `rag:admin`, `rag:audit`.

### 2.2 Tenant policy ceiling (new)
`config/tenant_policies.yaml`, loaded and validated at startup. Unknown tenant ⇒ the most restrictive policy.

```yaml
default: {allow_web_search: false, allow_url_ingest: false, url_allowlist: [],
          max_classification: 1, require_verified_sources: true,
          ungrounded_answer_policy: abstain, numeric_mismatch_policy: abstain}
tenants:
  equities-research: {allow_web_search: true, allow_url_ingest: true,
                      url_allowlist: [sec.gov, www.sec.gov], max_classification: 3,
                      require_verified_sources: false, ungrounded_answer_policy: warn}
  investment-banking: {allow_web_search: false, allow_url_ingest: false, max_classification: 3}
```

**Effective permission = tenant policy ∧ user scope ∧ per-request flag.** The tenant is a ceiling a user scope can never exceed; the request flag can only opt *out*.
`allow_web = policy.allow_web_search and "rag:web" in scopes and request.web_search is not False`.
Applies to both web triggers (weak-recall gate and post-grading top-up), URL ingest, and the maximum classification a tenant may assign.

### 2.3 Document ACL and enforcement (unchanged from v1)
Payload per chunk: `tenant_id, owner_id, acl_groups[], acl_public, classification`. Readable iff tenant matches ∧ `classification ≤ clearance` ∧ (owner ∨ public ∨ group overlap). Missing ACL ⇒ deny.
Two layers: (1) Qdrant pre-filter built **server-side** from the principal (nested filter verified to return exactly the expected ids on Qdrant's in-memory engine); (2) `authorize(chunks, principal)` guard on every candidate list after retrieval and after any web top-up. `metadata_filter` is removed from `QueryRequest`; an optional typed `source_in` is ANDed, never substituted.

### 2.4 Provenance (new — "sources of the documents for authenticity")
Every document gets an immutable provenance record, copied onto each of its chunks:

| Field | Meaning |
|---|---|
| `source_type` | `upload` · `url` · `api` · `xbrl` |
| `origin` | filename, or **final URL after redirects** |
| `content_sha256` | hash of the **raw bytes** as received (not the cleaned text) |
| `fetched_at`, `ingested_by` | UTC time, principal |
| `publisher` / `domain` | e.g. `sec.gov` |
| `trust_tier` | `verified_primary` (allow-listed regulator/issuer domain, or upload with a matching declared checksum/accession number) · `internal` · `unverified` |
| `external_id` | e.g. SEC accession number, filing form type |
| `as_of` / `fiscal_period` | what period the content describes |
| `supersedes` | doc_id of the version it replaces (restatements) |

- Trust tier is **computed by the server** from the allow-list and checks, never accepted from the client.
- Optional `expected_sha256` on upload/URL: mismatch ⇒ reject (authenticity of a document the user already trusts).
- Raw originals kept in a pluggable `RawStore` (filesystem default, S3 optional) keyed by `content_sha256`, so a figure can be traced back to the exact bytes it came from.
- Answers and `sources[]` carry `trust_tier`, `origin`, `as_of`. With `require_verified_sources`, only `verified_primary`/`internal` chunks may feed an answer.

### 2.5 Safe ingestion (new)
All fetches (`/ingest/url`, `/ingest/api`, XBRL taxonomy fetches, web search results are separate) go through **one** `SafeFetcher`:

1. **URL policy**: `https` only (http behind a flag); no userinfo; ports 443; host must match the tenant `url_allowlist`; otherwise 403.
2. **SSRF**: resolve DNS, reject loopback / RFC1918 / link-local (incl. `169.254.169.254`) / IPv6 equivalents / multicast; connect to the **validated IP** (defeats DNS rebinding); re-validate on **every redirect**, max 3 hops.
3. **Resource limits**: connect+read timeouts; streamed download with a hard byte cap (default 10 MB); decompression-bomb guard; per-principal rate limit.
4. **Type checks**: allow-list on declared `Content-Type` **and** magic bytes (HTML, PDF, JSON, XML/iXBRL); anything else rejected.
5. **HTML sanitisation**: remove `script/style/iframe/object/embed/form/svg/meta/noscript`, comments, and **hidden text** (`display:none`, `visibility:hidden`, `aria-hidden`, zero/tiny font, off-screen positioning, white-on-white), which is the main prompt-injection carrier.
6. **Text hygiene**: strip zero-width and bidi-override characters, NFKC, control chars (table whitespace preserved — see 2.6).
7. **Injection screening**: heuristic detector for instruction-like content (role markers, "ignore previous instructions", "system prompt", tool-call syntax, markdown/HTML image or link exfil patterns, base64 blobs). Hits ⇒ chunk stored with `quarantined=true`, **excluded from retrieval** until a `rag:admin` releases it; an audit event is written. This is a heuristic filter, not a guarantee — hence the generator hardening in 2.7 as a second layer.
8. **Parser safety**: XML via `defusedxml`/`lxml` with entity resolution and network disabled (XXE); DOCX zip size/entry-count limits; PDF: reject encrypted files, cap pages and OCR pages; JSON depth/size caps.
9. **Uploads**: extension must match magic bytes; size cap; optional malware-scan hook (ClamAV interface, off by default).
10. `/ingest/api`: callers may not set `Authorization`/`Host` headers on the outbound request.

### 2.6 Table-aware and financial chunking (new)
**Root change:** extractors stop returning one flat string and return an ordered list of **blocks**: `Heading | Paragraph | Table | Footnote`. The cleaner becomes block-aware (no whitespace collapsing inside tables), and the router chunks per block type, preserving the heading path (e.g. `Item 7 > Results of Operations`).

| Source | Table extraction |
|---|---|
| HTML (web, 10-K, earnings release) | parse `<table>` with `lxml`/`bs4`: header rows, row labels, `colspan/rowspan`, units/scale ("in millions"), parenthesised negatives |
| PDF | `pdfplumber` (lattice + stream strategies); fall back to text; scanned pages via OCR flagged `low_confidence` |
| DOCX | `python-docx` tables |
| **Inline XBRL** (`ix:nonFraction`, `ix:nonNumeric`) in 10-K/10-Q HTML | parse facts: concept QName, value, `scale`, `sign`, `decimals`, unit, **context → entity + period + dimensions** |
| XBRL instance (`.xml`) | same fact model |

`TableChunker` rules (all unit-tested):
- never split a table mid-row; each chunk repeats **caption + header rows + units**; wide tables split by column group, always keeping the row-label column;
- two representations: the **embedded text** is a linearised, self-describing row (`Revenue | FY2024 | USD millions | 1,234 (restated)`), the original Markdown table is stored in the payload for display;
- structured numerics in the payload (`value`, `unit`, `scale`, `period`, `concept`) so answer safety can verify against data rather than prose;
- footnotes stay attached to their table;
- XBRL facts grouped by statement (balance sheet, income, cash flow) into fact chunks with human-readable labels (taxonomy labels optional; fallback humanised concept name); duplicates deduped; sign and scale applied exactly once;
- router `auto` selects `table_aware` when blocks include tables or the file is (i)XBRL; mixed documents send prose to the recursive chunker. Existing `fixed_token` / `recursive` behaviour is unchanged.
- `ContextBuilder` gains a **never-truncate-a-table** rule and a token (not word) estimate for table chunks.

Out of scope: full XBRL taxonomy resolution/validation (Arelle). Revisit if label/linkbase fidelity matters.

### 2.7 Answer safety (new)
Layered, deterministic checks first, LLM judgement second:

1. **Prompt hardening**: context wrapped in explicit data delimiters; system prompt states that context is untrusted data and instructions inside it must be ignored; chat history likewise.
2. **Citation enforcement**: every sentence with a factual or numeric claim needs `[n]`; every `[n]` must exist. Violations ⇒ regenerate once, then policy.
3. **Numeric verification (deterministic)**: extract numbers from the answer; normalise (commas, `(x)` negatives, `%`, currency, million/billion scale); each must match a value in a **cited** chunk's structured payload (or its text). Result is recorded per number: `verified | not_found | unit_mismatch | period_mismatch`. Derived figures (growth %, margins) are accepted only when the system computes them from cited values via a small deterministic calculator and shows the inputs.
4. **Entity/period/unit consistency**: the answer's period and entity must match the cited facts' context; mixing fiscal periods or restated vs original values is flagged.
5. **Source trust gate**: answers using `unverified`/web sources are labelled, or refused when `require_verified_sources`.
6. **Output sanitisation**: strip HTML, markdown images and links to hosts not present in `sources` (closes the exfiltration channel).
7. **Policy**: `numeric_mismatch_policy` / `ungrounded_answer_policy` per tenant (`warn` | `abstain`); finance tenants default to `abstain`. Existing LLM grader stays as the semantic layer.
8. **Optional controls**: configurable "not investment advice" footer; refuse trade/price-target recommendations when policy says so; `needs_review=true` flag for classification ≥ 2 or any verification failure (the review queue itself is out of scope).
9. Response gains a `verification` object: citations, per-number checks, trust tiers.

### 2.8 Audit and compliance (new)
Append-only, **hash-chained** JSONL events (each carries the previous event's hash ⇒ tamper-evident), behind an `AuditSink` interface (file default; Postgres / object-lock S3 pluggable).

- **Events**: `auth.fail`, `query` (user, tenant, **query hash**, returned doc ids, filtered-out count, web used, grounded, verification result, policy applied), `ingest` (provenance, trust tier, quarantine), `acl.change`, `doc.delete`, `quarantine.release`, `policy.deny` (why: tenant ceiling / scope / SSRF block), `audit.read`.
- **Minimisation**: ids and hashes, not content; no document text or answers in logs.
- `GET /audit` (`rag:audit`, tenant-scoped); `scripts/verify_audit_chain.py` detects gaps/edits.
- **Retention + erasure**: configurable retention; `DELETE /documents/{id}` removes chunks and the raw object, and writes a deletion event; legal-hold flag blocks deletion.
- LangSmith: `langsmith_hide_content=true` by default (hide inputs/outputs, keep stage names, latencies, ids).
- Qdrant API key required outside dev.

## 3. Phases (one branch/PR each; each ends green)

| Phase | Content | Depends on | Size |
|---|---|---|---|
| **P0** | Principal + JWT validation, scopes, fail-closed dev mode, tenant-policy loader, frontend bearer token | — | M |
| **P1** | Audit core: hash-chained sink, `emit()` used by every later phase, verify script | P0 | M |
| **P2** | ACL stamping at ingest, tenant-scoped doc/point ids (fixes F4), payload indexes, validation that users grant only their own groups/clearance (capped by tenant `max_classification`) | P0 | M |
| **P3** | Entitlement-filtered retrieval: server-built filter, `authorize()` guard, remove `metadata_filter`, ACL-aware BM25 hook, tenant-ceiling web gating, correct the BM25 claim in the architecture doc | P2 | M |
| **P4** | `SafeFetcher`, HTML sanitisation, text hygiene, injection screening + quarantine, parser hardening, upload checks | P1 | L |
| **P5** | Provenance record, trust tier, `expected_sha256`, `RawStore`, provenance in sources/answers | P4 | M |
| **P6** | Block-based extraction, block-aware cleaner, `TableChunker`, iXBRL/XBRL fact parser, router + context-builder changes | P4 | L |
| **P7** | Answer safety: prompt hardening, citation + numeric verification, calculator, output sanitisation, per-tenant policy, `verification` in response | P5, P6 | L |
| **P8** | Legacy backfill, `PATCH /documents/{id}/acl`, `DELETE /documents/{id}`, retention/erasure, LangSmith redaction, Qdrant key | P3, P1 | M |

P4 can start in parallel with P2/P3 (it only needs P1).

## 4. Acceptance criteria (excerpt — each phase adds its own tests)

```gherkin
Feature: Tenancy and entitlements
  Scenario: Tenant isolation / group / clearance / default-deny / revocation   # as in v1
  Scenario: Identical uploads from two tenants keep distinct ids and ACLs
  Scenario: Client cannot widen access with any filter field

Feature: Tenant ceiling
  Scenario: User has rag:web but tenant disallows web search
    Then no web search happens on either trigger and the audit log records policy.deny
  Scenario: Tenant allows web search but user lacks rag:web
    Then no web search happens
  Scenario: Request sets web_search=false
    Then no web search happens even when tenant and scope allow it

Feature: Safe ingestion
  Scenario Outline: SSRF targets are refused
    Examples: http://127.0.0.1, https://169.254.169.254/, https://10.0.0.5, a hostname resolving to a private IP,
              a public URL that redirects to a private IP, a redirect chain longer than 3
  Scenario: Oversized or wrong-type response is rejected before being fully read
  Scenario: Hidden text containing "ignore previous instructions" never reaches an indexed chunk unquarantined
  Scenario: XML with an external entity is parsed without resolving it
  Scenario: Host outside the tenant allow-list is refused

Feature: Provenance
  Scenario: Trust tier is computed server-side; a client-supplied tier is ignored
  Scenario: expected_sha256 mismatch rejects the document
  Scenario: With require_verified_sources an unverified chunk is excluded from the answer

Feature: Table-aware chunking
  Scenario: No table row is ever split across chunks; every chunk repeats caption, headers and units
  Scenario: Parenthesised negatives, scale ("in millions") and percentages are preserved exactly
  Scenario: An iXBRL fact keeps concept, period, unit, scale and sign
  Scenario: Existing fixed_token and recursive chunking output is byte-identical to before

Feature: Answer safety
  Scenario: A figure in the answer that is not in a cited chunk fails verification and is refused (finance policy)
  Scenario: Answer cites FY2024 but the cited fact is FY2023 → period_mismatch
  Scenario: A markdown image or link to an unlisted host in the model output is stripped
  Scenario: Derived growth % is computed by the calculator from cited values, inputs shown

Feature: Audit
  Scenario: Every query/ingest/ACL change/delete/policy denial produces one chained event
  Scenario: Editing or deleting a past event makes verify_audit_chain fail
  Scenario: Audit events contain ids and hashes, never document text
```

## 5. Test strategy
- **Isolation matrix** on `QdrantClient(":memory:")`: tenants × groups × classifications × public/owner; the Qdrant filter and the Python guard must agree on every row (property test over random ACLs).
- **SSRF suite** with an injected resolver (no real network): table of hostile URLs, redirect chains, rebinding.
- **Golden fixtures** for tables: a small earnings-release HTML, a PDF with a ruled table, a DOCX table, and an iXBRL snippet; asserts on row integrity, header repetition, and exact numerics.
- **Adversarial prompt-injection corpus** (hidden text, role markers, exfil markdown) run through ingest → retrieval → generation.
- **Numeric verifier** unit tests (formats, scale, negatives, rounding) plus a replay of the generation-grading flow tests with finance policy.
- Existing 60 tests stay green; the multi-file upload and all chunking strategies keep their current behaviour (checked by a snapshot test).

## 6. New dependencies (to approve)
`PyJWT[crypto]`, `pdfplumber`, `lxml`, `defusedxml`; optional `boto3` (S3 raw store/audit sink) and a ClamAV client. No Arelle in this program.

## 7. Risks and non-goals
- **Heuristics are not guarantees.** Injection screening and numeric verification reduce risk; they do not eliminate it. Layers and quarantine exist because any single check can miss.
- **Table extraction from PDFs is imperfect**, especially scanned or borderless tables; low-confidence tables are flagged, and numeric verification is the backstop.
- **Compliance**: this builds controls (tamper-evident audit, retention, erasure, access control). Mapping to FINRA 17a-4, SEC, SOC 2, GDPR needs your compliance/legal team; I will not claim certification.
- **Migration**: P2 changes id derivation; documents already indexed are covered by the P8 backfill (old ids kept), only new ingests use tenant-scoped ids.
- LLM providers still see whatever context a principal may read; for restricted data you need privately hosted models — per-classification egress rules are a follow-up.
- Non-goals: row-level encryption/BYOK, full XBRL taxonomy validation, human review queue UI, evaluation harness (own plan), DLP/PII redaction of answers.

## 8. Recommended start
**P0 + P1** (auth, tenant policy, audit core): everything later emits audit events and consults the policy, so building them first avoids retrofitting.
