---
title: Access control and entitlement-filtered retrieval
status: DRAFT            # awaiting approval — nothing below is implemented
risk-tier: R4            # security boundary: a defect leaks one tenant's documents to another
branch: feat/access-control   # to be cut from feat/generation-grading once that merges
---

# Access control and entitlement-filtered retrieval

## 1. Problem — what the code does today

Verified against `feat/generation-grading` (b70fe63):

| # | Finding | Where | Consequence |
|---|---|---|---|
| F1 | No authentication on any route | `gateway.py` | Anyone who can reach the port can query, ingest, scrape URLs. |
| F2 | `metadata_filter` is a **client-supplied** field passed straight into `Filter(**...)` | `QueryRequest`, `vector_db/client.py:search` | The caller chooses their own filter, or none. Any access rule built on it is bypassable. |
| F3 | Chunks carry no owner, tenant, group or classification | `_index_document`, `MetadataEnricher` | There is nothing to filter on. |
| F4 | `doc_id = sha256(text)` and point id = `uuid5(doc_id, chunk_index)` | `metadata_enricher.py`, `gateway.py` | Two tenants uploading identical content produce the **same point ids**. The second upsert overwrites the first one's payload, including any ACL. This must be fixed *before* ACLs are stored in payload. |
| F5 | `HybridRetriever(vector_db=...)` is built **without a corpus**, so BM25 is never active in the gateway | `gateway.py:104` | Retrieval is vector-only today. Good for this work (one path to secure) but `ARCHITECTURE_DEEP_DIVE.md` describes BM25 as live. Doc correction is part of this plan. BM25 has no filter parameter at all, so it must be secured before anyone wires it up. |
| F6 | `/ingest/url` and `/ingest/api` fetch arbitrary caller-supplied URLs | extractors | Server-side request forgery (internal metadata endpoints, localhost services). |
| F7 | Full query/context/answer are sent to LangSmith when tracing is on | `observability/tracing.py` | Restricted content leaves the trust boundary. |
| F8 | Web search and grading send content to third-party APIs | `web_search.py`, graders | No per-classification egress control. Logged here; **out of scope** except the rules in P4. |

## 2. Design

### 2.1 Principal (who is asking)

A signed JWT (OIDC access token) validated against the IdP's JWKS. Claims map to:

```python
@dataclass(frozen=True)
class Principal:
    user_id: str
    tenant_id: str            # business line / client; hard isolation boundary
    groups: frozenset[str]    # entitlement groups (e.g. "desk-fx", "research-eu")
    clearance: int            # 0 public · 1 internal · 2 confidential · 3 restricted
    scopes: frozenset[str]    # e.g. "rag:query", "rag:ingest", "rag:web", "rag:admin"
```

- New FastAPI dependency `current_principal` on **every** route except `/health`.
- `AUTH_MODE=jwt` (default) or `AUTH_MODE=disabled`. `disabled` is honoured only when `APP_ENV=dev`; any other environment with `disabled` **refuses to start** (fail closed).
- Missing/invalid/expired token → 401. Valid token lacking the scope → 403.

### 2.2 Document ACL (what a chunk requires)

Stamped into every chunk's Qdrant payload at ingest:

| Field | Type | Meaning |
|---|---|---|
| `tenant_id` | keyword | Owning tenant. Always from the token, never from the request. |
| `owner_id` | keyword | Uploader. |
| `acl_groups` | keyword[] | Groups allowed to read. |
| `acl_public` | bool | Readable by everyone **in the tenant** (still subject to clearance). |
| `classification` | int | Minimum clearance needed. |

Rule — a principal may read a chunk iff **all** hold:
1. `chunk.tenant_id == principal.tenant_id`
2. `chunk.classification <= principal.clearance`
3. `chunk.owner_id == principal.user_id` **or** `chunk.acl_public` **or** `chunk.acl_groups ∩ principal.groups ≠ ∅`

Chunks with **missing** ACL fields are denied (default deny).

### 2.3 Enforcement — two layers

1. **Pre-filter in Qdrant** (primary; also keeps top-k full of *allowed* chunks rather than filtering a result set down to nothing). Built server-side from the Principal. Verified in Qdrant's in-memory mode that this nested filter returns exactly the expected ids:
   ```python
   Filter(must=[
     FieldCondition("tenant_id", MatchValue(p.tenant_id)),
     FieldCondition("classification", Range(lte=p.clearance)),
     Filter(should=[FieldCondition("owner_id", MatchValue(p.user_id)),
                    FieldCondition("acl_public", MatchValue(True)),
                    FieldCondition("acl_groups", MatchAny(list(p.groups)))]),
   ])
   ```
2. **Post-retrieval guard** `authorize(chunks, principal)` run on every candidate list before rerank/context, regardless of which retriever produced it. It re-applies the same rule in Python, drops violators and logs the count. Purpose: defence in depth against a filter bug, a future retriever (BM25), or a stale cache. Web results are tagged `acl_public=True, tenant_id=principal.tenant_id` and are exempt only because they are public by construction.

The client-supplied `metadata_filter` is **removed from `QueryRequest`**. If callers still need narrowing (e.g. by source), accept a small typed `source_in: list[str]` that is **ANDed** with the ACL filter, never replaces it.

### 2.4 Tenancy model — decision needed (see §6)

Default: **one collection, payload filter, `tenant_id` payload index** (cheapest, what the code supports). The retrieval path gets a single seam `collection_for(principal)` so moving to collection-per-tenant (stronger isolation, what regulated deployments usually want) is a config change, not a rewrite.

## 3. Phases

Each phase is independently shippable and ends green. Order matters: P1 must precede P2.

### P0 — Principal and authentication
- `online_pipeline/auth/principal.py` (dataclass), `auth/jwt.py` (JWKS fetch + cache, `PyJWT`), `auth/deps.py` (`current_principal`, `require_scope`).
- Settings: `auth_mode`, `app_env`, `oidc_issuer`, `oidc_audience`, `oidc_jwks_url`, claim names (`tenant_claim`, `groups_claim`, `clearance_claim`).
- Wire dependency into `/query`, `/ingest*`. Frontend: attach `Authorization: Bearer` (calls are in `FileUpload.jsx`, `QueryPanel.jsx`).
- Dev token helper script `scripts/make_dev_token.py` (HS256, dev only).

### P1 — Stamp ACLs on ingest (and fix id collisions)
- `doc_id = sha256(tenant_id + "\0" + text)`; point id `uuid5(tenant_id/doc_id/chunk_index)` → F4 closed.
- `_index_document(doc, principal, acl)` adds the five payload fields. Upload form accepts optional `acl_groups[]`, `classification`, `acl_public`.
- Validation: a user may only grant groups they belong to, and classification ≤ their own clearance. Otherwise 403.
- Kafka path: `ChunkProducer`/`EmbeddingWorker` carry the ACL fields through unchanged (they already forward `metadata`; add a test that proves it).
- Create Qdrant payload indexes (`tenant_id`, `classification`, `acl_groups`, `owner_id`) in `ensure_collection`.

### P2 — Entitlement-filtered retrieval
- `online_pipeline/auth/acl.py`: `build_qdrant_filter(principal)`, `is_authorized(payload, principal)`.
- `VectorDBClient.search` takes an optional `principal`-derived filter; **no caller-built filter reaches Qdrant**.
- Gateway: build filter from `Principal`, call `authorize()` guard after retrieve and after any web top-up, remove `metadata_filter` from the request model.
- `BM25Retriever` / `HybridRetriever`: accept the same `is_authorized` predicate and apply it *before* ranking, so wiring BM25 later cannot leak.
- Fix `ARCHITECTURE_DEEP_DIVE.md`: state that BM25 is not wired in the gateway today.

### P3 — Legacy data and ACL management
- `scripts/backfill_acl.py`: one-off `set_payload` over existing points → `tenant_id=<chosen>`, `owner_id=<admin>`, `classification=1`, `acl_public=false`. Dry-run by default, prints counts.
- `PATCH /documents/{doc_id}/acl` (owner or `rag:admin`) using `set_payload` keyed by `doc_id` filter, so **revocation takes effect on the next query** with no re-embedding.
- `DELETE /documents/{doc_id}` (same authorisation).

### P4 — Hardening that entitlements depend on
- SSRF guard for `/ingest/url` and `/ingest/api`: resolve host, reject private/loopback/link-local ranges, cap redirects and size, require `rag:ingest`.
- Web search requires scope `rag:web`; never send chat history; searches only the rewritten query.
- LangSmith: `langsmith_hide_content=true` by default (hide inputs/outputs; keep stage names, latencies, ids).
- Audit event per query (structured log): `user_id, tenant_id, query_sha256, returned_doc_ids, filtered_out_count, web_used, grounded`. (A tamper-evident audit *store* is a separate plan.)
- Qdrant API key required when `APP_ENV != dev`.

## 4. Acceptance criteria

```gherkin
Feature: Entitlement-filtered retrieval

  Scenario: Tenant isolation
    Given tenant A and tenant B each ingested a document
    When a user of tenant A queries on content only in B's document
    Then B's document is neither retrieved nor cited nor in the answer

  Scenario: Identical uploads do not collide
    Given tenants A and B upload byte-identical files with different ACLs
    Then both documents exist with distinct point ids and their own ACLs

  Scenario: Group entitlement
    Given a document readable by group "desk-fx"
    Then a member of "desk-fx" retrieves it and a non-member of the same tenant does not

  Scenario: Clearance
    Given a document with classification 3
    Then a principal with clearance 2 never sees it

  Scenario: Client cannot widen access
    Given a request that includes a "metadata_filter" or other filter fields
    Then the ACL filter is still applied and unknown fields are rejected or ignored

  Scenario: Default deny
    Given a legacy chunk with no ACL fields
    Then no principal retrieves it

  Scenario: Guard catches a filter bug
    Given the vector filter is bypassed in a test
    Then the post-retrieval guard still removes unauthorised chunks and logs the count

  Scenario: Revocation
    Given a user can read a document
    When its ACL is changed to exclude them
    Then their very next query no longer returns it

  Scenario: Auth fails closed
    Given no token, an expired token, or a wrong audience
    Then /query and /ingest return 401, and a missing scope returns 403
    And starting with AUTH_MODE=disabled outside APP_ENV=dev aborts startup

  Scenario: Existing behaviour preserved
    Given a valid principal with access
    Then multi-file upload, all three chunking strategies, web fallback and generation grading behave as before
```

## 5. Test strategy

- **Isolation matrix** (the important one): real `QdrantClient(":memory:")`, a fixture corpus spanning tenants × groups × classifications × public/owner, and a table of principals with their expected visible-id sets — asserted for both the Qdrant filter *and* the Python `is_authorized` guard, which must agree on every row (a property test over random ACLs/principals).
- Gateway tests with stubbed stages (same pattern as `TestGradedQueryFlow`) proving the filter reaches retrieval and the guard runs after web top-up.
- JWT tests: valid / expired / bad signature / wrong audience / missing claims / `alg=none`.
- All 60 existing tests stay green; existing tests that call `/query` or `_index_document` get a principal fixture.

## 6. Decisions needed from you before P0

1. **Identity provider**: Okta, Azure AD/Entra, Keycloak, Auth0, or none yet (dev tokens only)? Determines claim names and JWKS URL.
2. **Tenancy**: payload filter in one collection (default, fastest), or collection-per-tenant now?
3. **Classification levels**: are 0–3 (public/internal/confidential/restricted) right, or do you have a scheme?
4. **Scope**: is `rag:web` gating acceptable (web search off by default for most users)?
5. **Legacy data**: who owns the documents already indexed (backfill target tenant/owner)?

## 7. Risks and non-goals

- Migration downtime: P1 changes ids, so re-ingestion is needed for existing docs not covered by P3 backfill (backfill keeps old ids; only *new* ingests use tenant-scoped ids).
- Qdrant payload filtering is only as strong as the service's own network/API-key protection (P4).
- Non-goals: row-level encryption, KMS/BYOK, tamper-evident audit store, DLP/PII redaction, egress allow-lists per classification (F8), evaluation harness. Each deserves its own plan.
- LLM providers still see whatever context a principal is *allowed* to read; entitlement filtering does not make third-party model use acceptable for restricted data.
