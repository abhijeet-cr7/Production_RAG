# Production RAG System

A retrieval-augmented generation system for **financial documents** (filings, earnings releases, XBRL), built for several tenants that must never see each other's data, where every answer is checked before it is shown.

- **Ingest** PDF, DOCX, TXT, HTML and XBRL files, or fetch a URL / JSON API, through a hardened pipeline (SSRF-proof fetching, sandboxed parsing, hidden-text and injection screening).
- **Keep tables whole**: ruled and borderless PDF tables, HTML and DOCX tables, and inline-XBRL facts are chunked without splitting rows and stored with exact numbers.
- **Retrieve only what the caller may read**: per-chunk access rules enforced inside the vector store and re-checked in code, with per-tenant keyword search.
- **Answer with checks**: citations must exist, every figure must match a cited source (value, period, currency, direction), output is sanitised, and each tenant chooses warn or abstain.
- **Audit everything** in a tamper-evident log.

For how it all works and why, read **[ARCHITECTURE_DEEP_DIVE.md](ARCHITECTURE_DEEP_DIVE.md)**.

---

## Start it with Docker (recommended)

You need Docker Desktop (or Docker Engine with Compose v2) running.

```bash
cp .env.example .env            # then add at least one LLM key, e.g. GROQ_API_KEY=...
./scripts/docker-up.sh --dev-auth
```

That one command:

1. **checks that the host ports are free** (see below),
2. builds the images and starts Qdrant, Redis, the API and the UI,
3. waits until the API reports healthy (the first start loads the models; allow a few minutes the first time because the image downloads PyTorch),
4. prints the URLs.

`--dev-auth` is for **local use without Okta**: it creates `.env.dev-auth` (a random signing secret, dev tokens, and a permissive `dev` tenant policy). It is refused outside `APP_ENV=dev`.

Then mint a token and open the UI:

```bash
docker compose exec api python scripts/make_dev_token.py --tenant dev \
  --groups finance --clearance 3 --scopes rag:query,rag:ingest,rag:web,rag:admin
```

Open the UI URL the script printed (default <http://localhost:8080>), paste the token into the **token box** at the top, then upload documents and ask questions.

| What | Where (defaults) |
|---|---|
| UI | http://localhost:8080 |
| API + interactive docs | http://localhost:8000/docs |
| Qdrant dashboard | http://localhost:6333/dashboard |

Useful commands:

```bash
docker compose logs -f api          # follow the API log
docker compose down                 # stop (keeps your data)
docker compose down -v              # stop and delete all data (vectors, audit log, model cache)
./scripts/docker-up.sh --no-build   # restart without rebuilding
./scripts/docker-up.sh --kafka      # also start Kafka + ZooKeeper (asynchronous ingestion)
```

### Port checking

Before starting any container, `scripts/docker-up.sh` checks every host port the stack wants:

| Service | Variable | Default |
|---|---|---|
| API | `API_PORT` | 8000 |
| UI | `FRONTEND_PORT` | 8080 |
| Qdrant | `QDRANT_PORT` / `QDRANT_GRPC_PORT` | 6333 / 6334 |
| Redis | `REDIS_PORT` | 6379 |
| Kafka / ZooKeeper (with `--kafka`) | `KAFKA_PORT` / `ZOOKEEPER_PORT` | 9092 / 2181 |

For each one it tells you whether it is free, and if not, **who is holding it** (a Docker container such as `job-pipeline-worker-1`, or a local process with its PID). Ports used by this stack's own running containers are not conflicts, so re-running is safe.

```text
  SERVICE      VARIABLE           PORT     STATUS
  api          API_PORT           8001     busy (docker container 'job-pipeline-worker-1') -> using 8001 instead
  frontend     FRONTEND_PORT      8080     free
```

- default: a busy port is replaced by the next free one, and the final URLs are printed;
- `--strict`: stop with an error instead (free the port, or set e.g. `API_PORT=8001` in `.env`);
- `--check-only`: report and exit without starting anything.

To pick ports yourself, put them in `.env` (`API_PORT=9000`, `FRONTEND_PORT=9001`, ...).

### Production authentication (Okta)

In `.env`:

```bash
APP_ENV=prod
AUTH_MODE=okta
OKTA_ISSUER=https://YOUR-ORG.okta.com/oauth2/default
OKTA_AUDIENCE=api://default
AUDIT_HMAC_KEY=<long random secret>
QDRANT_API_KEY=<long random secret>      # also set on the Qdrant service
```

On the Okta authorization server define the scopes `rag:query rag:ingest rag:web rag:admin rag:audit` and add three custom claims to the access token: `tenant_id` (a single string), `groups` (list) and `clearance` (integer 0 to 3). A token without a single `tenant_id` is rejected. The API **refuses to start** on an unsafe setup (dev auth outside dev, a non-https issuer, no audit or Qdrant key in production).

Per-tenant rules (web search, URL ingest and its allow-list, highest classification, verified-sources-only, warn/abstain, disclaimer) live in `config/tenant_policies.yaml`; edit it and restart `api`. Unknown tenants get the most restrictive policy.

---

## Try it with curl

```bash
TOKEN=$(docker compose exec -T api python scripts/make_dev_token.py --tenant dev --groups finance \
  --clearance 3 --scopes rag:query,rag:ingest,rag:admin)
API=http://localhost:8000          # use the API port the script printed

# upload one or more documents (private to you unless you share them)
curl -H "Authorization: Bearer $TOKEN" -F "files=@10-K.pdf" -F "acl_groups=finance" \
     -F "as_of=2024-09-28" $API/ingest

# ask a question
curl -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
     -d '{"query": "What were net sales in FY2024?"}' $API/query

curl -H "Authorization: Bearer $TOKEN" $API/documents                      # your documents and their status
curl -H "Authorization: Bearer $TOKEN" -X PATCH -H 'Content-Type: application/json' \
     -d '{"acl_groups": []}' $API/documents/<doc_id>/acl                   # revoke sharing
```

A response includes `sources` (provenance only), `verification` (which figures matched which source), `warnings` and `needs_review`. The full endpoint list is in the architecture document.

---

## Run without Docker

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env              # set AUTH_MODE=dev_token, DEV_JWT_SECRET (32+ chars), an LLM key
docker compose up -d qdrant redis # the two services the API needs
uvicorn online_pipeline.api.gateway:app --reload          # API on :8000
cd frontend && npm ci && npm run dev                       # UI on :3000, proxied to the API
```

Tesseract (`brew install tesseract`) is needed only for OCR of scanned PDFs.

## Tests

```bash
pytest tests/ -q          # about 930 tests, roughly 80 seconds
```

Many of them build hostile files on the fly (PDFs with JavaScript or decompression bombs, DOCX with macros or XXE, SSRF targets) and run them through the real sandbox.

## Maintenance scripts

```bash
python scripts/verify_audit_chain.py                       # detects edited, removed or reordered audit events
python scripts/backfill_acl.py --tenant legacy --owner you@acme.com [--apply]    # give old, pre-ACL data an owner
python scripts/purge_expired.py --days 2555 [--apply]      # retention; skips legal holds; dry run by default
```

## Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| `./scripts/docker-up.sh` says a port is busy | It tells you who holds it and uses the next free port. Use `--strict` to fail instead, or set the port in `.env`. |
| API container exits straight away | Read `docker compose logs api`. Almost always the auth check: set `AUTH_MODE=dev_token` + `DEV_JWT_SECRET` (or use `--dev-auth`), or a real `OKTA_ISSUER`. |
| `401 Authentication required` in the UI | Paste a token into the token box. |
| `403 Missing required scope` | Mint the token with the scope that route needs (see above). |
| Answers say figures "could not be verified" | Working as designed: the answer's figures did not match its cited sources. Check `verification` in the response. |
| First `docker-up` is slow or a download times out | The image downloads PyTorch (~200 MB); pip retries, and a re-run reuses cached layers. |
| `can't acces os.tempDir ... permission denied` from Docker | Your shell's `TMPDIR` is not writable; `export TMPDIR=/tmp`. |
| Reset everything | `docker compose down -v` |

## Project layout

```
config/                    settings, tenant_policies.yaml (+ .dev.yaml)
offline_pipeline/
  safety/                  SSRF-safe fetcher, sandbox + worker, PDF/DOCX/XML/HTML guards, injection screen
  tables/                  numbers, blocks, table chunker, PDF table detector, XBRL
  extractors/ preprocessors/ chunkers/ kafka/ embedding_workers/
online_pipeline/
  api/gateway.py           routes and the query pipeline
  auth/                    Okta tokens, scopes, tenant policy, ACL rule, rate limit
  audit/                   hash-chained audit log
  provenance/              trust tiers, raw-original store
  answer_safety/           figure verifier, output sanitiser, advice screen
  grading/ llm/ retrieval/ reranker/ context_builder/ cache/ observability/
  lifecycle.py             delete, retention, backfill (shared with the scripts)
vector_db/                 Qdrant client
scripts/                   docker-up.sh, make_dev_token.py, verify_audit_chain.py, backfill_acl.py, purge_expired.py
frontend/                  React UI (+ nginx.conf, Dockerfile)
docs/plans/                the design plan this was built from
tests/
Dockerfile  docker-compose.yml  .env.example
```
