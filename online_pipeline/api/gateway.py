"""FastAPI gateway: entry point for all online RAG queries."""

from __future__ import annotations

import logging
import hashlib
import re
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, NamedTuple

from urllib.parse import urlsplit

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from config.settings import settings
from offline_pipeline.chunkers.router import ChunkingStrategyRouter
from offline_pipeline.extractors.api_extractor import APIExtractor
from offline_pipeline.extractors.api_extractor import API_CONTENT_TYPES
from offline_pipeline.extractors.web_extractor import URL_CONTENT_TYPES, WebExtractor
from offline_pipeline.preprocessors.cleaner import TextCleaner
from offline_pipeline.preprocessors.metadata_enricher import MetadataEnricher
from offline_pipeline.safety.errors import UnsafeContent
from offline_pipeline.safety.injection import (
    MEDIUM,
    Finding,
    scan_for_injection,
    should_quarantine,
    unicode_findings,
)
from offline_pipeline.safety.net import SafeFetcher
from offline_pipeline.safety.sandbox import safe_extract_file
from offline_pipeline.safety.upload_guard import ALLOWED_EXTENSIONS
from offline_pipeline.tables.blocks import HEADING, PARAGRAPH, blocks_from_json
from online_pipeline.audit.logger import AuditError, get_audit, record
from online_pipeline.auth.acl import ACLError, DocumentACL, authorize, build_document_acl
from online_pipeline.auth.deps import require_scope
from online_pipeline.auth.policy import (
    POLICY_DENIALS,
    RESTRICTIVE,
    TenantPolicy,
    TenantPolicyStore,
    url_host_allowed,
    web_search_decision,
)
from online_pipeline.auth.principal import SCOPE_ADMIN, SCOPE_INGEST, SCOPE_QUERY, Principal
from online_pipeline.auth.ratelimit import RateLimiter
from online_pipeline.auth.tokens import validate_auth_config
from online_pipeline.answer_safety.advice import REFUSAL as ADVICE_REFUSAL
from online_pipeline.answer_safety.advice import is_investment_advice_request
from online_pipeline.answer_safety.sanitize import sanitize_answer
from online_pipeline.answer_safety.verify import Verification, verify_answer
from online_pipeline.cache.embedding_cache import EmbeddingCache, embed_text, embed_texts
from online_pipeline.provenance.rawstore import RawStore
from online_pipeline.provenance.trust import (
    Provenance,
    ProvenanceError,
    check_expected_hash,
    host_of,
    normalise_expected_hash,
    for_remote,
    for_upload,
    parse_as_of,
    public_source,
    trust_allows,
    web_result_provenance,
)
from online_pipeline.context_builder.builder import ContextBuilder
from online_pipeline.grading.grader import GradeResult
from online_pipeline.llm.router import LLMRouter
from online_pipeline.observability.tracing import configure_tracing, traced
from online_pipeline.reranker.reranker import Reranker
from online_pipeline.retrieval.hybrid_retriever import HybridRetriever
from online_pipeline.retrieval.web_search import TavilyWebSearch
from vector_db.client import VectorDBClient

logger = logging.getLogger(__name__)


# ── Request / Response models ────────────────────────────────────────────────

_ALLOWED_EXTENSIONS = ALLOWED_EXTENSIONS


class QueryRequest(BaseModel):
    # Unknown fields are rejected, so a caller cannot smuggle in a filter of
    # their own: the entitlement filter is built server-side from the token.
    model_config = ConfigDict(extra="forbid")

    query: str
    # Optional narrowing to named sources; ANDed with the ACL filter, never a
    # replacement for it.
    source_in: list[str] | None = None
    top_k: int = settings.top_k_rerank
    chat_history: list[dict] = Field(default_factory=list)
    # None = follow settings.web_search_enabled; True/False force the behaviour.
    web_search: bool | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    web_search_used: bool = False
    # None when grading is disabled or every grader provider failed.
    grounded: bool | None = None
    answer_useful: bool | None = None
    generation_attempts: int = 1
    # Provenance summary of the sources actually used, e.g. {"verified_primary": 2}.
    trust_summary: dict[str, int] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    # Deterministic check of citations and figures against the cited sources.
    verification: dict | None = None
    # True when a person should look before this is relied on: restricted sources
    # were used, a figure failed verification, or the answer was not grounded.
    needs_review: bool = False


class IngestFileResult(BaseModel):
    filename: str
    # "indexed" | "quarantined" (stored but hidden pending review) |
    # "rejected" (failed a security check) | "error"
    status: str
    chunks_indexed: int = 0
    doc_id: str = ""
    detail: str = ""


class IngestResponse(BaseModel):
    files: list[IngestFileResult]
    total_chunks_indexed: int


class IngestUrlRequest(BaseModel):
    url: str
    acl_groups: list[str] = Field(default_factory=list)
    acl_public: bool = False
    classification: int | None = None
    # Provenance the caller may declare. Trust tier, publisher and origin are
    # always computed server-side and cannot be supplied.
    as_of: str | None = None
    expected_sha256: str | None = None
    supersedes: str | None = None


class IngestApiRequest(BaseModel):
    url: str
    params: dict | None = None
    text_fields: list[str] | None = None
    acl_groups: list[str] = Field(default_factory=list)
    acl_public: bool = False
    classification: int | None = None
    as_of: str | None = None
    expected_sha256: str | None = None
    supersedes: str | None = None


class IngestSourceResponse(BaseModel):
    source: str
    chunks_indexed: int
    doc_id: str
    status: str = "indexed"  # "indexed" | "quarantined"
    screen_flags: list[str] = Field(default_factory=list)


# ── Dependency singletons (initialised at startup) ───────────────────────────

_llm_router: LLMRouter | None = None
_cache: EmbeddingCache | None = None
_retriever: HybridRetriever | None = None
_reranker: Reranker | None = None
_context_builder: ContextBuilder | None = None
_vector_db: VectorDBClient | None = None
_web_search: TavilyWebSearch | None = None
_policies: TenantPolicyStore | None = None
_raw_store: RawStore | None = None


def _verified_publishers() -> list[str]:
    return [p.strip() for p in settings.verified_publishers.split(",") if p.strip()]


def _policy_for(principal: Principal) -> TenantPolicy:
    return RESTRICTIVE if _policies is None else _policies.for_tenant(principal.tenant_id)


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    global _llm_router, _cache, _retriever, _reranker, _context_builder, _vector_db, _web_search
    global _policies
    validate_auth_config(settings)  # refuse to start on an unsafe auth setup
    global _raw_store
    _policies = TenantPolicyStore.load(settings.tenant_policies_path)
    _raw_store = RawStore(settings.raw_store_dir) if settings.raw_store_enabled else None
    # Fail closed at boot if the audit log cannot be written.
    record("service.start", actor="system", auth_mode=settings.auth_mode, app_env=settings.app_env)
    configure_tracing()
    _vector_db = VectorDBClient()
    _vector_db.ensure_collection()
    _llm_router = LLMRouter()
    _cache = EmbeddingCache(redis_url=settings.redis_url)
    _retriever = HybridRetriever(
        vector_db=_vector_db,
        bm25_weight=settings.bm25_weight,
        vector_weight=settings.vector_weight,
    )
    _refresh_lexical_index()
    _reranker = Reranker()
    _context_builder = ContextBuilder()
    _web_search = TavilyWebSearch()
    yield


app = FastAPI(title="Production RAG API", version="0.1.0", lifespan=lifespan)


@app.exception_handler(AuditError)
async def _audit_unavailable(_: Request, exc: AuditError) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={"detail": "Request could not be completed: audit log unavailable."},
    )


# ── Routes ───────────────────────────────────────────────────────────────────


def _refresh_lexical_index() -> None:
    """Rebuild the BM25 index from the chunks currently in the vector store.

    The retriever is constructed without a corpus, so without this call the
    lexical branch never runs and "hybrid" retrieval is dense-only.
    """
    if not settings.bm25_enabled or _retriever is None or _vector_db is None:
        return
    try:
        corpus = _vector_db.scroll_all(limit=settings.bm25_max_corpus_chunks)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read corpus for BM25 (%s); retrieval is dense-only.", exc)
        return
    _retriever.load_corpus(corpus)


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "lexical_retrieval": bool(_retriever and _retriever.lexical_enabled),
    }


class IndexResult(NamedTuple):
    doc_id: str
    chunks: int
    quarantined: bool = False
    rules: tuple[str, ...] = ()


@traced(name="index_document", run_type="chain")
def _index_document(
    doc: dict, principal: Principal, acl: DocumentACL, provenance: Provenance
) -> IndexResult:
    """Screen, clean, enrich, chunk, embed and upsert *doc*.

    Every chunk is stamped with *acl*; tenant and owner come from the token, so
    nothing in the request can change who owns a document. Content that trips the
    injection screen is stored but marked ``quarantined``: invisible to retrieval
    until an administrator releases it.
    """
    metadata = doc.setdefault("metadata", {})
    findings: list[Finding] = unicode_findings(doc["text"])  # needs the raw text
    cleaner = TextCleaner()
    doc["text"] = cleaner.clean(doc["text"])
    # Structure-aware extractors return blocks (prose, tables, XBRL facts). Table
    # cells are cleaned cell by cell at extraction; prose is cleaned here.
    blocks = blocks_from_json(doc["blocks"]) if doc.get("blocks") else None
    for block in blocks or []:
        if block.kind in (HEADING, PARAGRAPH):
            block.text = cleaner.clean(block.text)
    findings += scan_for_injection(doc["text"] + "\n" + str(metadata.get("title", "")))
    if any(metadata.get(k) for k in ("hidden_chars_removed", "hidden_elements_removed", "hidden_runs_removed")):
        findings.append(Finding("hidden_content_removed", MEDIUM))
    quarantined = should_quarantine(findings)
    rules = tuple(sorted({f.rule for f in findings}))

    doc = MetadataEnricher().enrich(
        doc, tenant_id=principal.tenant_id, owner_id=principal.user_id
    )

    chunker = ChunkingStrategyRouter(
        strategy=settings.chunking_strategy,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = chunker.chunk(doc["text"], doc["metadata"], blocks=blocks)

    if not chunks:
        raise HTTPException(status_code=422, detail="Document produced no text chunks.")

    # One batched embed call for the whole document, and one upsert: embedding
    # per chunk dominated ingest time, and a round trip per chunk followed it.
    stamp = {
        **acl.as_payload(),
        **provenance.as_payload(),
        "quarantined": quarantined,
        "screen_flags": list(rules),
    }
    for chunk in chunks:
        # Authoritative stamp: overrides anything an extractor or chunker set.
        chunk["metadata"].update(stamp)

    vectors = embed_texts([chunk["text"] for chunk in chunks])

    points = []
    for chunk, vector in zip(chunks, vectors):
        # Qdrant requires IDs to be unsigned ints or UUIDs.
        # Derive a stable UUID5 from the doc_id + chunk_index.
        point_id = str(uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"{doc['metadata']['doc_id']}_{chunk['metadata']['chunk_index']}",
        ))
        points.append({
            "id": point_id,
            "vector": vector,
            "payload": chunk["metadata"],
            "text": chunk["text"],
        })
    _vector_db.upsert(points)

    return IndexResult(doc["metadata"]["doc_id"], len(chunks), quarantined, rules)


def _resolve_acl(
    principal: Principal,
    *,
    acl_groups: list[str] | None,
    acl_public: bool,
    classification: int | None,
) -> DocumentACL:
    """Validate the requested ACL up front so a bad one rejects the whole request."""
    try:
        return build_document_acl(
            principal,
            _policy_for(principal),
            acl_groups=acl_groups,
            acl_public=acl_public,
            classification=classification,
        )
    except ACLError as exc:
        record(
            "policy.deny",
            actor=principal,
            decision="deny",
            reason="acl_not_permitted",
            resource="document_acl",
            code=exc.code,
        )
        raise HTTPException(status_code=403, detail=str(exc)) from exc


def _audit_ingest(
    principal: Principal,
    source_type: str,
    origin: str,
    status: str,
    *,
    doc_id: str | None = None,
    chunks: int = 0,
    reason: str | None = None,
    flags: tuple[str, ...] = (),
    provenance: Provenance | None = None,
) -> None:
    """One audit event per ingested source; origin is fingerprinted, not stored."""
    decision = {"indexed": "allow", "quarantined": "quarantine", "rejected": "reject"}.get(status, "error")
    record(
        "ingest",
        actor=principal,
        resource={"doc_id": doc_id} if doc_id else None,
        decision=decision,
        reason=reason,
        source_type=source_type,
        origin_fp=get_audit().fingerprint(origin),
        status=status,
        chunks=chunks,
        screen_flags=list(flags),
        content_sha256=provenance.content_sha256 if provenance else None,
        trust_tier=provenance.trust_tier if provenance else None,
    )


# Rejection codes that mean "the request itself was a policy violation" (403),
# as opposed to the remote content simply being unusable (422).
_POLICY_REJECTIONS = frozenset({"ssrf_blocked", "host_not_allowed", "dns_failure"})


def _rejection_status(code: str) -> int:
    return 403 if code in _POLICY_REJECTIONS or code.startswith("url_") else 422


def _require_url_ingest(principal: Principal, url: str, source_type: str) -> None:
    """Enforce the tenant URL policy before any outbound fetch happens."""
    if url_host_allowed(_policy_for(principal), url):
        return
    try:
        host = urlsplit(url.strip()).hostname or ""
    except ValueError:
        host = ""
    record(
        "policy.deny",
        actor=principal,
        decision="deny",
        reason="url_not_allowed",
        resource={"host": host},
        source_type=source_type,
    )
    raise HTTPException(
        status_code=403, detail="URL ingestion is not permitted for this tenant or host."
    )


_fetch_limiter = RateLimiter(settings.ingest_rate_limit_per_minute)


def _enforce_fetch_rate_limit(principal: Principal, source_type: str) -> None:
    if _fetch_limiter.allow(f"{principal.tenant_id}:{principal.user_id}"):
        return
    record(
        "policy.deny",
        actor=principal,
        decision="deny",
        reason="rate_limited",
        resource="outbound_fetch",
        source_type=source_type,
    )
    raise HTTPException(
        status_code=429, detail="Too many ingest requests; slow down.", headers={"Retry-After": "60"}
    )


def _fetcher_for(principal: Principal, content_types: dict[str, str]) -> SafeFetcher:
    """A fetcher bound to the caller's tenant allow-list and the configured caps."""
    allow_http = settings.ingest_allow_http and settings.app_env == "dev"
    return SafeFetcher(
        allowlist=_policy_for(principal).url_allowlist,
        max_bytes=settings.ingest_max_fetch_bytes,
        max_redirects=settings.ingest_max_redirects,
        total_timeout=settings.ingest_fetch_timeout_seconds,
        allow_http=allow_http,
        allowed_ports=(80, 443) if allow_http else (443,),
        allowed_content_types=content_types,
    )


def _parser_limits() -> dict:
    return {
        "timeout": settings.parser_timeout_seconds,
        "cpu_seconds": settings.parser_cpu_seconds,
        "max_memory_mb": settings.parser_max_memory_mb,
        "max_pages": settings.pdf_max_pages,
        "max_chars": settings.extracted_text_max_chars,
        "ocr_max_pages": settings.pdf_ocr_max_pages,
    }


def _extract_upload(path: str, suffix: str) -> dict:
    """Parse an uploaded file in the resource-limited sandbox subprocess."""
    return safe_extract_file(path, suffix, **_parser_limits())


def _display_name(filename: str) -> str:
    """Basename without control characters: what we store and show as the source.

    Over-long names are shortened in the stem so the extension survives; the
    extension is what decides whether the file is accepted at all.
    """
    name = re.sub(r"[\x00-\x1f\x7f]", "", Path(filename.replace("\\", "/")).name)
    if len(name) <= 200:
        return name
    suffix = Path(name).suffix[:16]
    return name[: 200 - len(suffix)] + suffix


async def _save_upload(file: UploadFile, suffix: str) -> tuple[str, str]:
    """Stream *file* to a temp path, refusing anything over the size cap.

    Returns (path, sha256 of the exact bytes received).
    """
    total = 0
    digest = hashlib.sha256()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        path = tmp.name
        try:
            while chunk := await file.read(1024 * 1024):
                total += len(chunk)
                if total > settings.ingest_max_upload_bytes:
                    raise UnsafeContent("file_too_large", "File exceeds the upload size limit.")
                digest.update(chunk)
                tmp.write(chunk)
        except BaseException:
            tmp.close()
            Path(path).unlink(missing_ok=True)
            raise
    return path, digest.hexdigest()


async def _resolve_provenance_inputs(
    principal: Principal,
    *,
    as_of: str | None,
    expected_sha256: str | None,
    supersedes: str | None,
) -> tuple[str | None, str | None, str | None]:
    """Validate what the caller may declare about a document's provenance."""
    try:
        as_of_value = parse_as_of(as_of)
        expected = normalise_expected_hash(expected_sha256)
    except ProvenanceError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    target = (supersedes or "").strip() or None
    if target:
        info = await run_in_threadpool(_vector_db.get_document_info, principal.tenant_id, target)
        if info is None:
            raise HTTPException(status_code=404, detail="supersedes: no such document.")
        if info.get("owner_id") != principal.user_id and not principal.has_scope(SCOPE_ADMIN):
            record(
                "policy.deny", actor=principal, decision="deny",
                reason="supersedes_not_permitted", resource={"doc_id": target},
            )
            raise HTTPException(status_code=403, detail="You can only supersede your own documents.")
    return as_of_value, expected, target


async def _finish_supersession(
    principal: Principal, old_doc_id: str | None, indexed: "IndexResult"
) -> None:
    """Hide the replaced version once its successor is live (never for quarantined ones)."""
    if not old_doc_id or indexed.quarantined or indexed.doc_id == old_doc_id:
        return
    changed = await run_in_threadpool(
        _vector_db.set_document_payload, principal.tenant_id, old_doc_id,
        {"superseded": True, "superseded_by": indexed.doc_id},
    )
    record(
        "doc.superseded", actor=principal, resource={"doc_id": old_doc_id},
        superseded_by=indexed.doc_id, chunks=changed,
    )


def _store_raw_file(tenant_id: str, path: str, sha256: str) -> None:
    if _raw_store is not None:
        _raw_store.put_file(tenant_id, path, sha256)


def _store_raw_bytes(tenant_id: str, data: bytes, sha256: str) -> None:
    if _raw_store is not None:
        _raw_store.put_bytes(tenant_id, data, sha256)


@app.post("/ingest", response_model=IngestResponse)
async def ingest(
    files: list[UploadFile] = File(...),
    acl_groups: list[str] = Form(default=[]),
    acl_public: bool = Form(default=False),
    classification: int | None = Form(default=None),
    as_of: str | None = Form(default=None),
    expected_sha256: str | None = Form(default=None),
    supersedes: str | None = Form(default=None),
    principal: Principal = Depends(require_scope(SCOPE_INGEST)),
) -> IngestResponse:
    """Upload and index one or more documents (PDF, DOCX, or TXT).

    Each file is size-capped, type-checked, parsed in a sandboxed subprocess,
    screened, and only then runs through the offline pipeline:
    clean → enrich → chunk → embed → upsert.
    Files are processed independently; a failure in one does not abort the rest.
    """
    acl = _resolve_acl(
        principal, acl_groups=acl_groups, acl_public=acl_public, classification=classification
    )
    as_of_value, expected, supersedes_id = await _resolve_provenance_inputs(
        principal, as_of=as_of, expected_sha256=expected_sha256, supersedes=supersedes
    )
    results: list[IngestFileResult] = []

    for file in files:
        filename = _display_name(file.filename or "")
        suffix = Path(filename).suffix.lower()
        if suffix not in _ALLOWED_EXTENSIONS:
            results.append(IngestFileResult(
                filename=filename,
                status="error",
                detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(_ALLOWED_EXTENSIONS)}",
            ))
            _audit_ingest(principal, "file", filename, "error", reason="unsupported_type")
            continue

        tmp_path: str | None = None
        provenance: Provenance | None = None
        try:
            tmp_path, sha256 = await _save_upload(file, suffix)
            if not check_expected_hash(expected, sha256):
                raise UnsafeContent("hash_mismatch", "File does not match the expected SHA-256.")
            doc = await run_in_threadpool(_extract_upload, tmp_path, suffix)
            # The extractor knows nothing about the user's file name, and must
            # never leave a server temp path as the "source" shown to users.
            doc["metadata"]["source"] = filename
            doc["metadata"]["file_type"] = suffix.lstrip(".")

            provenance = for_upload(
                filename=filename, content_sha256=sha256, user_id=principal.user_id,
                as_of=as_of_value, integrity_verified=expected is not None, supersedes=supersedes_id,
            )
            await run_in_threadpool(_store_raw_file, principal.tenant_id, tmp_path, sha256)
            indexed = await run_in_threadpool(_index_document, doc, principal, acl, provenance)
            await _finish_supersession(principal, supersedes_id, indexed)
            status = "quarantined" if indexed.quarantined else "indexed"
            results.append(IngestFileResult(
                filename=filename,
                status=status,
                chunks_indexed=indexed.chunks,
                doc_id=indexed.doc_id,
                detail=(f"Held for review (matched: {', '.join(indexed.rules)})"
                        if indexed.quarantined else ""),
            ))
            _audit_ingest(principal, "file", filename, status, doc_id=indexed.doc_id,
                          chunks=indexed.chunks, flags=indexed.rules, provenance=provenance)
        except UnsafeContent as exc:
            results.append(IngestFileResult(
                filename=filename, status="rejected",
                detail=f"Rejected by security checks ({exc.code}): {exc.detail}",
            ))
            _audit_ingest(principal, "file", filename, "rejected", reason=exc.code)
        except HTTPException as exc:
            results.append(IngestFileResult(filename=filename, status="error", detail=str(exc.detail)))
            _audit_ingest(principal, "file", filename, "error", reason=f"http_{exc.status_code}")
        except Exception as exc:  # noqa: BLE001
            results.append(IngestFileResult(filename=filename, status="error", detail=str(exc)))
            _audit_ingest(principal, "file", filename, "error", reason=type(exc).__name__)
        finally:
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)

    if all(r.status in {"error", "rejected"} for r in results):
        raise HTTPException(status_code=400, detail="; ".join(f"{r.filename}: {r.detail}" for r in results))

    # New chunks are invisible to the lexical branch until the index is rebuilt.
    await run_in_threadpool(_refresh_lexical_index)

    return IngestResponse(
        files=results,
        total_chunks_indexed=sum(r.chunks_indexed for r in results),
    )


async def _ingest_remote(
    principal: Principal,
    request: IngestUrlRequest | IngestApiRequest,
    source_type: str,
    fetch: "callable",
) -> IngestSourceResponse:
    """Shared flow for /ingest/url and /ingest/api: policy, ACL, safe fetch, index."""
    if not request.url.strip().lower().startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="URL must start with http:// or https://")
    _require_url_ingest(principal, request.url, source_type)
    _enforce_fetch_rate_limit(principal, source_type)
    acl = _resolve_acl(
        principal,
        acl_groups=request.acl_groups,
        acl_public=request.acl_public,
        classification=request.classification,
    )
    as_of_value, expected, supersedes_id = await _resolve_provenance_inputs(
        principal, as_of=request.as_of, expected_sha256=request.expected_sha256,
        supersedes=request.supersedes,
    )

    try:
        doc = await run_in_threadpool(fetch)
        meta = doc["metadata"]
        if not check_expected_hash(expected, meta["content_sha256"]):
            raise UnsafeContent("hash_mismatch", "Fetched content does not match the expected SHA-256.")
    except UnsafeContent as exc:
        _audit_ingest(principal, source_type, request.url, "rejected", reason=exc.code)
        if exc.code in _POLICY_REJECTIONS or exc.code.startswith("url_"):
            record(
                "policy.deny", actor=principal, decision="deny", reason=exc.code,
                resource="outbound_fetch", source_type=source_type,
            )
        raise HTTPException(
            status_code=_rejection_status(exc.code),
            detail=f"Rejected by security checks ({exc.code}): {exc.detail}",
        ) from exc

    raw = doc.pop("raw", None)
    provenance = for_remote(
        source_type=source_type, final_url=meta["final_url"], content_sha256=meta["content_sha256"],
        fetched_at=meta["fetched_at"], user_id=principal.user_id,
        verified_publishers=_verified_publishers(), as_of=as_of_value,
        integrity_verified=expected is not None, supersedes=supersedes_id,
    )
    if raw is not None:
        await run_in_threadpool(_store_raw_bytes, principal.tenant_id, raw, meta["content_sha256"])
    indexed = await run_in_threadpool(_index_document, doc, principal, acl, provenance)
    await _finish_supersession(principal, supersedes_id, indexed)
    status = "quarantined" if indexed.quarantined else "indexed"
    _audit_ingest(principal, source_type, request.url, status, doc_id=indexed.doc_id,
                  chunks=indexed.chunks, flags=indexed.rules, provenance=provenance)
    await run_in_threadpool(_refresh_lexical_index)
    return IngestSourceResponse(
        source=request.url, chunks_indexed=indexed.chunks, doc_id=indexed.doc_id,
        status=status, screen_flags=list(indexed.rules),
    )


@app.post("/ingest/url", response_model=IngestSourceResponse)
async def ingest_url(
    request: IngestUrlRequest,
    principal: Principal = Depends(require_scope(SCOPE_INGEST)),
) -> IngestSourceResponse:
    """Fetch a web page through the SSRF-safe fetcher, sanitise it and index it."""
    extractor = WebExtractor(_fetcher_for(principal, URL_CONTENT_TYPES), _parser_limits())
    return await _ingest_remote(
        principal, request, "url", lambda: extractor.extract(request.url.strip())
    )


@app.post("/ingest/api", response_model=IngestSourceResponse)
async def ingest_api(
    request: IngestApiRequest,
    principal: Principal = Depends(require_scope(SCOPE_INGEST)),
) -> IngestSourceResponse:
    """Fetch JSON from a REST endpoint through the SSRF-safe fetcher and index it."""
    extractor = APIExtractor(_fetcher_for(principal, API_CONTENT_TYPES))
    return await _ingest_remote(
        principal, request, "api",
        lambda: extractor.extract(
            request.url.strip(), params=request.params, text_fields=request.text_fields
        ),
    )


# ── Quarantine review (tenant administrators) ───────────────────────────────

@app.get("/admin/quarantine")
async def list_quarantine(
    principal: Principal = Depends(require_scope(SCOPE_ADMIN)),
) -> dict:
    """Documents held back by the injection screen, for this tenant only."""
    documents = await run_in_threadpool(_vector_db.list_quarantined, principal.tenant_id)
    record("quarantine.list", actor=principal, count=len(documents))
    return {"documents": documents}


@app.post("/admin/quarantine/{doc_id}/release")
async def release_quarantine(
    doc_id: str,
    principal: Principal = Depends(require_scope(SCOPE_ADMIN)),
) -> dict:
    """Make a reviewed, quarantined document retrievable again."""
    released = await run_in_threadpool(
        _vector_db.release_quarantine, principal.tenant_id, doc_id, principal.user_id
    )
    if not released:
        raise HTTPException(status_code=404, detail="No quarantined document with that id.")
    record(
        "quarantine.release", actor=principal, resource={"doc_id": doc_id}, chunks=released
    )
    await run_in_threadpool(_refresh_lexical_index)
    return {"doc_id": doc_id, "chunks_released": released}


@traced(name="query_rewrite", run_type="llm")
def _stage_rewrite(query: str) -> str:
    return _llm_router.rewrite_query(query)


@traced(name="embed_query", run_type="embedding")
def _stage_embed(text: str) -> list[float]:
    embedding = _cache.get(text)
    if embedding is None:
        embedding = embed_text(text)
        _cache.set(text, embedding)
    return embedding


@traced(name="hybrid_retrieve", run_type="retriever")
def _stage_retrieve(
    query_text: str,
    query_vector: list[float],
    principal: Principal,
    source_in: list[str] | None,
) -> list[dict]:
    return _retriever.retrieve(
        query_text=query_text,
        query_vector=query_vector,
        principal=principal,
        top_k=settings.top_k_retrieval,
        source_in=source_in,
    )


@traced(name="tavily_web_search", run_type="tool")
def _stage_web_search(query: str, include_domains: list[str] | None = None) -> list[dict]:
    return _web_search.search(query, include_domains=include_domains)


@traced(name="rerank", run_type="chain")
def _stage_rerank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    return _reranker.rerank(query, candidates, top_k=top_k)


@traced(name="build_context", run_type="chain")
def _stage_build_context(chunks: list[dict]) -> str:
    return _context_builder.build(chunks)


@traced(name="generate_answer", run_type="llm")
def _stage_generate(
    query: str,
    context: str,
    chat_history: list[dict],
    feedback: str | None = None,
) -> str:
    return _llm_router.generate_answer(
        query=query,
        context=context,
        chat_history=chat_history,
        feedback=feedback,
    )


@traced(name="grade_generation", run_type="llm")
def _stage_grade(query: str, context: str, answer: str) -> GradeResult | None:
    return _llm_router.grade_answer(query, context, answer)


@traced(name="verify_answer", run_type="chain")
def _stage_verify(answer: str, included: list[dict]) -> Verification:
    return verify_answer(answer, included, require_citations=settings.require_citations)


def _included_chunks(chunks: list[dict]) -> list[dict]:
    """The chunks the prompt actually contains, in order: citation [n] is position n."""
    return chunks if _context_builder is None else _context_builder.select(chunks)


def _evaluate(
    query: str, context: str, included: list[dict], answer: str
) -> tuple[GradeResult | None, Verification | None]:
    grade = _stage_grade(query, context, answer) if settings.answer_grading_enabled else None
    verification = _stage_verify(answer, included) if settings.answer_verification_enabled else None
    return grade, verification


def _needs_work(grade: GradeResult | None, verification: Verification | None) -> bool:
    return (grade is not None and not grade.grounded) or (verification is not None and not verification.ok)


def _unacceptable(grade: GradeResult | None, verification: Verification | None) -> bool:
    return (grade is not None and not grade.acceptable) or (verification is not None and not verification.ok)


def _combined_feedback(grade: GradeResult | None, verification: Verification | None) -> str | None:
    parts = []
    if grade is not None and not grade.acceptable:
        parts.append(grade.feedback())
    if verification is not None and not verification.ok:
        parts.append(verification.feedback())
    return " ".join(parts) or None


@app.post("/query", response_model=QueryResponse)
@traced(name="rag_query", run_type="chain")
async def query(
    request: QueryRequest,
    principal: Principal = Depends(require_scope(SCOPE_QUERY)),
) -> QueryResponse:
    """Run the full online RAG pipeline for the given user query."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    request_id = uuid.uuid4().hex
    query_fp = get_audit().fingerprint(request.query)
    try:
        # The pipeline is synchronous and network/CPU bound; run it off the event
        # loop so concurrent queries overlap instead of queueing behind one another.
        response, audit_details = await run_in_threadpool(_answer_query, request, principal)
    except Exception as exc:  # noqa: BLE001
        record(
            "query",
            actor=principal,
            decision="error",
            reason=type(exc).__name__,
            request_id=request_id,
            query_fp=query_fp,
        )
        raise
    record("query", actor=principal, request_id=request_id, query_fp=query_fp, **audit_details)
    return response


def _answer_query(request: QueryRequest, principal: Principal) -> tuple[QueryResponse, dict]:
    policy = _policy_for(principal)

    # Declined before any retrieval or model call: documents can tell you what a
    # filing reports, not what you should buy.
    if policy.refuse_investment_advice and is_investment_advice_request(request.query):
        refusal = QueryResponse(answer=ADVICE_REFUSAL, sources=[], warnings=["refused_investment_advice"])
        return refusal, {"refused": "investment_advice", "n_sources": 0, "returned_doc_ids": []}

    rewritten = _stage_rewrite(request.query)
    embedding = _stage_embed(rewritten)
    guard = _AclGuard(principal, policy)
    candidates = guard.filter(_stage_retrieve(rewritten, embedding, principal, request.source_in))
    reranked = guard.filter(_stage_rerank(rewritten, candidates, request.top_k))

    # Web search must pass the tenant ceiling, the user's scope and the request.
    allowed, deny_reason = web_search_decision(
        policy=policy,
        principal=principal,
        request_flag=request.web_search,
        global_enabled=settings.web_search_enabled,
    )
    allow_web = allowed and _web_search is not None and _web_search.available
    web_denied: str | None = None  # access-control reason a wanted web search was blocked

    # Top up with live web results when the local corpus answers poorly, then
    # rerank everything together so both sources compete on one relevance scale.
    web_search_used = False
    if _local_recall_is_weak(candidates, reranked):
        if allow_web:
            topped_up = _web_top_up(rewritten, candidates, request.top_k, principal, policy)
            if topped_up is not None:
                reranked, web_search_used = guard.filter(topped_up), True
        elif deny_reason in POLICY_DENIALS:
            web_denied = deny_reason

    context = _stage_build_context(reranked)
    included = _included_chunks(reranked)
    answer = _stage_generate(request.query, context, request.chat_history)
    attempts = 1
    grade, verification = _evaluate(request.query, context, included, answer)

    # A draft that is ungrounded or whose figures do not check out is regenerated
    # with specific feedback, bounded by max_generation_attempts so a stubborn
    # model cannot loop forever.
    while _needs_work(grade, verification) and attempts < settings.max_generation_attempts:
        answer = _stage_generate(
            request.query, context, request.chat_history, _combined_feedback(grade, verification)
        )
        attempts += 1
        grade, verification = _evaluate(request.query, context, included, answer)

    # Still unusable and the web has not been tried: the context itself is the
    # problem, so fetch fresh evidence and make one last attempt.
    if _unacceptable(grade, verification) and not web_search_used:
        if allow_web:
            topped_up = _web_top_up(rewritten, candidates, request.top_k, principal, policy)
            if topped_up is not None:
                reranked, web_search_used = guard.filter(topped_up), True
                context = _stage_build_context(reranked)
                included = _included_chunks(reranked)
                answer = _stage_generate(
                    request.query, context, request.chat_history, _combined_feedback(grade, verification)
                )
                attempts += 1
                grade, verification = _evaluate(request.query, context, included, answer)
        elif deny_reason in POLICY_DENIALS:
            web_denied = deny_reason

    ungrounded = grade is not None and not grade.grounded
    figures_failed = verification is not None and not verification.ok
    answer, policies_applied = _apply_answer_policies(answer, policy, ungrounded, verification)

    if guard.dropped:
        # The pre-filter should have made this impossible: treat as an incident.
        record(
            "acl.guard_dropped",
            actor=principal,
            decision="deny",
            reason="prefilter_bypass_suspected",
            resource="retrieval",
            dropped=guard.dropped,
        )

    if web_denied:
        record(
            "policy.deny",
            actor=principal,
            decision="deny",
            reason=web_denied,
            resource="web_search",
        )

    metadata_used = [r.get("metadata", {}) for r in reranked]

    # What leaves the service is plain text: no images, HTML or URLs except the
    # answer's own sources, whatever a poisoned document told the model to emit.
    allowed_hosts = {host_of(str(m.get("origin", ""))) for m in metadata_used} - {""}
    answer, sanitize_flags = sanitize_answer(answer, allowed_hosts)
    if policy.answer_disclaimer:
        answer = f"{answer}\n\n{policy.answer_disclaimer}"

    sources = [public_source(m) for m in metadata_used]  # never expose ACL internals
    trust_summary: dict[str, int] = {}
    for m in metadata_used:
        tier = m.get("trust_tier", "unverified")
        trust_summary[tier] = trust_summary.get(tier, 0) + 1
    warnings: list[str] = []
    if trust_summary.get("unverified"):
        warnings.append("unverified_sources_used")
    if figures_failed:
        warnings.append("figures_unverified")
    if ungrounded:
        warnings.append("ungrounded")
    if sanitize_flags:
        warnings.append("output_sanitized")
    needs_review = bool(
        ungrounded or figures_failed
        or any(int(m.get("classification", 0) or 0) >= 2 for m in metadata_used)
    )
    response = QueryResponse(
        answer=answer,
        sources=sources,
        trust_summary=trust_summary,
        warnings=warnings,
        web_search_used=web_search_used,
        grounded=None if grade is None else grade.grounded,
        answer_useful=None if grade is None else grade.answers_question,
        generation_attempts=attempts,
        verification=None if verification is None else verification.summary(),
        needs_review=needs_review,
    )
    audit_details = {
        "returned_doc_ids": [m["doc_id"] for m in sources if m.get("doc_id")][:50],
        "n_sources": len(sources),
        "guard_dropped": guard.dropped,
        "unverified_dropped": guard.unverified_dropped,
        "trust_summary": trust_summary,
        "web_used": web_search_used,
        "web_denied": web_denied,
        "grounded": response.grounded,
        "answer_useful": response.answer_useful,
        "attempts": attempts,
        "policies_applied": policies_applied,
        "verification": None if verification is None else {
            "ok": verification.ok, "abstained": verification.abstained,
            "counts": verification.summary()["counts"],
            "invalid_citations": verification.invalid_citations,
        },
        "output_sanitized": sanitize_flags,
        "needs_review": needs_review,
    }
    return response, audit_details


def _web_top_up(
    query: str,
    candidates: list[dict],
    top_k: int,
    principal: Principal,
    policy: TenantPolicy,
) -> list[dict] | None:
    """Rerank local candidates together with web hits; None when the web gave nothing."""
    # A tenant that only accepts verified sources may only search verified publishers.
    domains = _verified_publishers() if policy.require_verified_sources else None
    web_results = _stage_web_search(query, domains)
    if not web_results:
        return None
    stamped = _stamp_web_chunks(web_results, principal)
    return _stage_rerank(query, candidates + stamped, top_k)


def _stamp_web_chunks(results: list[dict], principal: Principal) -> list[dict]:
    """Give web hits an explicit ACL and a computed trust tier.

    They are public by construction, but stamping them (rather than exempting
    them) means the guard runs the same rules over every chunk, whatever its
    origin. The tier comes from the host, never from anything the page says.
    """
    for hit in results:
        meta = hit.setdefault("metadata", {})
        url = str(meta.get("source") or meta.get("origin") or "")
        meta.update(
            tenant_id=principal.tenant_id,
            owner_id="web",
            acl_groups=[],
            acl_public=True,
            classification=0,
            **web_result_provenance(url, _verified_publishers()),
        )
    return results


class _AclGuard:
    """Second enforcement layer: drop what the principal may not read or the
    tenant does not accept.

    The vector store already pre-filters ACLs, so a non-zero ``dropped`` means a
    filter was bypassed or a new retriever skipped it; that is audited as an
    anomaly, never silently absorbed. ``unverified_dropped`` is ordinary policy:
    the tenant requires verified sources and this chunk is not one.
    """

    def __init__(self, principal: Principal, policy: TenantPolicy) -> None:
        self.principal = principal
        self.require_verified = policy.require_verified_sources
        self.dropped = 0
        self.unverified_dropped = 0

    def filter(self, chunks: list[dict]) -> list[dict]:
        allowed, dropped = authorize(chunks, self.principal)
        self.dropped += dropped
        trusted = [c for c in allowed if trust_allows(c.get("metadata") or {}, self.require_verified)]
        self.unverified_dropped += len(allowed) - len(trusted)
        return trusted


_UNGROUNDED_REFUSAL = (
    "I could not produce an answer that is fully supported by the "
    "available sources, so I am not going to guess."
)
_FIGURES_REFUSAL = (
    "I could not verify the figures in this answer against the cited sources, "
    "so I am not going to present them."
)


def _apply_answer_policies(
    answer: str, policy: TenantPolicy, ungrounded: bool, verification: Verification | None
) -> tuple[str, list[str]]:
    """Apply the tenant's warn/abstain rules for ungrounded text and unverified figures."""
    figures_failed = verification is not None and not verification.ok
    if not ungrounded and not figures_failed:
        return answer, []

    applied = []
    if ungrounded:
        applied.append(f"ungrounded:{policy.ungrounded_answer_policy}")
    if figures_failed:
        applied.append(f"figures:{policy.numeric_mismatch_policy}")

    if (ungrounded and policy.ungrounded_answer_policy == "abstain") or (
        figures_failed and policy.numeric_mismatch_policy == "abstain"
    ):
        return (_UNGROUNDED_REFUSAL if ungrounded else _FIGURES_REFUSAL), applied

    notes = []
    if ungrounded:
        notes.append("Note: parts of this answer could not be verified against the sources.")
    if figures_failed:
        bad = ", ".join(dict.fromkeys(c.text for c in verification.failed)) if verification else ""
        if verification is not None and not verification.has_citation:
            notes.append("Note: this answer does not cite its sources.")
        if bad:
            notes.append(f"Note: these figures could not be verified against the cited sources: {bad}.")
    return "\n".join(notes) + "\n\n" + answer, applied


def _local_recall_is_weak(candidates: list[dict], reranked: list[dict]) -> bool:
    """True when the indexed corpus is too thin or too irrelevant to answer alone."""
    if len(candidates) < settings.web_search_min_local_results:
        return True

    # rerank_score is absent when the cross-encoder is unavailable, in which case
    # retrieval order carries no absolute relevance signal to threshold against.
    top_score = next(
        (doc["rerank_score"] for doc in reranked if "rerank_score" in doc),
        None,
    )
    if top_score is None:
        return False
    return float(top_score) < settings.web_search_min_relevance_score
