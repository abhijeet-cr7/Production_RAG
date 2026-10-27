"""FastAPI gateway: entry point for all online RAG queries."""

from __future__ import annotations

import logging
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from urllib.parse import urlsplit

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from config.settings import settings
from offline_pipeline.chunkers.router import ChunkingStrategyRouter
from offline_pipeline.extractors.api_extractor import APIExtractor
from offline_pipeline.extractors.document_extractor import DocumentExtractor
from offline_pipeline.extractors.web_extractor import WebExtractor
from offline_pipeline.preprocessors.cleaner import TextCleaner
from offline_pipeline.preprocessors.metadata_enricher import MetadataEnricher
from online_pipeline.audit.logger import AuditError, get_audit, record
from online_pipeline.auth.acl import ACLError, DocumentACL, build_document_acl
from online_pipeline.auth.deps import require_scope
from online_pipeline.auth.policy import (
    POLICY_DENIALS,
    RESTRICTIVE,
    TenantPolicy,
    TenantPolicyStore,
    url_host_allowed,
    web_search_decision,
)
from online_pipeline.auth.principal import SCOPE_INGEST, SCOPE_QUERY, Principal
from online_pipeline.auth.tokens import validate_auth_config
from online_pipeline.cache.embedding_cache import EmbeddingCache, embed_text, embed_texts
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

_ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}


class QueryRequest(BaseModel):
    query: str
    metadata_filter: dict | None = None
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


class IngestFileResult(BaseModel):
    filename: str
    status: str  # "indexed" | "error"
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


class IngestApiRequest(BaseModel):
    url: str
    params: dict | None = None
    text_fields: list[str] | None = None
    acl_groups: list[str] = Field(default_factory=list)
    acl_public: bool = False
    classification: int | None = None


class IngestSourceResponse(BaseModel):
    source: str
    chunks_indexed: int
    doc_id: str


# ── Dependency singletons (initialised at startup) ───────────────────────────

_llm_router: LLMRouter | None = None
_cache: EmbeddingCache | None = None
_retriever: HybridRetriever | None = None
_reranker: Reranker | None = None
_context_builder: ContextBuilder | None = None
_vector_db: VectorDBClient | None = None
_web_search: TavilyWebSearch | None = None
_policies: TenantPolicyStore | None = None


def _policy_for(principal: Principal) -> TenantPolicy:
    return RESTRICTIVE if _policies is None else _policies.for_tenant(principal.tenant_id)


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    global _llm_router, _cache, _retriever, _reranker, _context_builder, _vector_db, _web_search
    global _policies
    validate_auth_config(settings)  # refuse to start on an unsafe auth setup
    _policies = TenantPolicyStore.load(settings.tenant_policies_path)
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


@traced(name="index_document", run_type="chain")
def _index_document(doc: dict, principal: Principal, acl: DocumentACL) -> tuple[str, int]:
    """Clean, enrich, chunk, embed and upsert *doc*. Returns (doc_id, chunk count).

    Every chunk is stamped with *acl*; tenant and owner come from the token, so
    nothing in the request can change who owns a document.
    """
    doc["text"] = TextCleaner().clean(doc["text"])
    doc = MetadataEnricher().enrich(
        doc, tenant_id=principal.tenant_id, owner_id=principal.user_id
    )

    chunker = ChunkingStrategyRouter(
        strategy=settings.chunking_strategy,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = chunker.chunk(doc["text"], doc["metadata"])

    if not chunks:
        raise HTTPException(status_code=422, detail="Document produced no text chunks.")

    # One batched embed call for the whole document, and one upsert: embedding
    # per chunk dominated ingest time, and a round trip per chunk followed it.
    acl_payload = acl.as_payload()
    for chunk in chunks:
        # Authoritative stamp: overrides anything an extractor or chunker set.
        chunk["metadata"].update(acl_payload)

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

    return doc["metadata"]["doc_id"], len(chunks)


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
) -> None:
    """One audit event per ingested source; origin is fingerprinted, not stored."""
    record(
        "ingest",
        actor=principal,
        resource={"doc_id": doc_id} if doc_id else None,
        decision="allow" if status == "indexed" else "error",
        reason=reason,
        source_type=source_type,
        origin_fp=get_audit().fingerprint(origin),
        status=status,
        chunks=chunks,
    )


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


@app.post("/ingest", response_model=IngestResponse)
async def ingest(
    files: list[UploadFile] = File(...),
    acl_groups: list[str] = Form(default=[]),
    acl_public: bool = Form(default=False),
    classification: int | None = Form(default=None),
    principal: Principal = Depends(require_scope(SCOPE_INGEST)),
) -> IngestResponse:
    """Upload and index one or more documents (PDF, DOCX, or TXT).

    Each file runs through the full offline pipeline:
    extract → clean → enrich → chunk → embed → upsert.
    Files are processed independently; a failure in one does not abort the rest.
    """
    acl = _resolve_acl(
        principal, acl_groups=acl_groups, acl_public=acl_public, classification=classification
    )
    results: list[IngestFileResult] = []

    for file in files:
        filename = file.filename or ""
        suffix = Path(filename).suffix.lower()
        if suffix not in _ALLOWED_EXTENSIONS:
            results.append(IngestFileResult(
                filename=filename,
                status="error",
                detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(_ALLOWED_EXTENSIONS)}",
            ))
            _audit_ingest(principal, "file", filename, "error", reason="unsupported_type")
            continue

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        try:
            doc = DocumentExtractor().extract(tmp_path)
            doc["metadata"].setdefault("source", filename)
            doc["metadata"].setdefault("file_type", suffix.lstrip("."))

            doc_id, chunk_count = _index_document(doc, principal, acl)
            results.append(IngestFileResult(
                filename=filename,
                status="indexed",
                chunks_indexed=chunk_count,
                doc_id=doc_id,
            ))
            _audit_ingest(principal, "file", filename, "indexed", doc_id=doc_id, chunks=chunk_count)
        except HTTPException as exc:
            results.append(IngestFileResult(filename=filename, status="error", detail=str(exc.detail)))
            _audit_ingest(principal, "file", filename, "error", reason=f"http_{exc.status_code}")
        except Exception as exc:  # noqa: BLE001
            results.append(IngestFileResult(filename=filename, status="error", detail=str(exc)))
            _audit_ingest(principal, "file", filename, "error", reason=type(exc).__name__)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    if all(r.status == "error" for r in results):
        raise HTTPException(status_code=400, detail="; ".join(f"{r.filename}: {r.detail}" for r in results))

    # New chunks are invisible to the lexical branch until the index is rebuilt.
    _refresh_lexical_index()

    return IngestResponse(
        files=results,
        total_chunks_indexed=sum(r.chunks_indexed for r in results),
    )


@app.post("/ingest/url", response_model=IngestSourceResponse)
async def ingest_url(
    request: IngestUrlRequest,
    principal: Principal = Depends(require_scope(SCOPE_INGEST)),
) -> IngestSourceResponse:
    """Scrape a web page and index its text content."""
    if not request.url.strip().lower().startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="URL must start with http:// or https://")
    _require_url_ingest(principal, request.url, "url")
    acl = _resolve_acl(
        principal,
        acl_groups=request.acl_groups,
        acl_public=request.acl_public,
        classification=request.classification,
    )

    try:
        doc = WebExtractor().extract(request.url.strip())
    except Exception as exc:  # noqa: BLE001
        _audit_ingest(principal, "url", request.url, "error", reason=type(exc).__name__)
        raise HTTPException(status_code=422, detail=f"Failed to fetch URL: {exc}") from exc

    doc_id, chunk_count = _index_document(doc, principal, acl)
    _audit_ingest(principal, "url", request.url, "indexed", doc_id=doc_id, chunks=chunk_count)
    _refresh_lexical_index()
    return IngestSourceResponse(source=request.url, chunks_indexed=chunk_count, doc_id=doc_id)


@app.post("/ingest/api", response_model=IngestSourceResponse)
async def ingest_api(
    request: IngestApiRequest,
    principal: Principal = Depends(require_scope(SCOPE_INGEST)),
) -> IngestSourceResponse:
    """Fetch JSON from a REST endpoint and index it."""
    if not request.url.strip().lower().startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="URL must start with http:// or https://")
    _require_url_ingest(principal, request.url, "api")
    acl = _resolve_acl(
        principal,
        acl_groups=request.acl_groups,
        acl_public=request.acl_public,
        classification=request.classification,
    )

    try:
        doc = APIExtractor().extract(
            request.url.strip(),
            params=request.params,
            text_fields=request.text_fields,
        )
    except Exception as exc:  # noqa: BLE001
        _audit_ingest(principal, "api", request.url, "error", reason=type(exc).__name__)
        raise HTTPException(status_code=422, detail=f"Failed to fetch API: {exc}") from exc

    doc_id, chunk_count = _index_document(doc, principal, acl)
    _audit_ingest(principal, "api", request.url, "indexed", doc_id=doc_id, chunks=chunk_count)
    _refresh_lexical_index()
    return IngestSourceResponse(source=request.url, chunks_indexed=chunk_count, doc_id=doc_id)


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
    metadata_filter: dict | None,
) -> list[dict]:
    return _retriever.retrieve(
        query_text=query_text,
        query_vector=query_vector,
        metadata_filter=metadata_filter,
        top_k=settings.top_k_retrieval,
    )


@traced(name="tavily_web_search", run_type="tool")
def _stage_web_search(query: str) -> list[dict]:
    return _web_search.search(query)


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

    rewritten = _stage_rewrite(request.query)
    embedding = _stage_embed(rewritten)
    candidates = _stage_retrieve(rewritten, embedding, request.metadata_filter)
    reranked = _stage_rerank(rewritten, candidates, request.top_k)

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
            topped_up = _web_top_up(rewritten, candidates, request.top_k)
            if topped_up is not None:
                reranked, web_search_used = topped_up, True
        elif deny_reason in POLICY_DENIALS:
            web_denied = deny_reason

    context = _stage_build_context(reranked)
    answer = _stage_generate(request.query, context, request.chat_history)
    attempts = 1

    grade: GradeResult | None = None
    ungrounded_policy_applied = False
    if settings.answer_grading_enabled:
        grade = _stage_grade(request.query, context, answer)

        # Ungrounded draft: regenerate with the grader's feedback, bounded by
        # max_generation_attempts so a stubborn model cannot loop forever.
        while (
            grade is not None
            and not grade.grounded
            and attempts < settings.max_generation_attempts
        ):
            answer = _stage_generate(request.query, context, request.chat_history, grade.feedback())
            attempts += 1
            grade = _stage_grade(request.query, context, answer)

        # Still unusable and the web has not been tried: the context itself is
        # the problem, so fetch fresh evidence and make one last attempt.
        if grade is not None and not grade.acceptable and not web_search_used:
            if allow_web:
                topped_up = _web_top_up(rewritten, candidates, request.top_k)
                if topped_up is not None:
                    reranked, web_search_used = topped_up, True
                    context = _stage_build_context(reranked)
                    answer = _stage_generate(
                        request.query, context, request.chat_history, grade.feedback()
                    )
                    attempts += 1
                    grade = _stage_grade(request.query, context, answer)
            elif deny_reason in POLICY_DENIALS:
                web_denied = deny_reason

        if grade is not None and not grade.grounded:
            answer = _apply_ungrounded_policy(answer, policy)
            ungrounded_policy_applied = True

    if web_denied:
        record(
            "policy.deny",
            actor=principal,
            decision="deny",
            reason=web_denied,
            resource="web_search",
        )

    sources = [r.get("metadata", {}) for r in reranked]
    response = QueryResponse(
        answer=answer,
        sources=sources,
        web_search_used=web_search_used,
        grounded=None if grade is None else grade.grounded,
        answer_useful=None if grade is None else grade.answers_question,
        generation_attempts=attempts,
    )
    audit_details = {
        "returned_doc_ids": [m["doc_id"] for m in sources if m.get("doc_id")][:50],
        "n_sources": len(sources),
        "web_used": web_search_used,
        "web_denied": web_denied,
        "grounded": response.grounded,
        "answer_useful": response.answer_useful,
        "attempts": attempts,
        "ungrounded_policy_applied": ungrounded_policy_applied,
    }
    return response, audit_details


def _web_top_up(query: str, candidates: list[dict], top_k: int) -> list[dict] | None:
    """Rerank local candidates together with web hits; None when the web gave nothing."""
    web_results = _stage_web_search(query)
    if not web_results:
        return None
    return _stage_rerank(query, candidates + web_results, top_k)


def _apply_ungrounded_policy(answer: str, policy: TenantPolicy) -> str:
    if policy.ungrounded_answer_policy == "abstain":
        return (
            "I could not produce an answer that is fully supported by the "
            "available sources, so I am not going to guess."
        )
    return (
        "Note: parts of this answer could not be verified against the sources.\n\n"
        + answer
    )


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
