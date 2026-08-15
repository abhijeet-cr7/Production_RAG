"""FastAPI gateway: entry point for all online RAG queries."""

from __future__ import annotations

import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from config.settings import settings
from offline_pipeline.chunkers.router import ChunkingStrategyRouter
from offline_pipeline.extractors.api_extractor import APIExtractor
from offline_pipeline.extractors.document_extractor import DocumentExtractor
from offline_pipeline.extractors.web_extractor import WebExtractor
from offline_pipeline.preprocessors.cleaner import TextCleaner
from offline_pipeline.preprocessors.metadata_enricher import MetadataEnricher
from online_pipeline.cache.embedding_cache import EmbeddingCache, embed_text
from online_pipeline.context_builder.builder import ContextBuilder
from online_pipeline.llm.router import LLMRouter
from online_pipeline.observability.tracing import configure_tracing, traced
from online_pipeline.reranker.reranker import Reranker
from online_pipeline.retrieval.hybrid_retriever import HybridRetriever
from online_pipeline.retrieval.web_search import TavilyWebSearch
from vector_db.client import VectorDBClient


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


class IngestApiRequest(BaseModel):
    url: str
    params: dict | None = None
    text_fields: list[str] | None = None


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


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    global _llm_router, _cache, _retriever, _reranker, _context_builder, _vector_db, _web_search
    configure_tracing()
    _vector_db = VectorDBClient()
    _vector_db.ensure_collection()
    _llm_router = LLMRouter()
    _cache = EmbeddingCache(redis_url=settings.redis_url)
    _retriever = HybridRetriever(vector_db=_vector_db)
    _reranker = Reranker()
    _context_builder = ContextBuilder()
    _web_search = TavilyWebSearch()
    yield


app = FastAPI(title="Production RAG API", version="0.1.0", lifespan=lifespan)


# ── Routes ───────────────────────────────────────────────────────────────────


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@traced(name="index_document", run_type="chain")
def _index_document(doc: dict) -> tuple[str, int]:
    """Clean, enrich, chunk, embed and upsert *doc*. Returns (doc_id, chunk count)."""
    doc["text"] = TextCleaner().clean(doc["text"])
    doc = MetadataEnricher().enrich(doc)

    chunker = ChunkingStrategyRouter(
        strategy=settings.chunking_strategy,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = chunker.chunk(doc["text"], doc["metadata"])

    if not chunks:
        raise HTTPException(status_code=422, detail="Document produced no text chunks.")

    for chunk in chunks:
        vector = embed_text(chunk["text"])
        # Qdrant requires IDs to be unsigned ints or UUIDs.
        # Derive a stable UUID5 from the doc_id + chunk_index.
        point_id = str(uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"{doc['metadata']['doc_id']}_{chunk['metadata']['chunk_index']}",
        ))
        _vector_db.upsert([{
            "id": point_id,
            "vector": vector,
            "payload": chunk["metadata"],
            "text": chunk["text"],
        }])

    return doc["metadata"]["doc_id"], len(chunks)


@app.post("/ingest", response_model=IngestResponse)
async def ingest(files: list[UploadFile] = File(...)) -> IngestResponse:
    """Upload and index one or more documents (PDF, DOCX, or TXT).

    Each file runs through the full offline pipeline:
    extract → clean → enrich → chunk → embed → upsert.
    Files are processed independently; a failure in one does not abort the rest.
    """
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
            continue

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        try:
            doc = DocumentExtractor().extract(tmp_path)
            doc["metadata"].setdefault("source", filename)
            doc["metadata"].setdefault("file_type", suffix.lstrip("."))

            doc_id, chunk_count = _index_document(doc)
            results.append(IngestFileResult(
                filename=filename,
                status="indexed",
                chunks_indexed=chunk_count,
                doc_id=doc_id,
            ))
        except HTTPException as exc:
            results.append(IngestFileResult(filename=filename, status="error", detail=str(exc.detail)))
        except Exception as exc:  # noqa: BLE001
            results.append(IngestFileResult(filename=filename, status="error", detail=str(exc)))
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    if all(r.status == "error" for r in results):
        raise HTTPException(status_code=400, detail="; ".join(f"{r.filename}: {r.detail}" for r in results))

    return IngestResponse(
        files=results,
        total_chunks_indexed=sum(r.chunks_indexed for r in results),
    )


@app.post("/ingest/url", response_model=IngestSourceResponse)
async def ingest_url(request: IngestUrlRequest) -> IngestSourceResponse:
    """Scrape a web page and index its text content."""
    if not request.url.strip().lower().startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="URL must start with http:// or https://")

    try:
        doc = WebExtractor().extract(request.url.strip())
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=f"Failed to fetch URL: {exc}") from exc

    doc_id, chunk_count = _index_document(doc)
    return IngestSourceResponse(source=request.url, chunks_indexed=chunk_count, doc_id=doc_id)


@app.post("/ingest/api", response_model=IngestSourceResponse)
async def ingest_api(request: IngestApiRequest) -> IngestSourceResponse:
    """Fetch JSON from a REST endpoint and index it."""
    if not request.url.strip().lower().startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="URL must start with http:// or https://")

    try:
        doc = APIExtractor().extract(
            request.url.strip(),
            params=request.params,
            text_fields=request.text_fields,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=f"Failed to fetch API: {exc}") from exc

    doc_id, chunk_count = _index_document(doc)
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
def _stage_generate(query: str, context: str, chat_history: list[dict]) -> str:
    return _llm_router.generate_answer(
        query=query,
        context=context,
        chat_history=chat_history,
    )


@app.post("/query", response_model=QueryResponse)
@traced(name="rag_query", run_type="chain")
async def query(request: QueryRequest) -> QueryResponse:
    """Run the full online RAG pipeline for the given user query."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    rewritten = _stage_rewrite(request.query)
    embedding = _stage_embed(rewritten)
    candidates = _stage_retrieve(rewritten, embedding, request.metadata_filter)
    reranked = _stage_rerank(rewritten, candidates, request.top_k)

    # Top up with live web results when the local corpus answers poorly, then
    # rerank everything together so both sources compete on one relevance scale.
    allow_web = settings.web_search_enabled if request.web_search is None else request.web_search
    web_search_used = False
    if allow_web and _web_search is not None and _web_search.available:
        if _local_recall_is_weak(candidates, reranked):
            web_results = _stage_web_search(rewritten)
            if web_results:
                reranked = _stage_rerank(rewritten, candidates + web_results, request.top_k)
                web_search_used = True

    context = _stage_build_context(reranked)
    answer = _stage_generate(request.query, context, request.chat_history)

    sources = [r.get("metadata", {}) for r in reranked]
    return QueryResponse(answer=answer, sources=sources, web_search_used=web_search_used)


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
