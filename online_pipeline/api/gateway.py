"""FastAPI gateway: entry point for all online RAG queries."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from config.settings import settings
from online_pipeline.cache.embedding_cache import EmbeddingCache
from online_pipeline.context_builder.builder import ContextBuilder
from online_pipeline.llm.llm_client import LLMClient
from online_pipeline.query_rewrite.rewriter import QueryRewriter
from online_pipeline.reranker.reranker import Reranker
from online_pipeline.retrieval.hybrid_retriever import HybridRetriever


# ── Request / Response models ────────────────────────────────────────────────


class QueryRequest(BaseModel):
    query: str
    metadata_filter: dict | None = None
    top_k: int = settings.top_k_rerank


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]


# ── Dependency singletons (initialised at startup) ───────────────────────────

_rewriter: QueryRewriter | None = None
_cache: EmbeddingCache | None = None
_retriever: HybridRetriever | None = None
_reranker: Reranker | None = None
_context_builder: ContextBuilder | None = None
_llm: LLMClient | None = None


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    global _rewriter, _cache, _retriever, _reranker, _context_builder, _llm
    _rewriter = QueryRewriter()
    _cache = EmbeddingCache(redis_url=settings.redis_url)
    _retriever = HybridRetriever()
    _reranker = Reranker()
    _context_builder = ContextBuilder()
    _llm = LLMClient()
    yield


app = FastAPI(title="Production RAG API", version="0.1.0", lifespan=lifespan)


# ── Routes ───────────────────────────────────────────────────────────────────


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """Run the full online RAG pipeline for the given user query."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    # 1. Query rewrite
    rewritten = _rewriter.rewrite(request.query)

    # 2. Embedding with cache
    embedding = _cache.get(rewritten)
    if embedding is None:
        from online_pipeline.cache.embedding_cache import embed_text
        embedding = embed_text(rewritten)
        _cache.set(rewritten, embedding)

    # 3. Hybrid retrieval + metadata filtering
    candidates = _retriever.retrieve(
        query_text=rewritten,
        query_vector=embedding,
        metadata_filter=request.metadata_filter,
        top_k=settings.top_k_retrieval,
    )

    # 4. Rerank
    reranked = _reranker.rerank(rewritten, candidates, top_k=request.top_k)

    # 5. Build context
    context = _context_builder.build(reranked)

    # 6. LLM generation
    answer = _llm.generate(query=request.query, context=context)

    sources = [r.get("metadata", {}) for r in reranked]
    return QueryResponse(answer=answer, sources=sources)
