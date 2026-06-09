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
from offline_pipeline.chunkers.text_chunker import TextChunker
from offline_pipeline.extractors.document_extractor import DocumentExtractor
from offline_pipeline.preprocessors.cleaner import TextCleaner
from offline_pipeline.preprocessors.metadata_extractor import MetadataExtractor
from online_pipeline.cache.embedding_cache import EmbeddingCache, embed_text
from online_pipeline.context_builder.builder import ContextBuilder
from online_pipeline.llm.llm_client import LLMClient
from online_pipeline.query_rewrite.rewriter import QueryRewriter
from online_pipeline.reranker.reranker import Reranker
from online_pipeline.retrieval.hybrid_retriever import HybridRetriever
from vector_db.client import VectorDBClient


# ── Request / Response models ────────────────────────────────────────────────

_ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}


class QueryRequest(BaseModel):
    query: str
    metadata_filter: dict | None = None
    top_k: int = settings.top_k_rerank
    chat_history: list[dict[str, str]] = Field(default_factory=list)


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]


class IngestResponse(BaseModel):
    filename: str
    chunks_indexed: int
    doc_id: str


# ── Dependency singletons (initialised at startup) ───────────────────────────

_rewriter: QueryRewriter | None = None
_cache: EmbeddingCache | None = None
_retriever: HybridRetriever | None = None
_reranker: Reranker | None = None
_context_builder: ContextBuilder | None = None
_llm: LLMClient | None = None
_vector_db: VectorDBClient | None = None


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    global _rewriter, _cache, _retriever, _reranker, _context_builder, _llm, _vector_db
    _vector_db = VectorDBClient()
    _vector_db.ensure_collection()
    _rewriter = QueryRewriter()
    _cache = EmbeddingCache(redis_url=settings.redis_url)
    _retriever = HybridRetriever(vector_db=_vector_db)
    _reranker = Reranker()
    _context_builder = ContextBuilder()
    _llm = LLMClient()
    yield


app = FastAPI(title="Production RAG API", version="0.1.0", lifespan=lifespan)


# ── Routes ───────────────────────────────────────────────────────────────────


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)) -> IngestResponse:
    """Upload and index a document (PDF, DOCX, or TXT) into the vector store.

    Steps:
        1. Validate file extension.
        2. Save to a temp file and extract text via DocumentExtractor.
        3. Clean text with TextCleaner.
        4. Enrich metadata with MetadataExtractor.
        5. Split into chunks with TextChunker.
        6. Embed each chunk and upsert into Qdrant.
    """
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(_ALLOWED_EXTENSIONS)}",
        )

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        # 1. Extract
        doc = DocumentExtractor().extract(tmp_path)
        doc["metadata"].setdefault("source", file.filename)
        doc["metadata"].setdefault("file_type", suffix.lstrip("."))

        # 2. Clean
        doc["text"] = TextCleaner().clean(doc["text"])

        # 3. Enrich metadata
        doc = MetadataExtractor().enrich(doc)

        # 4. Chunk
        chunker = TextChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        chunks = chunker.chunk(doc["text"], doc["metadata"])

        if not chunks:
            raise HTTPException(status_code=422, detail="Document produced no text chunks.")

        # 5. Embed + upsert
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

    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return IngestResponse(
        filename=file.filename or "",
        chunks_indexed=len(chunks),
        doc_id=doc["metadata"]["doc_id"],
    )


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
    answer = _llm.generate(
        query=request.query,
        context=context,
        chat_history=request.chat_history,
    )

    sources = [r.get("metadata", {}) for r in reranked]
    return QueryResponse(answer=answer, sources=sources)
