"""Unit tests for the online pipeline components."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch


# ── EmbeddingCache ────────────────────────────────────────────────────────────

class TestEmbeddingCache:
    def _make_cache(self, redis_mock):
        from online_pipeline.cache.embedding_cache import EmbeddingCache
        with patch("online_pipeline.cache.embedding_cache.EmbeddingCache._connect", return_value=redis_mock):
            return EmbeddingCache(redis_url="redis://localhost:6379/0")

    def test_cache_miss_returns_none(self):
        redis_mock = MagicMock()
        redis_mock.get.return_value = None
        cache = self._make_cache(redis_mock)
        assert cache.get("unseen query") is None

    def test_cache_hit_returns_vector(self):
        import json
        vector = [0.1, 0.2, 0.3]
        redis_mock = MagicMock()
        redis_mock.get.return_value = json.dumps(vector)
        cache = self._make_cache(redis_mock)
        result = cache.get("some query")
        assert result == vector

    def test_set_calls_setex(self):
        redis_mock = MagicMock()
        cache = self._make_cache(redis_mock)
        cache.set("hello", [1.0, 2.0])
        redis_mock.setex.assert_called_once()

    def test_graceful_degradation_without_redis(self):
        from online_pipeline.cache.embedding_cache import EmbeddingCache
        with patch("online_pipeline.cache.embedding_cache.EmbeddingCache._connect", return_value=None):
            cache = EmbeddingCache(redis_url="redis://invalid")
        # Should not raise
        assert cache.get("q") is None
        cache.set("q", [1.0])  # no-op

    def test_key_is_namespaced_by_model_id(self):
        redis_mock = MagicMock()
        with patch("online_pipeline.cache.embedding_cache.EmbeddingCache._connect", return_value=redis_mock):
            from online_pipeline.cache.embedding_cache import EmbeddingCache

            cache_a = EmbeddingCache(
                redis_url="redis://localhost:6379/0",
                model_id="sentence-transformers:all-MiniLM-L6-v2",
            )
            cache_b = EmbeddingCache(
                redis_url="redis://localhost:6379/0",
                model_id="openai:text-embedding-3-small",
            )

        key_a = cache_a._make_key("same query")
        key_b = cache_b._make_key("same query")
        assert key_a != key_b

    def test_set_uses_namespaced_key(self):
        redis_mock = MagicMock()
        with patch("online_pipeline.cache.embedding_cache.EmbeddingCache._connect", return_value=redis_mock):
            from online_pipeline.cache.embedding_cache import EmbeddingCache

            cache = EmbeddingCache(
                redis_url="redis://localhost:6379/0",
                model_id="provider:model-x",
            )

        cache.set("hello", [1.0, 2.0])
        setex_args = redis_mock.setex.call_args.args
        assert setex_args[0].startswith("emb:provider:model-x:")


# ── BM25Retriever ─────────────────────────────────────────────────────────────

class TestBM25Retriever:
    def _make_corpus(self):
        return [
            {"text": "the cat sat on the mat", "metadata": {"doc_id": "1"}},
            {"text": "dogs are great pets", "metadata": {"doc_id": "2"}},
            {"text": "cats and dogs are common household pets", "metadata": {"doc_id": "3"}},
        ]

    def test_returns_relevant_results(self):
        try:
            from online_pipeline.retrieval.bm25_retriever import BM25Retriever
        except ImportError:
            pytest.skip("rank-bm25 not installed")
        corpus = self._make_corpus()
        retriever = BM25Retriever(corpus)
        results = retriever.retrieve("cat sat", top_k=2)
        assert len(results) <= 2
        texts = [r["text"] for r in results]
        assert any("cat" in t for t in texts)

    def test_top_k_respected(self):
        try:
            from online_pipeline.retrieval.bm25_retriever import BM25Retriever
        except ImportError:
            pytest.skip("rank-bm25 not installed")
        corpus = self._make_corpus()
        retriever = BM25Retriever(corpus)
        results = retriever.retrieve("pets", top_k=1)
        assert len(results) <= 1

    def test_no_match_returns_empty(self):
        try:
            from online_pipeline.retrieval.bm25_retriever import BM25Retriever
        except ImportError:
            pytest.skip("rank-bm25 not installed")
        corpus = self._make_corpus()
        retriever = BM25Retriever(corpus)
        results = retriever.retrieve("xyzzy nonsense gibberish", top_k=5)
        assert isinstance(results, list)


# ── ContextBuilder ────────────────────────────────────────────────────────────

class TestContextBuilder:
    def setup_method(self):
        from online_pipeline.context_builder.builder import ContextBuilder
        self.builder = ContextBuilder(max_tokens=1000)

    def test_empty_chunks_returns_empty_string(self):
        assert self.builder.build([]) == ""

    def test_single_chunk_no_source(self):
        chunks = [{"text": "hello world"}]
        result = self.builder.build(chunks)
        assert "hello world" in result
        assert "[1]" in result

    def test_source_included_in_output(self):
        chunks = [{"text": "foo", "metadata": {"source": "file.pdf"}}]
        result = self.builder.build(chunks)
        assert "file.pdf" in result

    def test_multiple_chunks_separated(self):
        chunks = [
            {"text": "first chunk"},
            {"text": "second chunk"},
        ]
        result = self.builder.build(chunks)
        assert "first chunk" in result
        assert "second chunk" in result

    def test_token_budget_respected(self):
        from online_pipeline.context_builder.builder import ContextBuilder
        # Budget of 10 fits the first chunk (~6 words) but not both chunks.
        builder = ContextBuilder(max_tokens=10)
        chunks = [
            {"text": "a b c d e"},
            {"text": "f g h i j"},
        ]
        result = builder.build(chunks)
        # First chunk is returned; second is excluded (budget exhausted)
        assert "a b c d e" in result
        assert "f g h i j" not in result


# ── HybridRetriever (unit — no real backends) ─────────────────────────────────

class TestHybridRetriever:
    def test_empty_corpus_and_no_vector_db(self):
        from online_pipeline.retrieval.hybrid_retriever import HybridRetriever
        retriever = HybridRetriever()
        results = retriever.retrieve(
            query_text="test", query_vector=[0.1] * 5, top_k=5
        )
        assert results == []

    def test_rrf_fusion_deduplicates(self):
        from online_pipeline.retrieval.hybrid_retriever import HybridRetriever
        retriever = HybridRetriever()
        doc = {"text": "shared doc", "metadata": {"doc_id": "x"}}
        fused = retriever._rrf_fuse([doc], [doc], top_k=5)
        assert len(fused) == 1

    def test_rrf_score_increases_with_both_branches(self):
        from online_pipeline.retrieval.hybrid_retriever import HybridRetriever
        retriever = HybridRetriever()
        shared = {"text": "shared", "metadata": {"doc_id": "s"}}
        bm25_only = {"text": "bm25 only", "metadata": {"doc_id": "b"}}
        fused = retriever._rrf_fuse([shared, bm25_only], [shared], top_k=5)
        scores = {r["metadata"]["doc_id"]: r["hybrid_score"] for r in fused}
        assert scores["s"] > scores["b"]


# ── LLMRouter ────────────────────────────────────────────────────────────────

class TestLLMRouter:
    def test_generate_answer_uses_primary_provider(self):
        from online_pipeline.llm.router import LLMRouter

        with patch("online_pipeline.llm.router.LLMClient") as client_cls:
            client_cls.return_value.generate.return_value = "answer"
            router = LLMRouter(fallback_providers="gemini")

            result = router.generate_answer(query="q", context="ctx")

        assert result == "answer"
        assert client_cls.call_args.kwargs["provider"] == "groq"

    def test_generate_answer_falls_back_to_next_provider(self):
        from online_pipeline.llm.router import LLMRouter

        primary = MagicMock()
        primary.generate.side_effect = RuntimeError("primary failed")
        fallback = MagicMock()
        fallback.generate.return_value = "fallback answer"

        with patch("online_pipeline.llm.router.LLMClient", side_effect=[primary, fallback]) as client_cls:
            router = LLMRouter(fallback_providers="gemini")
            result = router.generate_answer(query="q", context="ctx")

        assert result == "fallback answer"
        assert client_cls.call_args_list[1].kwargs["provider"] == "gemini"


# ── TavilyWebSearch ──────────────────────────────────────────────────────────

class TestTavilyWebSearch:
    def _make_search(self, client=None):
        from online_pipeline.retrieval.web_search import TavilyWebSearch
        with patch.object(TavilyWebSearch, "_connect", return_value=client):
            return TavilyWebSearch(api_key="test-key")

    def test_unavailable_without_client(self):
        search = self._make_search(client=None)
        assert search.available is False
        assert search.search("anything") == []

    def test_normalises_results_to_chunk_shape(self):
        client = MagicMock()
        client.search.return_value = {
            "results": [
                {
                    "title": "Qdrant docs",
                    "url": "https://qdrant.tech/docs",
                    "content": "Qdrant is a vector database.",
                    "score": 0.91,
                }
            ]
        }
        results = self._make_search(client).search("what is qdrant")

        assert len(results) == 1
        assert results[0]["text"] == "Qdrant is a vector database."
        assert results[0]["metadata"]["source"] == "https://qdrant.tech/docs"
        assert results[0]["metadata"]["file_type"] == "web_search"
        assert results[0]["web_score"] == pytest.approx(0.91)

    def test_skips_results_without_content(self):
        client = MagicMock()
        client.search.return_value = {"results": [{"url": "https://x.dev", "content": "   "}]}
        assert self._make_search(client).search("q") == []

    def test_api_failure_degrades_to_empty_list(self):
        client = MagicMock()
        client.search.side_effect = RuntimeError("network down")
        assert self._make_search(client).search("q") == []

    def test_blank_query_skips_api_call(self):
        client = MagicMock()
        assert self._make_search(client).search("   ") == []
        client.search.assert_not_called()


# ── LangSmith tracing ────────────────────────────────────────────────────────

class TestTracing:
    def test_decorator_is_transparent_when_disabled(self):
        from online_pipeline.observability import tracing

        with patch.object(tracing, "configure_tracing", return_value=False):
            @tracing.traced(name="unit", run_type="chain")
            def add(a, b):
                return a + b

            assert add(2, 3) == 5

    def test_wraps_with_langsmith_when_enabled(self):
        from online_pipeline.observability import tracing

        def fake_traceable(name=None, run_type=None):
            def decorator(fn):
                def wrapped(*args, **kwargs):
                    return f"traced:{fn(*args, **kwargs)}"
                return wrapped
            return decorator

        traceable = MagicMock(side_effect=fake_traceable)
        with patch.object(tracing, "configure_tracing", return_value=True), \
             patch.dict("sys.modules", {"langsmith": MagicMock(traceable=traceable)}):
            @tracing.traced(name="unit", run_type="llm")
            def echo(value):
                return value

            assert echo("x") == "traced:x"
        traceable.assert_called_once_with(name="unit", run_type="llm")

    def test_tracing_disabled_without_api_key(self):
        from online_pipeline.observability import tracing

        with patch.object(tracing.settings, "langsmith_tracing", True), \
             patch.object(tracing.settings, "langsmith_api_key", ""):
            assert tracing.tracing_enabled() is False


# ── Web-search trigger ───────────────────────────────────────────────────────

class TestLocalRecallIsWeak:
    def _docs(self, count, score=None):
        docs = []
        for i in range(count):
            doc = {"text": f"chunk {i}", "metadata": {"doc_id": str(i)}}
            if score is not None:
                doc["rerank_score"] = score
            docs.append(doc)
        return docs

    def test_thin_recall_triggers_web_search(self):
        from online_pipeline.api.gateway import _local_recall_is_weak, settings

        candidates = self._docs(settings.web_search_min_local_results - 1, score=9.0)
        assert _local_recall_is_weak(candidates, candidates) is True

    def test_low_relevance_triggers_web_search_despite_enough_hits(self):
        from online_pipeline.api.gateway import _local_recall_is_weak, settings

        candidates = self._docs(settings.web_search_min_local_results + 2)
        reranked = self._docs(3, score=settings.web_search_min_relevance_score - 1.0)
        assert _local_recall_is_weak(candidates, reranked) is True

    def test_strong_relevance_skips_web_search(self):
        from online_pipeline.api.gateway import _local_recall_is_weak, settings

        candidates = self._docs(settings.web_search_min_local_results + 2)
        reranked = self._docs(3, score=settings.web_search_min_relevance_score + 5.0)
        assert _local_recall_is_weak(candidates, reranked) is False

    def test_missing_rerank_scores_do_not_trigger_web_search(self):
        from online_pipeline.api.gateway import _local_recall_is_weak, settings

        # Cross-encoder unavailable: no absolute score exists to threshold against.
        candidates = self._docs(settings.web_search_min_local_results + 2)
        assert _local_recall_is_weak(candidates, candidates) is False


