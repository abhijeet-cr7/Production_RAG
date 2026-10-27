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




# ── Generation grading ───────────────────────────────────────────────────────

class TestGenerationGrader:
    def test_parses_plain_json(self):
        from online_pipeline.grading.grader import GenerationGrader

        result = GenerationGrader.parse(
            '{"grounded": true, "answers_question": false, "unsupported_claims": []}'
        )
        assert result.grounded is True
        assert result.answers_question is False
        assert result.acceptable is False

    def test_parses_json_wrapped_in_code_fence(self):
        from online_pipeline.grading.grader import GenerationGrader

        raw = '```json\n{"grounded": false, "answers_question": true, "unsupported_claims": ["x"]}\n```'
        result = GenerationGrader.parse(raw)
        assert result.grounded is False
        assert result.unsupported_claims == ["x"]

    def test_string_verdicts_are_not_truthy(self):
        from online_pipeline.grading.grader import GenerationGrader

        result = GenerationGrader.parse('{"grounded": "false", "answers_question": "no"}')
        assert result.grounded is False
        assert result.answers_question is False

    def test_rejects_unparseable_or_incomplete_reply(self):
        from online_pipeline.grading.grader import GenerationGrader

        with pytest.raises(ValueError):
            GenerationGrader.parse("looks fine to me")
        with pytest.raises(ValueError):
            GenerationGrader.parse('{"grounded": true}')

    def test_feedback_differs_for_ungrounded_and_unhelpful(self):
        from online_pipeline.grading.grader import GradeResult

        assert "does not support" in GradeResult(False, True, ["moon is cheese"]).feedback()
        assert "moon is cheese" in GradeResult(False, True, ["moon is cheese"]).feedback()
        assert "did not answer" in GradeResult(True, False).feedback()

    def test_router_grading_fails_open_when_all_providers_fail(self):
        from online_pipeline.llm.router import LLMRouter

        with patch("online_pipeline.llm.router.GenerationGrader") as grader_cls:
            grader_cls.return_value.grade.side_effect = RuntimeError("down")
            router = LLMRouter(fallback_providers="gemini")
            assert router.grade_answer("q", "ctx", "a") is None
        assert grader_cls.call_count == 2  # primary + one fallback


class TestGradedQueryFlow:
    """Drive gateway.query() with every pipeline stage stubbed out."""

    def _run(self, grades, *, web_hits=None, policy="warn", max_attempts=2, allow_web=True):
        import asyncio
        from online_pipeline.api import gateway
        from online_pipeline.grading.grader import GradeResult

        grade_iter = iter([GradeResult(*g) if g is not None else None for g in grades])
        generate = MagicMock(side_effect=lambda *a, **k: f"draft{generate.call_count}")
        web = MagicMock(return_value=web_hits or [])
        web_client = MagicMock(available=allow_web)

        patches = [
            patch.object(gateway, "_stage_rewrite", side_effect=lambda q: q),
            patch.object(gateway, "_stage_embed", return_value=[0.0]),
            patch.object(gateway, "_stage_retrieve", return_value=[{"text": "t", "metadata": {}}] * 5),
            patch.object(gateway, "_stage_rerank", side_effect=lambda q, c, k: c[:k]),
            patch.object(gateway, "_local_recall_is_weak", return_value=False),
            patch.object(gateway, "_stage_build_context", return_value="ctx"),
            patch.object(gateway, "_stage_generate", generate),
            patch.object(gateway, "_stage_grade", side_effect=lambda *a: next(grade_iter)),
            patch.object(gateway, "_stage_web_search", web),
            patch.object(gateway, "_web_search", web_client),
            patch.object(gateway.settings, "web_search_enabled", True),
            patch.object(gateway.settings, "answer_grading_enabled", True),
            patch.object(gateway.settings, "ungrounded_answer_policy", policy),
            patch.object(gateway.settings, "max_generation_attempts", max_attempts),
        ]
        for p in patches:
            p.start()
        try:
            response = asyncio.run(gateway.query(gateway.QueryRequest(query="q")))
        finally:
            for p in reversed(patches):
                p.stop()
        return response, generate, web

    def test_accepted_first_draft_makes_no_extra_calls(self):
        response, generate, web = self._run([(True, True, [])])
        assert response.answer == "draft1"
        assert (response.grounded, response.answer_useful, response.generation_attempts) == (True, True, 1)
        assert generate.call_count == 1 and web.call_count == 0

    def test_ungrounded_draft_is_regenerated_with_feedback(self):
        response, generate, _ = self._run([(False, True, ["bad"]), (True, True, [])])
        assert response.answer == "draft2"
        assert response.generation_attempts == 2
        assert "bad" in generate.call_args_list[1].args[3]

    def test_unhelpful_answer_triggers_web_search_then_regenerates(self):
        hits = [{"text": "web", "metadata": {}}]
        response, generate, web = self._run([(True, False, []), (True, True, [])], web_hits=hits)
        assert web.call_count == 1
        assert response.web_search_used is True
        assert response.answer == "draft2"

    def test_persistent_hallucination_is_flagged_with_warning(self):
        response, generate, _ = self._run(
            [(False, True, []), (False, True, [])], allow_web=False
        )
        assert generate.call_count == 2  # bounded by max_generation_attempts
        assert response.grounded is False
        assert response.answer.startswith("Note: parts of this answer")

    def test_persistent_hallucination_can_abstain(self):
        response, _, _ = self._run(
            [(False, True, []), (False, True, [])], policy="abstain", allow_web=False
        )
        assert "not going to guess" in response.answer
        assert "draft" not in response.answer

    def test_grader_outage_returns_answer_ungraded(self):
        response, generate, _ = self._run([None])
        assert response.answer == "draft1"
        assert response.grounded is None and response.answer_useful is None
        assert generate.call_count == 1
