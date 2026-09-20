"""Regression tests for four defects that a fully-mocked suite did not catch.

Each test names the defect it guards. They assert on *behaviour* — how many
candidates survive fusion, how many distinct points reach the store, how many
times a model is constructed — rather than on which collaborators were called,
which is what let the originals pass while the pipeline was broken.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ── Defect 1: the embedding model was reloaded on every call ─────────────────

class TestEmbeddingBackendIsReused:
    def test_model_is_constructed_once_across_many_calls(self):
        from online_pipeline.cache import embedding_cache

        embedding_cache.reset_embedding_backends()
        fake_model = MagicMock()
        fake_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])

        with patch("sentence_transformers.SentenceTransformer", return_value=fake_model) as ctor:
            for _ in range(5):
                embedding_cache.embed_text("what drove the increase in net sales")

        assert ctor.call_count == 1, (
            f"model was constructed {ctor.call_count} times for 5 embeds; "
            "it must be built once per process"
        )
        embedding_cache.reset_embedding_backends()

    def test_batch_embed_is_a_single_encode_call(self):
        from online_pipeline.cache import embedding_cache

        embedding_cache.reset_embedding_backends()
        fake_model = MagicMock()
        fake_model.encode.return_value = np.array([[0.1, 0.2]] * 200)

        with patch("sentence_transformers.SentenceTransformer", return_value=fake_model):
            vectors = embedding_cache.embed_texts([f"chunk {i}" for i in range(200)])

        assert len(vectors) == 200
        assert fake_model.encode.call_count == 1, (
            "200 chunks must be embedded in one batched call, not one call each"
        )
        embedding_cache.reset_embedding_backends()


# ── Defect 2: RRF fusion keyed on doc_id, folding a document's chunks into one ─

class TestFusionKeepsChunksDistinct:
    @staticmethod
    def _chunks(doc_id: str, count: int) -> list[dict]:
        return [
            {"text": f"{doc_id} chunk {i}", "metadata": {"doc_id": doc_id, "chunk_index": i}}
            for i in range(count)
        ]

    def test_chunks_of_one_document_all_survive_fusion(self):
        from online_pipeline.retrieval.hybrid_retriever import HybridRetriever

        hits = self._chunks("AAPL_10K", 18) + self._chunks("MSFT_10K", 2)
        fused = HybridRetriever()._rrf_fuse([], hits, top_k=20)

        assert len(fused) == 20, (
            f"{len(fused)} of 20 chunks survived fusion; keying on doc_id alone "
            "collapses every chunk of a document into one"
        )

    def test_best_ranked_chunk_wins_not_the_last_seen(self):
        from online_pipeline.retrieval.hybrid_retriever import HybridRetriever

        hits = self._chunks("DOC", 10)
        fused = HybridRetriever()._rrf_fuse([], hits, top_k=10)

        # Rank 1 went in first, so it must come out on top.
        assert fused[0]["metadata"]["chunk_index"] == 0
        assert fused[0]["hybrid_score"] > fused[-1]["hybrid_score"]

    def test_a_chunk_found_by_both_branches_scores_higher(self):
        from online_pipeline.retrieval.hybrid_retriever import HybridRetriever

        shared = {"text": "both", "metadata": {"doc_id": "D", "chunk_index": 0}}
        dense_only = {"text": "dense", "metadata": {"doc_id": "D", "chunk_index": 1}}
        fused = HybridRetriever()._rrf_fuse([shared], [shared, dense_only], top_k=5)

        scores = {r["metadata"]["chunk_index"]: r["hybrid_score"] for r in fused}
        assert len(fused) == 2
        assert scores[0] > scores[1]

    def test_documents_without_chunk_index_still_dedupe(self):
        from online_pipeline.retrieval.hybrid_retriever import HybridRetriever

        doc = {"text": "whole document", "metadata": {"doc_id": "x"}}
        assert len(HybridRetriever()._rrf_fuse([doc], [doc], top_k=5)) == 1


# ── Defect 3: the Kafka worker used doc_id as the point ID ───────────────────

class TestWorkerPointIdentity:
    @staticmethod
    def _worker(chunks: list[dict], batch_size: int = 32):
        from offline_pipeline.embedding_workers.worker import EmbeddingWorker

        consumer = MagicMock()
        consumer.consume.return_value = iter(chunks)
        embedder = MagicMock()
        embedder.embed.side_effect = lambda texts: [[0.1, 0.2] for _ in texts]
        vector_db = MagicMock()
        worker = EmbeddingWorker(consumer, embedder, vector_db, batch_size=batch_size)
        return worker, vector_db

    @staticmethod
    def _chunks(count: int) -> list[dict]:
        return [
            {"text": f"chunk {i}", "metadata": {"doc_id": "FILING_A", "chunk_index": i}}
            for i in range(count)
        ]

    def test_every_chunk_gets_a_distinct_point_id(self):
        worker, vector_db = self._worker(self._chunks(64), batch_size=32)
        worker.run()

        ids = [p["id"] for call in vector_db.upsert.call_args_list for p in call.args[0]]
        assert len(ids) == 64
        assert len(set(ids)) == 64, (
            f"{len(set(ids))} distinct IDs for 64 chunks; colliding IDs mean each "
            "upsert overwrites the last and only one chunk survives"
        )

    def test_point_ids_are_stable_across_runs(self):
        first, db_a = self._worker(self._chunks(8), batch_size=8)
        first.run()
        second, db_b = self._worker(self._chunks(8), batch_size=8)
        second.run()

        ids_a = [p["id"] for p in db_a.upsert.call_args_list[0].args[0]]
        ids_b = [p["id"] for p in db_b.upsert.call_args_list[0].args[0]]
        assert ids_a == ids_b, "re-ingesting a document must update in place, not duplicate"

    def test_final_partial_batch_is_flushed(self):
        # 70 chunks at batch_size 32 leaves 6 in the tail.
        worker, vector_db = self._worker(self._chunks(70), batch_size=32)
        worker.run()

        upserted = sum(len(call.args[0]) for call in vector_db.upsert.call_args_list)
        assert upserted == 70, f"{upserted} of 70 chunks landed; the tail batch was dropped"


# ── Defect 4: BM25 was never constructed, so "hybrid" was dense-only ─────────

class TestLexicalBranchRuns:
    def test_retriever_starts_dense_only(self):
        from online_pipeline.retrieval.hybrid_retriever import HybridRetriever

        assert HybridRetriever().lexical_enabled is False

    def test_load_corpus_activates_the_lexical_branch(self):
        pytest.importorskip("rank_bm25")
        from online_pipeline.retrieval.hybrid_retriever import HybridRetriever

        retriever = HybridRetriever()
        retriever.load_corpus([
            {"text": "net sales rose on Services revenue",
             "metadata": {"doc_id": "A", "chunk_index": 0}},
            {"text": "CUSIP 037833100 term sheet",
             "metadata": {"doc_id": "B", "chunk_index": 0}},
        ])
        assert retriever.lexical_enabled is True

    def test_lexical_branch_finds_an_exact_match_dense_search_missed(self):
        pytest.importorskip("rank_bm25")
        from online_pipeline.retrieval.hybrid_retriever import HybridRetriever

        # A realistic corpus size matters here: BM25Okapi's IDF is
        # log(N - df + 0.5) - log(df + 0.5), which is exactly 0 for a term in
        # one of two documents, so a toy corpus scores every match at zero.
        corpus = [
            {"text": f"commentary on segment margins for period {i}",
             "metadata": {"doc_id": "C", "chunk_index": i}}
            for i in range(30)
        ]
        corpus.append({
            "text": "CUSIP 037833100 appears in the term sheet",
            "metadata": {"doc_id": "B", "chunk_index": 0},
        })

        retriever = HybridRetriever()
        retriever.load_corpus(corpus)
        # No vector branch at all: anything returned came from BM25.
        results = retriever.retrieve(query_text="037833100", query_vector=[], top_k=5)
        assert results, "the lexical branch returned nothing for an exact token match"
        assert results[0]["metadata"]["doc_id"] == "B"
