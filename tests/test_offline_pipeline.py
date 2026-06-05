"""Unit tests for the offline pipeline components."""

from __future__ import annotations

import pytest


# ── TextCleaner ───────────────────────────────────────────────────────────────

class TestTextCleaner:
    def setup_method(self):
        from offline_pipeline.preprocessors.cleaner import TextCleaner
        self.cleaner = TextCleaner()

    def test_collapses_multiple_blank_lines(self):
        raw = "Hello\n\n\n\nWorld"
        result = self.cleaner.clean(raw)
        assert "\n\n\n" not in result
        assert "Hello" in result
        assert "World" in result

    def test_collapses_multiple_spaces(self):
        raw = "too   many    spaces"
        result = self.cleaner.clean(raw)
        assert "  " not in result

    def test_strips_leading_trailing_whitespace(self):
        raw = "   hello world   "
        assert self.cleaner.clean(raw) == "hello world"

    def test_normalises_unicode(self):
        # NFKC normalisation should collapse fullwidth characters.
        # U+FF28..U+FF4F are FULLWIDTH LATIN letters; NFKC maps them to ASCII
        # but preserves the original case (Ｈ → H, ｈ → h).
        raw = "\uff48\uff45\uff4c\uff4c\uff4f"  # ｈｅｌｌｏ (all lowercase fullwidth)
        result = self.cleaner.clean(raw)
        assert result == "hello"

    def test_empty_string(self):
        assert self.cleaner.clean("") == ""


# ── MetadataExtractor ─────────────────────────────────────────────────────────

class TestMetadataExtractor:
    def setup_method(self):
        from offline_pipeline.preprocessors.metadata_extractor import MetadataExtractor
        self.extractor = MetadataExtractor()

    def test_adds_required_fields(self):
        doc = {"text": "hello world", "metadata": {"source": "test.txt"}}
        enriched = self.extractor.enrich(doc)
        meta = enriched["metadata"]
        assert "doc_id" in meta
        assert "char_count" in meta
        assert "word_count" in meta
        assert "ingested_at" in meta

    def test_doc_id_is_deterministic(self):
        doc = {"text": "same text", "metadata": {}}
        id1 = self.extractor.enrich(doc)["metadata"]["doc_id"]
        id2 = self.extractor.enrich(doc)["metadata"]["doc_id"]
        assert id1 == id2

    def test_word_count(self):
        doc = {"text": "one two three four", "metadata": {}}
        enriched = self.extractor.enrich(doc)
        assert enriched["metadata"]["word_count"] == 4

    def test_preserves_existing_metadata(self):
        doc = {"text": "abc", "metadata": {"source": "s3://bucket/file.pdf"}}
        enriched = self.extractor.enrich(doc)
        assert enriched["metadata"]["source"] == "s3://bucket/file.pdf"


# ── TextChunker ───────────────────────────────────────────────────────────────

class TestTextChunker:
    def setup_method(self):
        from offline_pipeline.chunkers.text_chunker import TextChunker
        # Use word-based splitting (no tiktoken needed in tests)
        self.chunker = TextChunker(chunk_size=5, chunk_overlap=1)

    def test_single_chunk_for_short_text(self):
        chunks = self.chunker.chunk("one two three")
        assert len(chunks) == 1

    def test_multiple_chunks(self):
        words = " ".join(f"word{i}" for i in range(20))
        chunks = self.chunker.chunk(words)
        assert len(chunks) > 1

    def test_chunk_metadata_has_index(self):
        chunks = self.chunker.chunk("a b c d e f g h i j")
        for i, chunk in enumerate(chunks):
            assert chunk["metadata"]["chunk_index"] == i

    def test_base_metadata_propagated(self):
        chunks = self.chunker.chunk("hello world", metadata={"source": "file.txt"})
        assert chunks[0]["metadata"]["source"] == "file.txt"

    def test_invalid_overlap_raises(self):
        from offline_pipeline.chunkers.text_chunker import TextChunker
        with pytest.raises(ValueError):
            TextChunker(chunk_size=10, chunk_overlap=10)
