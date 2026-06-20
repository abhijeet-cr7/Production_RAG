"""Recursive text chunker for preserving natural text boundaries."""

from __future__ import annotations

from typing import Any

from offline_pipeline.chunkers.text_chunker import TextChunker


class RecursiveTextChunker:
    """Split text by natural boundaries before falling back to token windows."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        separators: list[str] | None = None,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " "]
        self._fixed_chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def chunk(
        self, text: str, metadata: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Return chunks that prefer paragraph, line, sentence and word boundaries."""
        pieces = self._split_recursive(text.strip(), self.separators)
        chunk_texts = self._merge_pieces(pieces)
        chunks: list[dict[str, Any]] = []

        for index, chunk_text in enumerate(chunk_texts):
            chunk_metadata = dict(metadata or {})
            chunk_metadata["chunk_index"] = index
            chunk_metadata["token_count"] = self._token_count(chunk_text)
            chunk_metadata["chunking_strategy"] = "recursive"
            chunks.append({"text": chunk_text, "metadata": chunk_metadata})

        return chunks

    def _split_recursive(self, text: str, separators: list[str]) -> list[str]:
        if not text:
            return []
        if self._token_count(text) <= self.chunk_size:
            return [text]
        if not separators:
            return [chunk["text"] for chunk in self._fixed_chunker.chunk(text)]

        separator = separators[0]
        raw_parts = [part.strip() for part in text.split(separator) if part.strip()]
        if len(raw_parts) <= 1:
            return self._split_recursive(text, separators[1:])

        parts: list[str] = []
        for part in raw_parts:
            parts.extend(self._split_recursive(part, separators[1:]))
        return parts

    def _merge_pieces(self, pieces: list[str]) -> list[str]:
        chunks: list[str] = []
        current: list[str] = []

        for piece in pieces:
            candidate = self._join(current + [piece])
            if current and self._token_count(candidate) > self.chunk_size:
                previous = self._join(current)
                chunks.append(previous)
                overlap = self._overlap_text(previous)
                current = [overlap, piece] if overlap else [piece]
            else:
                current.append(piece)

        if current:
            chunks.append(self._join(current))
        return [chunk for chunk in chunks if chunk]

    @staticmethod
    def _join(parts: list[str]) -> str:
        return "\n\n".join(part for part in parts if part).strip()

    def _overlap_text(self, text: str) -> str:
        if self.chunk_overlap <= 0:
            return ""
        tokens = self._fixed_chunker._tokenise(text)
        if not tokens:
            return ""
        return self._fixed_chunker._detokenise(tokens[-self.chunk_overlap:])

    def _token_count(self, text: str) -> int:
        return len(self._fixed_chunker._tokenise(text))