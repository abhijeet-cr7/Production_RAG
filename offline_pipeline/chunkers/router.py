"""Chunking strategy router for ingestion orchestration."""

from __future__ import annotations

from typing import Any

from offline_pipeline.chunkers.recursive_chunker import RecursiveTextChunker
from offline_pipeline.chunkers.text_chunker import TextChunker


class ChunkingStrategyRouter:
    """Select a chunking strategy from document metadata and structure."""

    def __init__(
        self,
        strategy: str = "auto",
        chunk_size: int = 512,
        chunk_overlap: int = 64,
    ) -> None:
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(
        self, text: str, metadata: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Chunk text with the selected or automatically inferred strategy."""
        selected = self.select_strategy(text=text, metadata=metadata)
        chunker = self._chunker_for(selected)
        chunks = chunker.chunk(text, metadata)

        for chunk in chunks:
            chunk.setdefault("metadata", {})["chunking_strategy"] = selected
        return chunks

    def select_strategy(self, text: str, metadata: dict[str, Any] | None = None) -> str:
        """Return the concrete strategy name for a document."""
        requested = (self.strategy or "auto").lower()
        if requested != "auto":
            return requested

        file_type = str((metadata or {}).get("file_type", "")).lower().lstrip(".")
        if file_type in {"md", "markdown", "html", "web"}:
            return "recursive"
        if self._has_natural_structure(text):
            return "recursive"
        return "fixed_token"

    def _chunker_for(self, strategy: str) -> TextChunker | RecursiveTextChunker:
        if strategy == "recursive":
            return RecursiveTextChunker(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
        if strategy == "fixed_token":
            return TextChunker(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
        raise ValueError(f"Unsupported chunking strategy: {strategy}")

    @staticmethod
    def _has_natural_structure(text: str) -> bool:
        return "\n\n" in text or any(line.startswith("#") for line in text.splitlines())