"""Text chunker: splits cleaned text into overlapping token-aware chunks."""

from __future__ import annotations

from typing import Any


class TextChunker:
    """Split *text* into chunks of at most *chunk_size* tokens with *overlap*.

    Token counting uses ``tiktoken`` when available, falling back to
    whitespace splitting otherwise.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size.")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._encoder = self._load_encoder()

    # ── public API ───────────────────────────────────────────────────────────

    def chunk(
        self, text: str, metadata: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Return a list of chunk dicts, each with ``text`` and ``metadata``.

        Args:
            text: The cleaned document text to split.
            metadata: Base metadata propagated (and extended) into each chunk.

        Returns:
            List of dicts with keys ``text``, ``metadata``, ``chunk_index``.
        """
        tokens = self._tokenise(text)
        chunks: list[dict[str, Any]] = []
        start = 0
        index = 0

        while start < len(tokens):
            end = start + self.chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text = self._detokenise(chunk_tokens)

            chunk_metadata = dict(metadata or {})
            chunk_metadata["chunk_index"] = index
            chunk_metadata["token_count"] = len(chunk_tokens)

            chunks.append({"text": chunk_text, "metadata": chunk_metadata})

            start += self.chunk_size - self.chunk_overlap
            index += 1

        return chunks

    # ── private helpers ──────────────────────────────────────────────────────

    def _load_encoder(self) -> Any:
        try:
            import tiktoken
            return tiktoken.get_encoding("cl100k_base")
        except ImportError:
            return None

    def _tokenise(self, text: str) -> list[Any]:
        if self._encoder is not None:
            return self._encoder.encode(text)
        return text.split()

    def _detokenise(self, tokens: list[Any]) -> str:
        if self._encoder is not None:
            return self._encoder.decode(tokens)
        return " ".join(tokens)
