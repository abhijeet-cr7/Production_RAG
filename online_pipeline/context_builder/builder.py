"""Context builder: assembles reranked chunks into an LLM-ready context string."""

from __future__ import annotations

from typing import Any


class ContextBuilder:
    """Assemble retrieved document chunks into a single context string.

    Args:
        max_tokens: Approximate maximum tokens for the assembled context.
            Chunks are added in order until the budget is exhausted.
        separator: String inserted between consecutive chunks.
    """

    def __init__(self, max_tokens: int = 3000, separator: str = "\n\n---\n\n") -> None:
        self.max_tokens = max_tokens
        self.separator = separator

    # ── public API ───────────────────────────────────────────────────────────

    def build(self, chunks: list[dict[str, Any]]) -> str:
        """Return a formatted context string from *chunks*.

        Each chunk is prefixed with its source metadata when available.
        Chunks are included in order until *max_tokens* is reached.

        Args:
            chunks: Reranked list of chunk dicts, each with at least a
                ``text`` key and an optional ``metadata`` key.

        Returns:
            Formatted context string ready to be injected into an LLM prompt.
        """
        parts: list[str] = []
        token_budget = self.max_tokens

        for i, chunk in enumerate(chunks, start=1):
            text = chunk.get("text", "").strip()
            meta = chunk.get("metadata", {})
            source = meta.get("source", "")

            header = f"[{i}] Source: {source}" if source else f"[{i}]"
            block = f"{header}\n{text}"

            estimated_tokens = len(block.split())
            if token_budget <= 0:
                break
            if estimated_tokens > token_budget:
                break
            parts.append(block)
            token_budget -= estimated_tokens

        return self.separator.join(parts)
