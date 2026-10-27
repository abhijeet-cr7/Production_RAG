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

    def select(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """The chunks that fit the budget, in order. Citation ``[n]`` is position n here."""
        included: list[dict[str, Any]] = []
        token_budget = self.max_tokens
        for chunk in chunks:
            block = self._format(len(included) + 1, chunk)
            meta = chunk.get("metadata", {})
            estimated_tokens = len(block.split())
            if meta.get("block_kind") in ("table", "xbrl_facts"):
                # Numbers tokenise far worse than words ("391,035" is several tokens),
                # so a whitespace count would let tables overrun the model's budget.
                estimated_tokens = max(estimated_tokens, len(block) // 3)
            if token_budget <= 0 or estimated_tokens > token_budget:
                break
            included.append(chunk)
            token_budget -= estimated_tokens
        return included

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
        return self.separator.join(
            self._format(i, chunk) for i, chunk in enumerate(self.select(chunks), start=1)
        )

    @staticmethod
    def _format(index: int, chunk: dict[str, Any]) -> str:
        text = chunk.get("text", "").strip()
        source = (chunk.get("metadata") or {}).get("source", "")
        header = f"[{index}] Source: {source}" if source else f"[{index}]"
        return f"{header}\n{text}"
