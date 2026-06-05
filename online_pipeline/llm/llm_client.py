"""LLM client: wraps OpenAI (default) and Anthropic chat completion APIs."""

from __future__ import annotations

import logging
from typing import Any

from config.settings import settings

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Answer the user's question using ONLY the provided context. "
    "If the context does not contain enough information, say so clearly. "
    "Cite the source numbers (e.g. [1], [2]) where relevant."
)


class LLMClient:
    """Generate answers from a query + retrieved context.

    Args:
        provider: ``"openai"`` (default) or ``"anthropic"``.
        model: Model identifier to use (defaults to provider's recommended model).
        temperature: Sampling temperature.
        max_tokens: Maximum tokens in the completion.
    """

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> None:
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    # ── public API ───────────────────────────────────────────────────────────

    def generate(self, query: str, context: str) -> str:
        """Return a grounded answer for *query* given *context*.

        Args:
            query: The original user question.
            context: Assembled context string from the context builder.

        Returns:
            Generated answer string.
        """
        user_message = f"Context:\n{context}\n\nQuestion: {query}"
        if self.provider == "anthropic":
            return self._anthropic(user_message)
        return self._openai(user_message)

    # ── private helpers ──────────────────────────────────────────────────────

    def _openai(self, user_message: str) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=settings.openai_api_key)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content.strip()

    def _anthropic(self, user_message: str) -> str:
        import anthropic

        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        message = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        return message.content[0].text.strip()
