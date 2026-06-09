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

    # Provider → recommended free model
    _DEFAULT_MODELS: dict[str, str] = {
        "groq": "llama-3.1-8b-instant",   # free tier at console.groq.com
        "gemini": "gemini-1.5-flash",       # free tier at aistudio.google.com
        "cohere": "command-r",              # free tier at dashboard.cohere.com
        "mistral": "mistral-small-latest",  # free tier at console.mistral.ai
        "openai": "gpt-4o-mini",
        "anthropic": "claude-3-haiku-20240307",
    }

    def __init__(
        self,
        provider: str = "groq",
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> None:
        self.provider = provider
        self.model = model or self._DEFAULT_MODELS.get(provider, "llama-3.1-8b-instant")
        self.temperature = temperature
        self.max_tokens = max_tokens

    # ── public API ───────────────────────────────────────────────────────────

    def generate(
        self,
        query: str,
        context: str,
        chat_history: list[dict[str, str]] | None = None,
    ) -> str:
        """Return a grounded answer for *query* given *context*.

        Args:
            query: The original user question.
            context: Assembled context string from the context builder.

        Returns:
            Generated answer string.
        """
        history_blocks: list[str] = []
        for turn in chat_history or []:
            role = (turn.get("role") or "").strip().lower()
            content = (turn.get("content") or "").strip()
            if role not in {"user", "assistant"} or not content:
                continue
            speaker = "User" if role == "user" else "Assistant"
            history_blocks.append(f"{speaker}: {content}")

        history_text = "\n".join(history_blocks) if history_blocks else "(none)"
        user_message = (
            f"Conversation so far:\n{history_text}\n\n"
            f"Context:\n{context}\n\n"
            f"Latest user question: {query}"
        )
        dispatch = {
            "groq": self._groq,
            "gemini": self._gemini,
            "cohere": self._cohere,
            "mistral": self._mistral,
            "anthropic": self._anthropic,
        }
        handler = dispatch.get(self.provider, self._openai)
        return handler(user_message)

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

    def _groq(self, user_message: str) -> str:
        from groq import Groq

        client = Groq(api_key=settings.groq_api_key)
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

    def _gemini(self, user_message: str) -> str:
        import google.generativeai as genai

        genai.configure(api_key=settings.gemini_api_key)
        gemini_model = genai.GenerativeModel(
            model_name=self.model,
            system_instruction=_SYSTEM_PROMPT,
        )
        response = gemini_model.generate_content(
            user_message,
            generation_config=genai.GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
            ),
        )
        return response.text.strip()

    def _cohere(self, user_message: str) -> str:
        import cohere

        client = cohere.ClientV2(api_key=settings.cohere_api_key)
        response = client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )
        return response.message.content[0].text.strip()

    def _mistral(self, user_message: str) -> str:
        from mistralai import Mistral

        client = Mistral(api_key=settings.mistral_api_key)
        response = client.chat.complete(
            model=self.model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content.strip()
