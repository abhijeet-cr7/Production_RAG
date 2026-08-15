"""LLM router: selects models for different RAG tasks with fallbacks."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from config.settings import settings
from online_pipeline.llm.llm_client import LLMClient
from online_pipeline.query_rewrite.rewriter import QueryRewriter

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelRoute:
    """Provider/model selection for a single LLM task."""

    provider: str
    model: str | None = None
    temperature: float = 0.2
    max_tokens: int = 1024


class LLMRouter:
    """Route LLM work by task, routing mode and provider fallback order."""

    def __init__(
        self,
        routing_mode: str | None = None,
        fallback_providers: str | None = None,
    ) -> None:
        self.routing_mode = routing_mode or settings.llm_routing_mode
        fallback_csv = fallback_providers or settings.llm_fallback_providers
        self.fallback_providers = [
            provider.strip() for provider in fallback_csv.split(",") if provider.strip()
        ]

    def rewrite_query(self, query: str) -> str:
        """Rewrite a user query with the route selected for retrieval rewriting."""
        route = self._route_for("query_rewrite")
        providers = self._providers_with_fallbacks(route.provider)
        for provider in providers:
            try:
                return QueryRewriter(provider=provider, model=route.model or "").rewrite(query)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Query rewrite provider %s failed: %s", provider, exc)
        return query

    def generate_answer(
        self,
        query: str,
        context: str,
        chat_history: list[dict[str, str]] | None = None,
    ) -> str:
        """Generate a grounded answer with the selected answer model."""
        route = self._route_for("rag_answer")
        last_error: Exception | None = None
        for provider in self._providers_with_fallbacks(route.provider):
            try:
                client = LLMClient(
                    provider=provider,
                    model=route.model,
                    temperature=route.temperature,
                    max_tokens=route.max_tokens,
                )
                return client.generate(query=query, context=context, chat_history=chat_history)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning("Answer provider %s failed: %s", provider, exc)

        raise RuntimeError("All configured LLM providers failed.") from last_error

    def generate(self, task: str, **kwargs: Any) -> str:
        """Generic task entry point for orchestration-friendly callers."""
        if task == "query_rewrite":
            return self.rewrite_query(kwargs["query"])
        if task == "rag_answer":
            return self.generate_answer(
                query=kwargs["query"],
                context=kwargs["context"],
                chat_history=kwargs.get("chat_history"),
            )
        raise ValueError(f"Unsupported LLM task: {task}")

    def _route_for(self, task: str) -> ModelRoute:
        if task == "query_rewrite":
            return ModelRoute(
                provider=settings.query_rewrite_provider,
                model=settings.query_rewrite_model,
                temperature=0.0,
                max_tokens=256,
            )

        provider = settings.answer_generation_provider
        model = settings.answer_generation_model or None
        if self.routing_mode == "cost_optimized":
            provider = provider or "groq"
        elif self.routing_mode == "quality" and not settings.answer_generation_model:
            provider = provider or "anthropic"
        return ModelRoute(provider=provider, model=model)

    def _providers_with_fallbacks(self, primary: str) -> list[str]:
        providers = [primary]
        providers.extend(provider for provider in self.fallback_providers if provider != primary)
        return providers