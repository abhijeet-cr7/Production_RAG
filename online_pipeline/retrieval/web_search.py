"""Tavily web search: live results for questions the indexed corpus cannot answer."""

from __future__ import annotations

import logging
from typing import Any

from config.settings import settings

logger = logging.getLogger(__name__)


class TavilyWebSearch:
    """Query the Tavily API and normalise hits into retriever-shaped dicts.

    Results use the same ``{"text", "metadata"}`` shape as vector/BM25 hits so
    they can be reranked and turned into context alongside indexed chunks.
    """

    def __init__(
        self,
        api_key: str | None = None,
        max_results: int | None = None,
        search_depth: str | None = None,
    ) -> None:
        self.api_key = settings.tavily_api_key if api_key is None else api_key
        self.max_results = max_results or settings.web_search_max_results
        self.search_depth = search_depth or settings.tavily_search_depth
        self._client = self._connect()

    @property
    def available(self) -> bool:
        """True when a Tavily client was successfully constructed."""
        return self._client is not None

    def search(self, query: str, max_results: int | None = None) -> list[dict[str, Any]]:
        """Return web results for *query*, or an empty list on any failure."""
        if self._client is None or not query.strip():
            return []

        try:
            response = self._client.search(
                query=query,
                max_results=max_results or self.max_results,
                search_depth=self.search_depth,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Tavily search failed: %s", exc)
            return []

        return self._normalise(response)

    # ── private helpers ──────────────────────────────────────────────────────

    def _connect(self) -> Any | None:
        if not self.api_key:
            logger.info("TAVILY_API_KEY not set; web search disabled.")
            return None
        try:
            from tavily import TavilyClient
        except ImportError:
            logger.warning("tavily-python is not installed; web search disabled.")
            return None
        try:
            return TavilyClient(api_key=self.api_key)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Tavily client initialisation failed: %s", exc)
            return None

    @staticmethod
    def _normalise(response: Any) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for item in (response or {}).get("results", []):
            text = (item.get("content") or "").strip()
            if not text:
                continue
            url = item.get("url", "")
            results.append({
                "text": text,
                "metadata": {
                    "source": url,
                    "title": item.get("title", ""),
                    "file_type": "web_search",
                    "origin": "tavily",
                    "doc_id": url or text[:64],
                },
                "web_score": float(item.get("score") or 0.0),
            })
        return results
