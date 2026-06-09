"""Query rewriter: rewrites or expands user queries before embedding.

Uses an LLM to produce a more retrieval-friendly version of the
original query (e.g. HyDE, step-back prompting, or simple expansion).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a search query optimiser. "
    "Rewrite the user's question as a single, concise, self-contained "
    "search query that will retrieve the most relevant documents. "
    "Return only the rewritten query, nothing else."
)


class QueryRewriter:
    """Rewrite a raw user query into a retrieval-optimised query.

    Falls back to the original query when the LLM call fails or when
    no API key is configured.
    """

    def __init__(self, model: str = "llama-3.1-8b-instant") -> None:
        self.model = model

    def rewrite(self, query: str) -> str:
        """Return a rewritten version of *query*.

        Args:
            query: Raw user input query.

        Returns:
            Rewritten query string (or the original on failure).
        """
        try:
            return self._call_llm(query)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Query rewrite failed (%s); using original.", exc)
            return query

    def _call_llm(self, query: str) -> str:
        from groq import Groq
        from config.settings import settings

        client = Groq(api_key=settings.groq_api_key)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=0.0,
            max_tokens=256,
        )
        return response.choices[0].message.content.strip()
