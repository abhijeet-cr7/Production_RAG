"""API extractor: fetches and flattens JSON payloads from REST APIs."""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class APIExtractor:
    """Extract text content from a REST API endpoint.

    The extractor performs a GET request and converts the JSON
    response to a normalised text representation.
    """

    def __init__(
        self,
        headers: dict[str, str] | None = None,
        timeout: int = 15,
    ) -> None:
        self.headers = headers or {}
        self.timeout = timeout

    def extract(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        text_fields: list[str] | None = None,
    ) -> dict[str, Any]:
        """Fetch JSON from *url* and return text + metadata.

        Args:
            url: API endpoint URL.
            params: Optional query parameters.
            text_fields: If provided, only these top-level JSON fields are
                used to build the text; otherwise the full JSON is serialised.

        Returns:
            A dict with keys ``text`` (str) and ``metadata`` (dict).
        """
        try:
            import httpx
        except ImportError as exc:
            raise ImportError("httpx is required for API extraction.") from exc

        with httpx.Client(timeout=self.timeout) as client:
            response = client.get(url, params=params, headers=self.headers)
            response.raise_for_status()

        payload: Any = response.json()

        if text_fields and isinstance(payload, dict):
            selected = {k: payload[k] for k in text_fields if k in payload}
            text = json.dumps(selected, indent=2, ensure_ascii=False)
        else:
            text = json.dumps(payload, indent=2, ensure_ascii=False)

        return {
            "text": text,
            "metadata": {
                "source": url,
                "file_type": "api",
                "content_type": response.headers.get("content-type", ""),
            },
        }
