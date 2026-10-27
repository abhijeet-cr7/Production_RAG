"""API extractor: fetches and flattens JSON payloads from REST APIs."""

from __future__ import annotations

import json
from typing import Any

from offline_pipeline.safety.errors import UnsafeContent
from offline_pipeline.safety.net import SafeFetcher

# /ingest/api accepts JSON only.
API_CONTENT_TYPES = {"application/json": "json"}
MAX_JSON_CHARS = 5_000_000


class APIExtractor:
    """Fetch a JSON endpoint through the SSRF-safe fetcher and normalise it.

    The caller cannot attach request headers (so no credentials can be forwarded
    or Host spoofed); the fetcher is already bound to the tenant allow-list.
    """

    def __init__(self, fetcher: SafeFetcher) -> None:
        self.fetcher = fetcher

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
        """
        fetched = self.fetcher.fetch(url.strip(), params=params)
        try:
            payload: Any = json.loads(fetched.body)
        except (ValueError, RecursionError) as exc:
            raise UnsafeContent("json_invalid", "Response is not valid, bounded JSON.") from exc

        if text_fields and isinstance(payload, dict):
            payload = {k: payload[k] for k in text_fields if k in payload}
        try:
            text = json.dumps(payload, indent=2, ensure_ascii=False)
        except RecursionError as exc:
            raise UnsafeContent("json_too_deep", "JSON is nested too deeply.") from exc
        if len(text) > MAX_JSON_CHARS:
            raise UnsafeContent("text_too_large", "Extracted text is too large.")

        return {
            "text": text,
            "metadata": {
                "source": fetched.final_url,
                "file_type": "api",
                "content_type": fetched.content_type,
                "content_sha256": fetched.sha256,
                "final_url": fetched.final_url,
                "fetched_at": fetched.fetched_at,
                "redirect_count": len(fetched.redirects),
            },
            "raw": fetched.body,
        }
