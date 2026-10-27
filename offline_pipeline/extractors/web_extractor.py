"""Web extractor: fetch a page through the SSRF-safe fetcher, then parse it in the sandbox."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from offline_pipeline.safety.errors import UnsafeContent
from offline_pipeline.safety.net import FetchResult, SafeFetcher
from offline_pipeline.safety.sandbox import run_worker

# What /ingest/url will accept: pages (incl. inline-XBRL filings), PDFs, XBRL
# instance documents and plain text. JSON is the API route's business.
URL_CONTENT_TYPES = {
    "text/html": "html",
    "application/xhtml+xml": "html",
    "application/pdf": "pdf",
    "application/xml": "xml",
    "text/xml": "xml",
    "text/plain": "text",
}

_SANDBOX_KIND = {"html": ("html", ".html"), "pdf": ("pdf", ".pdf"), "xml": ("xbrl", ".xml")}


class WebExtractor:
    """Fetch text, structure and metadata from a URL, never touching anything unsafe.

    Downloaded pages and PDFs are exactly as untrusted as uploads, so they are
    parsed by the same resource-limited sandbox worker.

    Args:
        fetcher: A :class:`SafeFetcher` already bound to the tenant allow-list.
        parser_limits: Keyword limits forwarded to the sandbox (timeout, memory, ...).
    """

    def __init__(self, fetcher: SafeFetcher, parser_limits: dict[str, Any] | None = None) -> None:
        self.fetcher = fetcher
        self.parser_limits = parser_limits or {}

    def extract(self, url: str) -> dict[str, Any]:
        """Return ``{"text", "metadata", "blocks"?, "raw"}`` for *url*, or raise :class:`UnsafeContent`."""
        fetched = self.fetcher.fetch(url.strip())
        if fetched.family in _SANDBOX_KIND:
            parsed = self._parse(fetched)
        elif fetched.family == "text":
            parsed = {"text": fetched.body.decode("utf-8", errors="replace"), "metadata": {}}
        else:
            raise UnsafeContent("content_type_blocked", "This content type cannot be ingested as a page.")

        extra = dict(parsed.get("metadata", {}))
        extra.pop("file_type", None)
        metadata = {
            "source": fetched.final_url,
            "file_type": {"html": "web", "pdf": "pdf", "xml": "xbrl"}.get(fetched.family, "web"),
            "content_type": fetched.content_type,
            "content_sha256": fetched.sha256,
            "final_url": fetched.final_url,
            "fetched_at": fetched.fetched_at,
            "redirect_count": len(fetched.redirects),
            **extra,
        }
        doc: dict[str, Any] = {"text": parsed["text"], "metadata": metadata, "raw": fetched.body}
        if parsed.get("blocks"):
            doc["blocks"] = parsed["blocks"]
        return doc

    def _parse(self, fetched: FetchResult) -> dict[str, Any]:
        kind, suffix = _SANDBOX_KIND[fetched.family]
        fd, tmp = tempfile.mkstemp(suffix=suffix)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(fetched.body)
            return run_worker(kind, Path(tmp), **self.parser_limits)
        finally:
            os.unlink(tmp)
