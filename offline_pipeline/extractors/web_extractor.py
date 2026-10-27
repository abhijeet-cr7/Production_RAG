"""Web extractor: fetch a page through the SSRF-safe fetcher and sanitise it."""

from __future__ import annotations

import os
import tempfile
from typing import Any

from offline_pipeline.safety.errors import UnsafeContent
from offline_pipeline.safety.html_sanitizer import html_to_text, sanitize_html
from offline_pipeline.safety.net import FetchResult, SafeFetcher
from offline_pipeline.safety.sandbox import safe_extract_file

# What /ingest/url will accept: pages, PDFs and plain text. JSON/XML are not pages.
URL_CONTENT_TYPES = {
    "text/html": "html",
    "application/xhtml+xml": "html",
    "application/pdf": "pdf",
    "text/plain": "text",
}


class WebExtractor:
    """Fetch text and metadata from a URL, never touching anything unsafe.

    Args:
        fetcher: A :class:`SafeFetcher` already bound to the tenant allow-list.
    """

    def __init__(self, fetcher: SafeFetcher, parser_limits: dict[str, Any] | None = None) -> None:
        self.fetcher = fetcher
        self.parser_limits = parser_limits or {}

    def extract(self, url: str) -> dict[str, Any]:
        """Return ``{"text", "metadata"}`` for *url*, or raise :class:`UnsafeContent`."""
        fetched = self.fetcher.fetch(url.strip())
        if fetched.family == "html":
            text, extra = self._from_html(fetched)
        elif fetched.family == "pdf":
            text, extra = self._from_pdf(fetched)
        elif fetched.family == "text":
            text, extra = fetched.body.decode("utf-8", errors="replace"), {}
        else:
            raise UnsafeContent("content_type_blocked", "This content type cannot be ingested as a page.")

        metadata = {
            "source": fetched.final_url,
            "file_type": "web" if fetched.family != "pdf" else "pdf",
            "content_type": fetched.content_type,
            "content_sha256": fetched.sha256,
            "final_url": fetched.final_url,
            "fetched_at": fetched.fetched_at,
            "redirect_count": len(fetched.redirects),
            **extra,
        }
        # "raw" is the exact bytes received; the gateway stores and then drops it.
        return {"text": text, "metadata": metadata, "raw": fetched.body}

    @staticmethod
    def _from_html(fetched: FetchResult) -> tuple[str, dict[str, Any]]:
        charset_text = fetched.body.decode("utf-8", errors="replace")
        soup, report = sanitize_html(charset_text)
        title = ""
        if soup.title is not None and soup.title.string:
            title = " ".join(soup.title.string.split())[:300]
        return html_to_text(soup), {
            "title": title,
            "hidden_elements_removed": report.hidden_elements_removed,
            "active_elements_removed": report.active_elements_removed,
        }

    def _from_pdf(self, fetched: FetchResult) -> tuple[str, dict[str, Any]]:
        # Downloaded PDFs are exactly as untrusted as uploads: same sandbox.
        fd, tmp = tempfile.mkstemp(suffix=".pdf")
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(fetched.body)
            doc = safe_extract_file(tmp, ".pdf", **self.parser_limits)
        finally:
            os.unlink(tmp)
        meta = doc["metadata"]
        return doc["text"], {
            "page_count": meta.get("page_count"),
            "hidden_chars_removed": meta.get("hidden_chars_removed", 0),
        }
