"""Metadata extractor: enriches document metadata with derived fields."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any


class MetadataExtractor:
    """Derive and attach additional metadata to a document dict."""

    def enrich(self, document: dict[str, Any]) -> dict[str, Any]:
        """Return *document* with an enriched ``metadata`` sub-dict.

        Adds:
            - ``doc_id``: SHA-256 hash of the text content
            - ``char_count``: character count of the text
            - ``word_count``: approximate word count
            - ``ingested_at``: UTC ISO-8601 timestamp
        """
        text: str = document.get("text", "")
        metadata: dict[str, Any] = document.get("metadata", {})

        metadata["doc_id"] = hashlib.sha256(text.encode()).hexdigest()
        metadata["char_count"] = len(text)
        metadata["word_count"] = len(text.split())
        metadata["ingested_at"] = datetime.now(tz=timezone.utc).isoformat()

        return {**document, "metadata": metadata}
