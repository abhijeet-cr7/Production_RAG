"""Metadata enricher: derives and attaches document metadata fields."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any


class MetadataEnricher:
    """Derive and attach additional metadata to a document dict."""

    def enrich(
        self,
        document: dict[str, Any],
        *,
        tenant_id: str | None = None,
        owner_id: str | None = None,
    ) -> dict[str, Any]:
        """Return *document* with an enriched ``metadata`` sub-dict.

        Adds:
            - ``doc_id``: SHA-256 of tenant + owner + text when both are given
              (the API path, so identical uploads from different tenants or
              users never share an id or overwrite each other's ACL); the
              plain content hash otherwise (CLI/batch use)
            - ``char_count``: character count of the text
            - ``word_count``: approximate word count
            - ``ingested_at``: UTC ISO-8601 timestamp
        """
        text: str = document.get("text", "")
        metadata: dict[str, Any] = document.get("metadata", {})

        if tenant_id is not None and owner_id is not None:
            from online_pipeline.auth.acl import document_id

            metadata["doc_id"] = document_id(tenant_id, owner_id, text)
        else:
            metadata["doc_id"] = hashlib.sha256(text.encode()).hexdigest()
        metadata["char_count"] = len(text)
        metadata["word_count"] = len(text.split())
        metadata["ingested_at"] = datetime.now(tz=timezone.utc).isoformat()

        return {**document, "metadata": metadata}
