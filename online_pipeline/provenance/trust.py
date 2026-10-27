"""Provenance: where a document came from, and how far to trust it.

Everything here is computed on the server. A client can supply an ``as_of`` date,
an ``expected_sha256`` or a ``supersedes`` reference, but never a trust tier,
publisher or origin.

Trust tiers
-----------
``verified_primary``  fetched over https from a publisher on the server-side
                      allow-list (regulators, issuers), e.g. sec.gov
``internal``          uploaded by an authenticated user of the tenant
``unverified``        any other web/API source, and web-search results from
                      hosts that are not on the verified list
"""

from __future__ import annotations

import hashlib
import hmac
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlsplit

VERIFIED_PRIMARY = "verified_primary"
INTERNAL = "internal"
UNVERIFIED = "unverified"
TRUST_TIERS = (VERIFIED_PRIMARY, INTERNAL, UNVERIFIED)

# What a retrieved chunk may reveal to a reader: provenance and location, but
# never ACL internals (owner, groups, tenant) or processing details.
PUBLIC_SOURCE_FIELDS = (
    "doc_id", "chunk_index", "source", "title", "page_count", "page",
    "section_path", "block_kind", "table_id", "caption",
    "source_type", "origin", "publisher", "trust_tier", "integrity_verified",
    "content_sha256", "fetched_at", "ingested_at", "as_of", "fiscal_period",
    "external_id", "file_type", "rerank_score", "web_score",
)

_EDGAR_PATH = re.compile(r"/Archives/edgar/data/(\d+)/(\d{18})(?:/|$)", re.I)


class ProvenanceError(ValueError):
    """A client-supplied provenance field is invalid."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code


def host_of(url: str) -> str:
    try:
        return (urlsplit(url).hostname or "").lower().rstrip(".")
    except ValueError:
        return ""


def trust_tier_for_url(url: str, verified_publishers: Iterable[str]) -> str:
    """Tier for content fetched from *url*: verified only for an https allow-listed host."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return UNVERIFIED
    host = (parts.hostname or "").lower().rstrip(".")
    if parts.scheme != "https" or not host:
        return UNVERIFIED
    for publisher in verified_publishers:
        publisher = publisher.strip().lower().rstrip(".")
        if publisher and (host == publisher or host.endswith("." + publisher)):
            return VERIFIED_PRIMARY
    return UNVERIFIED


def edgar_accession(url: str) -> str | None:
    """SEC accession number (0000320193-23-000106) from an EDGAR archive URL."""
    match = _EDGAR_PATH.search(urlsplit(url).path)
    if not match:
        return None
    digits = match.group(2)
    return f"{digits[:10]}-{digits[10:12]}-{digits[12:]}"


def parse_as_of(value: str | None) -> str | None:
    """Validate an ISO date (YYYY-MM-DD) and return it normalised; None if empty."""
    if value is None or not str(value).strip():
        return None
    try:
        parsed = date.fromisoformat(str(value).strip())
    except ValueError as exc:
        raise ProvenanceError("bad_as_of", "as_of must be an ISO date, YYYY-MM-DD.") from exc
    if parsed.year < 1900 or parsed > date.today().replace(year=date.today().year + 1):
        raise ProvenanceError("bad_as_of", "as_of is outside a plausible range.")
    return parsed.isoformat()


def normalise_expected_hash(expected: str | None) -> str | None:
    """Lower-cased 64-hex hash, None when absent; raises on a malformed value."""
    if expected is None or not str(expected).strip():
        return None
    candidate = str(expected).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", candidate):
        raise ProvenanceError("bad_expected_sha256", "expected_sha256 must be 64 hex characters.")
    return candidate


def check_expected_hash(expected: str | None, actual: str) -> bool:
    """True when no hash was declared or it matches. Constant-time comparison."""
    if expected is None or not str(expected).strip():
        return True
    candidate = str(expected).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", candidate):
        raise ProvenanceError("bad_expected_sha256", "expected_sha256 must be 64 hex characters.")
    return hmac.compare_digest(candidate, actual)


def sha256_file(path: str | Path, chunk: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class Provenance:
    source_type: str          # "upload" | "url" | "api"
    origin: str               # display file name, or final URL
    content_sha256: str
    trust_tier: str
    publisher: str
    ingested_by: str
    fetched_at: str | None = None
    external_id: str | None = None
    as_of: str | None = None
    integrity_verified: bool = False
    supersedes: str | None = None

    def as_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "source_type": self.source_type,
            "origin": self.origin,
            "content_sha256": self.content_sha256,
            "trust_tier": self.trust_tier,
            "publisher": self.publisher,
            "ingested_by": self.ingested_by,
            "integrity_verified": self.integrity_verified,
            "superseded": False,
        }
        for key in ("fetched_at", "external_id", "as_of", "supersedes"):
            value = getattr(self, key)
            if value:
                payload[key] = value
        return payload


def for_upload(
    *, filename: str, content_sha256: str, user_id: str,
    as_of: str | None, integrity_verified: bool, supersedes: str | None,
) -> Provenance:
    return Provenance(
        source_type="upload", origin=filename, content_sha256=content_sha256,
        trust_tier=INTERNAL, publisher="upload", ingested_by=user_id,
        as_of=as_of, integrity_verified=integrity_verified, supersedes=supersedes,
    )


def for_remote(
    *, source_type: str, final_url: str, content_sha256: str, fetched_at: str, user_id: str,
    verified_publishers: Iterable[str], as_of: str | None, integrity_verified: bool,
    supersedes: str | None,
) -> Provenance:
    return Provenance(
        source_type=source_type, origin=final_url, content_sha256=content_sha256,
        trust_tier=trust_tier_for_url(final_url, verified_publishers),
        publisher=host_of(final_url), ingested_by=user_id, fetched_at=fetched_at,
        external_id=edgar_accession(final_url), as_of=as_of,
        integrity_verified=integrity_verified, supersedes=supersedes,
    )


def web_result_provenance(url: str, verified_publishers: Iterable[str]) -> dict[str, Any]:
    """Provenance fields for a live web-search hit (never trusted beyond its host)."""
    return {
        "source_type": "web_search",
        "origin": url,
        "publisher": host_of(url),
        "trust_tier": trust_tier_for_url(url, verified_publishers),
        "fetched_at": datetime.now(tz=timezone.utc).isoformat(),
    }


def public_source(metadata: dict[str, Any]) -> dict[str, Any]:
    """The subset of a chunk's metadata that is safe to return to a reader."""
    return {k: metadata[k] for k in PUBLIC_SOURCE_FIELDS if k in metadata}


def trust_allows(metadata: dict[str, Any], require_verified: bool) -> bool:
    """Whether a chunk may feed an answer under the tenant's source-trust rule.

    A chunk with no tier is treated as unverified (default deny).
    """
    if not require_verified:
        return True
    return metadata.get("trust_tier") in (VERIFIED_PRIMARY, INTERNAL)
