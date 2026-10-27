"""Document lifecycle operations shared by the API and the maintenance scripts.

Deleting, retention purges and the legacy backfill all live here so the HTTP
endpoints and the command-line scripts can never disagree about what they do
(for example: a document under legal hold is never deleted by either).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from online_pipeline.audit.logger import record
from online_pipeline.auth.acl import ACL_FIELDS, has_acl
from online_pipeline.provenance.rawstore import RawStore


class LegalHold(Exception):
    """The document is under legal hold and cannot be deleted."""


def delete_document_everywhere(
    vector_db: Any, raw_store: RawStore | None, tenant_id: str, doc_id: str
) -> dict[str, Any]:
    """Delete every chunk of *doc_id* and, if nothing else uses it, the original bytes.

    Raises :class:`LegalHold` first, without touching anything, if the document is held.
    """
    info = vector_db.get_document_info(tenant_id, doc_id)
    if info is None:
        return {"chunks": 0, "raw_deleted": False}
    if info.get("legal_hold") is True:
        raise LegalHold(doc_id)

    sha = info.get("content_sha256")
    chunks = vector_db.delete_document(tenant_id, doc_id)
    raw_deleted = False
    if raw_store is not None and sha and vector_db.count_by_hash(tenant_id, sha) == 0:
        raw_deleted = raw_store.delete(tenant_id, sha)
    return {"chunks": chunks, "raw_deleted": raw_deleted}


@dataclass
class PurgeReport:
    examined: int = 0
    expired: int = 0
    held: int = 0
    deleted: int = 0
    doc_ids: list[str] = field(default_factory=list)


def purge_expired(
    vector_db: Any,
    raw_store: RawStore | None,
    *,
    days: int,
    apply: bool = False,
    now: datetime | None = None,
) -> PurgeReport:
    """Remove documents ingested more than *days* ago, skipping those on legal hold.

    Dry-run unless ``apply``. Each deletion is audited, with no document content.
    """
    report = PurgeReport()
    if days <= 0:
        return report
    cutoff = (now or datetime.now(tz=timezone.utc)) - timedelta(days=days)
    for doc in vector_db.list_documents():
        report.examined += 1
        try:
            ingested = datetime.fromisoformat(str(doc.get("ingested_at")))
        except ValueError:
            continue  # no usable timestamp: never guess that something is expired
        if ingested.tzinfo is None:
            ingested = ingested.replace(tzinfo=timezone.utc)
        if ingested >= cutoff:
            continue
        report.expired += 1
        if doc.get("legal_hold"):
            report.held += 1
            continue
        report.doc_ids.append(str(doc["doc_id"]))
        if apply:
            result = delete_document_everywhere(vector_db, raw_store, doc["tenant_id"], doc["doc_id"])
            report.deleted += 1
            record(
                "doc.delete", actor="retention-job", tenant=doc["tenant_id"],
                resource={"doc_id": doc["doc_id"]}, reason=f"retention>{days}d", **result,
            )
    return report


@dataclass
class BackfillReport:
    scanned: int = 0
    needing_acl: int = 0
    updated: int = 0


LEGACY_DEFAULTS = {
    "acl_groups": [], "acl_public": False, "classification": 1,
    "quarantined": False, "superseded": False,
    "trust_tier": "unverified", "source_type": "legacy",
}


def backfill_acl(
    vector_db: Any, *, tenant_id: str, owner_id: str, apply: bool = False, batch: int = 200
) -> BackfillReport:
    """Give pre-ACL points an owner so they are readable by someone (default deny otherwise).

    Points missing any ACL field are assigned to *tenant_id*/*owner_id*, private to the
    owner, internal, and marked ``unverified`` (their origin is unknown). Points that
    already carry a full ACL are never touched, so re-running is safe.
    """
    report = BackfillReport()
    pending: list[tuple[str, dict[str, Any]]] = []

    def flush() -> None:
        nonlocal pending
        if apply and pending:
            vector_db.tag_points([pid for pid, _ in pending], _payload_for(tenant_id, owner_id))
            report.updated += len(pending)
        pending = []

    for point_id, payload in vector_db.iter_payloads():
        report.scanned += 1
        if has_acl(payload):
            continue
        report.needing_acl += 1
        pending.append((point_id, payload))
        if len(pending) >= batch:
            flush()
    flush()
    return report


def _payload_for(tenant_id: str, owner_id: str) -> dict[str, Any]:
    return {**LEGACY_DEFAULTS, "tenant_id": tenant_id, "owner_id": owner_id, "ingested_by": owner_id}
