"""Tamper-evident audit log: append-only JSONL with a keyed hash chain.

Every event stores ``prev_hash`` and ``hash = HMAC(key, prev_hash + body)``, so
editing, removing or reordering an event breaks every later link, and nobody
without the key can forge a valid chain. Events hold ids and keyed fingerprints,
never document text, queries or answers.

Limits: truncating the *tail* of the file is not detectable from the file alone;
anchor the latest hash externally (periodic export) if that matters to you.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:  # POSIX advisory locking so several worker processes share one chain.
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]

from config.settings import settings
from online_pipeline.auth.principal import Principal

logger = logging.getLogger(__name__)

GENESIS_HASH = "0" * 64
_TAIL_CHUNK = 65536


class AuditError(RuntimeError):
    """An audit event could not be written (or the log is unreadable)."""


def _canonical(body: dict[str, Any]) -> bytes:
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _chain_hash(key: bytes, prev_hash: str, body: dict[str, Any]) -> str:
    return hmac.new(key, prev_hash.encode() + b"\n" + _canonical(body), hashlib.sha256).hexdigest()


class AuditLog:
    def __init__(self, path: str | Path, key: bytes, fsync: bool = True) -> None:
        self.path = Path(path)
        self._key = key
        self._fsync = fsync
        self._thread_lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def fingerprint(self, text: str) -> str:
        """Keyed hash of *text*: lets auditors match a query without storing it."""
        return hmac.new(self._key, text.encode(), hashlib.sha256).hexdigest()

    def emit(
        self,
        action: str,
        *,
        actor: Principal | str | None = None,
        tenant: str | None = None,
        resource: Any = None,
        decision: str = "allow",
        reason: str | None = None,
        request_id: str | None = None,
        **details: Any,
    ) -> dict[str, Any]:
        if isinstance(actor, Principal):
            tenant = tenant or actor.tenant_id
            actor = actor.user_id
        try:
            with self._thread_lock, open(self.path, "a+b") as handle:
                if fcntl is not None:
                    fcntl.flock(handle, fcntl.LOCK_EX)
                try:
                    seq, prev_hash = self._tail_state(handle)
                    body = {
                        "seq": seq,
                        "event_id": uuid.uuid4().hex,
                        "ts": datetime.now(tz=timezone.utc).isoformat(),
                        "action": action,
                        "actor": actor,
                        "tenant": tenant,
                        "resource": resource,
                        "decision": decision,
                        "reason": reason,
                        "request_id": request_id,
                        "details": details,
                        "prev_hash": prev_hash,
                    }
                    event = {**body, "hash": _chain_hash(self._key, prev_hash, body)}
                    handle.write(json.dumps(event, sort_keys=True, ensure_ascii=True).encode() + b"\n")
                    handle.flush()
                    if self._fsync:
                        os.fsync(handle.fileno())
                finally:
                    if fcntl is not None:
                        fcntl.flock(handle, fcntl.LOCK_UN)
        except AuditError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise AuditError(f"Could not write audit event: {exc}") from exc
        return event

    @staticmethod
    def _tail_state(handle: Any) -> tuple[int, str]:
        """Return (next seq, previous hash) from the last line of the log."""
        size = handle.seek(0, os.SEEK_END)
        if size == 0:
            return 1, GENESIS_HASH
        window = min(size, _TAIL_CHUNK)
        while True:
            handle.seek(size - window)
            data = handle.read(window)
            lines = data.rstrip(b"\n").split(b"\n")
            # The first element may be a partial line unless we reached the start.
            if len(lines) > 1 or window == size:
                last = lines[-1]
                break
            window = min(size, window * 2)
        try:
            event = json.loads(last)
            return int(event["seq"]) + 1, str(event["hash"])
        except (ValueError, KeyError, TypeError) as exc:
            raise AuditError("Audit log tail is corrupt; refusing to append.") from exc


@dataclass(frozen=True)
class VerifyResult:
    ok: bool
    events: int
    bad_seq: int | None = None
    reason: str | None = None


def verify_chain(path: str | Path, key: bytes) -> VerifyResult:
    """Re-walk the whole log and report the first broken link, if any."""
    prev_hash, expected_seq, count = GENESIS_HASH, 1, 0
    try:
        lines = Path(path).read_bytes().split(b"\n")
    except FileNotFoundError:
        return VerifyResult(True, 0)
    if lines and lines[-1] == b"":
        lines.pop()
    for raw in lines:
        try:
            event = json.loads(raw)
            claimed = event.pop("hash")
            seq = event["seq"]
        except (ValueError, KeyError, TypeError, AttributeError):
            return VerifyResult(False, count, expected_seq, "unparseable line")
        if seq != expected_seq:
            return VerifyResult(False, count, expected_seq, f"expected seq {expected_seq}, found {seq}")
        if event.get("prev_hash") != prev_hash:
            return VerifyResult(False, count, seq, "prev_hash does not match previous event")
        if not hmac.compare_digest(_chain_hash(key, prev_hash, event), str(claimed)):
            return VerifyResult(False, count, seq, "hash mismatch (event altered or wrong key)")
        prev_hash, expected_seq, count = claimed, expected_seq + 1, count + 1
    return VerifyResult(True, count)


def read_events(
    path: str | Path,
    tenant: str,
    *,
    after_seq: int = 0,
    limit: int = 100,
    action: str | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Events of *tenant* with ``seq > after_seq``; returns (events, last seq scanned).

    Streams the file, never loading it whole, and returns only that tenant's rows:
    system events and other tenants' events are not visible through this API.
    """
    events: list[dict[str, Any]] = []
    last = after_seq
    try:
        handle = open(path, "rb")
    except FileNotFoundError:
        return [], after_seq
    with handle:
        for raw in handle:
            try:
                event = json.loads(raw)
                seq = int(event["seq"])
            except (ValueError, KeyError, TypeError):
                continue
            if seq <= after_seq:
                continue
            last = seq
            if event.get("tenant") != tenant or (action and event.get("action") != action):
                continue
            events.append(event)
            if len(events) >= limit:
                break
    return events, last


# ── process-wide instance and helpers ────────────────────────────────────────

_audit: AuditLog | None = None


def get_audit() -> AuditLog:
    global _audit
    if _audit is None:
        key = (settings.audit_hmac_key or "dev-audit-key-not-for-production").encode()
        _audit = AuditLog(settings.audit_log_path, key)
    return _audit


def set_audit(audit: AuditLog | None) -> None:
    global _audit
    _audit = audit


def record(action: str, **kwargs: Any) -> dict[str, Any] | None:
    """Emit an event; with ``audit_fail_closed`` a write failure propagates."""
    try:
        return get_audit().emit(action, **kwargs)
    except AuditError:
        if settings.audit_fail_closed:
            raise
        logger.exception("Audit write failed for %s (fail-open).", action)
        return None


def record_best_effort(action: str, **kwargs: Any) -> None:
    """Emit an event that must never block the response (e.g. a 401)."""
    try:
        get_audit().emit(action, **kwargs)
    except Exception:  # noqa: BLE001
        logger.exception("Audit write failed for %s.", action)
