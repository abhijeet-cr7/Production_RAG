"""Audit log: hash chain integrity, tamper detection, concurrency, privacy."""

from __future__ import annotations

import json
import multiprocessing
import threading

import pytest

from config.settings import settings
from online_pipeline.audit.logger import (
    GENESIS_HASH,
    AuditError,
    AuditLog,
    record,
    verify_chain,
)
from online_pipeline.auth.principal import Principal

KEY = b"test-audit-key-0123456789abcdef"


@pytest.fixture
def log(tmp_path):
    return AuditLog(tmp_path / "chain.jsonl", KEY, fsync=False)


def fill(log, n=5):
    for i in range(n):
        log.emit("query", actor="u1", tenant="t1", resource={"doc_id": f"d{i}"}, n=i)


def lines(log):
    return log.path.read_text().splitlines()


def write_lines(log, new_lines):
    log.path.write_text("\n".join(new_lines) + "\n")


class TestChain:
    def test_chain_links_and_verifies(self, log):
        fill(log, 5)
        events = [json.loads(line) for line in lines(log)]
        assert [e["seq"] for e in events] == [1, 2, 3, 4, 5]
        assert events[0]["prev_hash"] == GENESIS_HASH
        assert all(events[i]["prev_hash"] == events[i - 1]["hash"] for i in range(1, 5))
        assert verify_chain(log.path, KEY).ok and verify_chain(log.path, KEY).events == 5

    def test_empty_or_missing_log_verifies(self, log):
        assert verify_chain(log.path, KEY).ok

    def test_chain_continues_across_instances(self, log):
        fill(log, 2)
        AuditLog(log.path, KEY, fsync=False).emit("query", actor="u2")
        assert verify_chain(log.path, KEY) == verify_chain(log.path, KEY)
        assert verify_chain(log.path, KEY).events == 3

    def test_principal_actor_fills_tenant(self, log):
        event = log.emit("query", actor=Principal(user_id="u9", tenant_id="acme"))
        assert (event["actor"], event["tenant"]) == ("u9", "acme")


class TestTamperDetection:
    def test_edited_field_is_detected_at_that_event(self, log):
        fill(log, 5)
        rows = lines(log)
        event = json.loads(rows[2])
        event["decision"] = "allow"
        event["actor"] = "someone-else"
        rows[2] = json.dumps(event, sort_keys=True)
        write_lines(log, rows)
        result = verify_chain(log.path, KEY)
        assert (result.ok, result.bad_seq) == (False, 3)

    def test_deleted_event_is_detected(self, log):
        fill(log, 5)
        rows = lines(log)
        del rows[2]
        write_lines(log, rows)
        result = verify_chain(log.path, KEY)
        assert not result.ok and result.bad_seq == 3

    def test_reordered_events_are_detected(self, log):
        fill(log, 4)
        rows = lines(log)
        rows[1], rows[2] = rows[2], rows[1]
        write_lines(log, rows)
        assert not verify_chain(log.path, KEY).ok

    def test_forged_event_without_the_key_is_detected(self, log):
        fill(log, 3)
        forger = AuditLog(log.path.parent / "forged.jsonl", b"attacker-key-attacker-key-attacker")
        fill(forger, 3)
        # Attacker replaces the whole log with a self-consistent chain made with another key.
        log.path.write_bytes(forger.path.read_bytes())
        result = verify_chain(log.path, KEY)
        assert (result.ok, result.bad_seq) == (False, 1)

    def test_appended_forged_line_is_detected(self, log):
        fill(log, 2)
        last = json.loads(lines(log)[-1])
        forged = dict(last, seq=3, decision="allow", prev_hash=last["hash"], hash="0" * 64)
        with open(log.path, "a") as handle:
            handle.write(json.dumps(forged) + "\n")
        assert not verify_chain(log.path, KEY).ok

    def test_wrong_key_fails_immediately(self, log):
        fill(log, 2)
        assert verify_chain(log.path, b"some-other-key-some-other-key-xx").bad_seq == 1

    def test_garbage_line_is_detected(self, log):
        fill(log, 2)
        with open(log.path, "a") as handle:
            handle.write("not json\n")
        result = verify_chain(log.path, KEY)
        assert not result.ok and result.reason == "unparseable line"

    def test_tail_truncation_is_a_documented_limit(self, log):
        # Dropping the newest events leaves a valid prefix. Detecting this needs the
        # latest hash to be anchored externally; the module docstring says so.
        fill(log, 5)
        write_lines(log, lines(log)[:3])
        assert verify_chain(log.path, KEY).ok


class TestAppendSafety:
    def test_corrupt_tail_refuses_to_append(self, log):
        fill(log, 2)
        with open(log.path, "a") as handle:
            handle.write('{"seq": 3, "half a li')
        with pytest.raises(AuditError):
            log.emit("query")

    def test_event_larger_than_the_tail_window_still_chains(self, log):
        log.emit("query", blob="x" * 200_000)
        log.emit("query")
        assert verify_chain(log.path, KEY).events == 2

    def test_unwritable_location_raises(self, tmp_path):
        blocker = tmp_path / "file"
        blocker.write_text("x")
        with pytest.raises(OSError):
            AuditLog(blocker / "sub" / "audit.jsonl", KEY)

    def test_threads_share_one_valid_chain(self, log):
        def work():
            for _ in range(25):
                log.emit("query", actor="u")

        threads = [threading.Thread(target=work) for _ in range(8)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        result = verify_chain(log.path, KEY)
        assert result.ok and result.events == 200

    def test_processes_share_one_valid_chain(self, tmp_path):
        path = str(tmp_path / "mp.jsonl")
        ctx = multiprocessing.get_context("spawn")
        procs = [ctx.Process(target=_worker, args=(path, KEY.decode(), 25)) for _ in range(4)]
        [p.start() for p in procs]
        [p.join(timeout=60) for p in procs]
        assert all(p.exitcode == 0 for p in procs)
        result = verify_chain(path, KEY)
        assert result.ok and result.events == 100


def _worker(path: str, key: str, count: int) -> None:
    log = AuditLog(path, key.encode(), fsync=False)
    for _ in range(count):
        log.emit("query", actor="u")


class TestPrivacyAndPolicy:
    def test_fingerprint_is_keyed_deterministic_and_opaque(self, log, tmp_path):
        other = AuditLog(tmp_path / "o.jsonl", b"another-key-another-key-another-k")
        text = "confidential merger terms"
        assert log.fingerprint(text) == log.fingerprint(text)
        assert log.fingerprint(text) != other.fingerprint(text)
        assert text not in log.fingerprint(text) and len(log.fingerprint(text)) == 64

    def test_record_fails_closed_by_default(self, monkeypatch):
        from online_pipeline.audit import logger as mod

        class Broken:
            def emit(self, *a, **k):
                raise AuditError("nope")

        mod.set_audit(Broken())
        monkeypatch.setattr(settings, "audit_fail_closed", True)
        with pytest.raises(AuditError):
            record("query")

    def test_record_can_fail_open_when_configured(self, monkeypatch):
        from online_pipeline.audit import logger as mod

        class Broken:
            def emit(self, *a, **k):
                raise AuditError("nope")

        mod.set_audit(Broken())
        monkeypatch.setattr(settings, "audit_fail_closed", False)
        assert record("query") is None
