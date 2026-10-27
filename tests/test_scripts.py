"""The command-line scripts: they must be safe by default and tell the truth about what they did."""

from __future__ import annotations

import importlib.util
import json
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import jwt
import pytest

from config.settings import settings
from online_pipeline.audit.logger import AuditLog, set_audit
from vector_db.client import VectorDBClient

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
VECTOR = [1.0, 0.0, 0.0, 0.0]


def load(name: str):
    spec = importlib.util.spec_from_file_location(f"script_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(module, *argv: str) -> int:
    with patch.object(sys, "argv", [module.__file__, *argv]):
        return module.main()


@pytest.fixture
def store():
    client = VectorDBClient(url=":memory:", collection=f"s{time.monotonic_ns()}", dimension=4)
    client.ensure_collection()
    return client


def point(store, doc_id, **payload):
    store.upsert([{"id": str(uuid.uuid4()), "vector": VECTOR, "payload": {"doc_id": doc_id, **payload}, "text": f"t {doc_id}"}])


FULL = {"tenant_id": "eq", "owner_id": "u", "acl_groups": [], "acl_public": False, "classification": 1}


class TestBackfillScript:
    def test_dry_run_by_default_then_apply(self, store, capsys, audit_log):
        module = load("backfill_acl")
        point(store, "old")
        point(store, "new", **FULL)
        with patch.object(module, "VectorDBClient", return_value=store):
            assert run(module, "--tenant", "legacy", "--owner", "root") == 0
            assert "DRY RUN" in capsys.readouterr().out
            assert not any(p.get("tenant_id") == "legacy" for _, p in store.iter_payloads())
            run(module, "--tenant", "legacy", "--owner", "root", "--apply")
        out = capsys.readouterr().out
        assert "APPLIED" in out and "1 updated" in out
        assert {p["doc_id"]: p["tenant_id"] for _, p in store.iter_payloads()} == {"old": "legacy", "new": "eq"}
        event = [json.loads(l) for l in audit_log.path.read_text().splitlines()][-1]
        assert event["action"] == "admin.backfill_acl" and event["details"]["updated"] == 1

    def test_tenant_and_owner_are_required(self):
        module = load("backfill_acl")
        with pytest.raises(SystemExit):
            run(module, "--tenant", "legacy")


class TestPurgeScript:
    def seed(self, store):
        old = (datetime.now(tz=timezone.utc) - timedelta(days=400)).isoformat()
        point(store, "old", **FULL, ingested_at=old)
        point(store, "fresh", **FULL, ingested_at=datetime.now(tz=timezone.utc).isoformat())
        point(store, "held", **FULL, ingested_at=old, legal_hold=True)

    def test_dry_run_lists_without_deleting(self, store, capsys):
        module = load("purge_expired")
        self.seed(store)
        with patch.object(module, "VectorDBClient", return_value=store), patch.object(settings, "raw_store_enabled", False):
            assert run(module, "--days", "365") == 0
        out = capsys.readouterr().out
        assert "DRY RUN" in out and "would delete old" in out and "held 1" in out
        assert store.get_document_info("eq", "old") is not None

    def test_apply_deletes_only_the_expired_unheld_document(self, store, capsys):
        module = load("purge_expired")
        self.seed(store)
        with patch.object(module, "VectorDBClient", return_value=store), patch.object(settings, "raw_store_enabled", False):
            run(module, "--days", "365", "--apply")
        assert store.get_document_info("eq", "old") is None
        assert store.get_document_info("eq", "fresh") is not None and store.get_document_info("eq", "held") is not None
        assert "deleted old" in capsys.readouterr().out

    def test_without_a_retention_period_it_does_nothing(self, store, capsys):
        module = load("purge_expired")
        self.seed(store)
        with patch.object(module, "VectorDBClient", return_value=store), patch.object(settings, "retention_days", 0):
            assert run(module, "--apply") == 0
        assert "Nothing to do" in capsys.readouterr().out and store.get_document_info("eq", "old") is not None

    def test_the_period_can_come_from_settings(self, store, capsys):
        module = load("purge_expired")
        self.seed(store)
        with patch.object(module, "VectorDBClient", return_value=store), patch.object(settings, "raw_store_enabled", False), \
             patch.object(settings, "retention_days", 365):
            module = load("purge_expired")        # re-read the default from settings
            with patch.object(module, "VectorDBClient", return_value=store):
                run(module, "--apply")
        assert store.get_document_info("eq", "old") is None


class TestAuditVerifyScript:
    def test_reports_ok_then_tampering(self, tmp_path, capsys):
        module = load("verify_audit_chain")
        key = (settings.audit_hmac_key or "dev-audit-key-not-for-production").encode()
        log = AuditLog(tmp_path / "a.jsonl", key, fsync=False)
        for i in range(4):
            log.emit("query", actor="u", n=i)
        with patch.object(sys, "argv", ["x", str(log.path)]):
            assert module.main() == 0
        assert "OK: 4 events" in capsys.readouterr().out
        rows = log.path.read_text().splitlines()
        event = json.loads(rows[1])
        event["actor"] = "mallory"
        rows[1] = json.dumps(event, sort_keys=True)
        log.path.write_text("\n".join(rows) + "\n")
        with patch.object(sys, "argv", ["x", str(log.path)]):
            assert module.main() == 1
        assert "TAMPERED at seq 2" in capsys.readouterr().out


class TestDevTokenScript:
    def test_refuses_outside_dev_token_mode(self, capsys):
        module = load("make_dev_token")
        with patch.object(settings, "app_env", "prod"), patch.object(settings, "auth_mode", "okta"):
            assert run(module, "--tenant", "t") == 1
        assert "Refusing" in capsys.readouterr().err

    def test_refuses_a_short_secret(self, capsys):
        module = load("make_dev_token")
        with patch.object(settings, "app_env", "dev"), patch.object(settings, "auth_mode", "dev_token"), \
             patch.object(settings, "dev_jwt_secret", "short"):
            assert run(module, "--tenant", "t") == 1

    def test_mints_a_token_the_validator_accepts(self, capsys):
        from online_pipeline.auth.tokens import TokenValidator

        module = load("make_dev_token")
        with patch.object(settings, "app_env", "dev"), patch.object(settings, "auth_mode", "dev_token"), \
             patch.object(settings, "dev_jwt_secret", "s" * 40):
            assert run(module, "--tenant", "acme", "--groups", "fx,rates", "--clearance", "2",
                       "--scopes", "rag:query,rag:ingest", "--user", "alice") == 0
            token = capsys.readouterr().out.strip()
            principal = TokenValidator(settings).validate(token)
        assert (principal.user_id, principal.tenant_id, principal.clearance) == ("alice", "acme", 2)
        assert principal.groups == {"fx", "rates"} and principal.scopes == {"rag:query", "rag:ingest"}
