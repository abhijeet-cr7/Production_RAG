"""Shared fixtures: every test gets an isolated audit log."""

from __future__ import annotations

import pytest

from online_pipeline.audit.logger import AuditLog, set_audit

AUDIT_KEY = b"test-audit-key-0123456789abcdef"


@pytest.fixture(autouse=True)
def verification_off_by_default(monkeypatch):
    """Most gateway tests stub the model with a bare "answer"; verification has its own tests."""
    from config.settings import settings

    monkeypatch.setattr(settings, "answer_verification_enabled", False)


@pytest.fixture(autouse=True)
def audit_log(tmp_path):
    log = AuditLog(tmp_path / "audit.jsonl", AUDIT_KEY, fsync=False)
    set_audit(log)
    yield log
    set_audit(None)
