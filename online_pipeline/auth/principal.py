"""Authenticated caller identity and the scopes the API understands."""

from __future__ import annotations

from dataclasses import dataclass

SCOPE_QUERY = "rag:query"
SCOPE_INGEST = "rag:ingest"
SCOPE_WEB = "rag:web"
SCOPE_ADMIN = "rag:admin"
SCOPE_AUDIT = "rag:audit"

ALL_SCOPES = frozenset({SCOPE_QUERY, SCOPE_INGEST, SCOPE_WEB, SCOPE_ADMIN, SCOPE_AUDIT})

# 0 public · 1 internal · 2 confidential · 3 restricted
MIN_CLEARANCE = 0
MAX_CLEARANCE = 3


@dataclass(frozen=True)
class Principal:
    """Who is calling, derived only from a validated token."""

    user_id: str
    tenant_id: str
    groups: frozenset[str] = frozenset()
    clearance: int = MIN_CLEARANCE
    scopes: frozenset[str] = frozenset()

    def has_scope(self, scope: str) -> bool:
        return scope in self.scopes
