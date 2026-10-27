"""Document ACLs: what a chunk requires, who may read it, and how to enforce it.

A chunk is readable by a principal iff ALL hold:
  1. chunk.tenant_id == principal.tenant_id
  2. chunk.classification <= principal.clearance
  3. chunk.owner_id == principal.user_id, or chunk.acl_public, or the principal
     shares at least one group with chunk.acl_groups

A chunk with missing or malformed ACL fields is denied (default deny).

Enforcement is two-layer: :func:`build_qdrant_filter` pre-filters inside the
vector store, and :func:`is_authorized` re-checks in Python on every candidate
list, so a filter bug or a new retriever cannot leak on its own.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Iterable

from online_pipeline.auth.policy import TenantPolicy
from online_pipeline.auth.principal import MAX_CLEARANCE, MIN_CLEARANCE, Principal

DEFAULT_CLASSIFICATION = 1  # internal
MAX_GROUPS_PER_DOCUMENT = 50
MAX_GROUP_NAME_LENGTH = 128

ACL_FIELDS = ("tenant_id", "owner_id", "acl_groups", "acl_public", "classification")


class ACLError(ValueError):
    """The requested ACL is not something this principal may grant."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code


@dataclass(frozen=True)
class DocumentACL:
    tenant_id: str
    owner_id: str
    acl_groups: tuple[str, ...]
    acl_public: bool
    classification: int

    def as_payload(self) -> dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "owner_id": self.owner_id,
            "acl_groups": list(self.acl_groups),
            "acl_public": self.acl_public,
            "classification": self.classification,
        }


def build_document_acl(
    principal: Principal,
    policy: TenantPolicy,
    *,
    acl_groups: Iterable[str] | None = None,
    acl_public: bool = False,
    classification: int | None = None,
) -> DocumentACL:
    """Derive an ACL from the token and the (validated) request.

    Tenant and owner always come from the token. A caller can only grant groups
    they belong to, and cannot classify above their own clearance or above the
    tenant's ceiling. With no groups and ``acl_public`` false the document is
    private to its owner.
    """
    groups = _normalise_groups(acl_groups)
    not_held = [g for g in groups if g not in principal.groups]
    if not_held:
        raise ACLError("group_not_held", "You can only share with groups you belong to.")

    ceiling = min(principal.clearance, policy.max_classification)
    if classification is None:
        level = min(DEFAULT_CLASSIFICATION, ceiling)
    else:
        if isinstance(classification, bool) or not MIN_CLEARANCE <= classification <= MAX_CLEARANCE:
            raise ACLError("bad_classification", "classification must be an integer from 0 to 3.")
        if classification > principal.clearance:
            raise ACLError("above_clearance", "You cannot classify above your own clearance.")
        if classification > policy.max_classification:
            raise ACLError("above_tenant_max", "Classification exceeds your tenant's maximum.")
        level = classification

    return DocumentACL(
        tenant_id=principal.tenant_id,
        owner_id=principal.user_id,
        acl_groups=tuple(groups),
        acl_public=bool(acl_public),
        classification=level,
    )


def _normalise_groups(groups: Iterable[str] | None) -> list[str]:
    seen: dict[str, None] = {}
    for raw in groups or ():
        name = str(raw).strip()
        if not name:
            continue
        if len(name) > MAX_GROUP_NAME_LENGTH:
            raise ACLError("bad_group", "Group name too long.")
        seen[name] = None
    if len(seen) > MAX_GROUPS_PER_DOCUMENT:
        raise ACLError("too_many_groups", "Too many groups.")
    return list(seen)


def document_id(tenant_id: str, owner_id: str, text: str) -> str:
    """Stable id scoped to tenant AND owner as well as content.

    Content alone would let two tenants (or two users) who upload identical
    bytes overwrite each other's ACL. Length-prefixing keeps ("ab","c") and
    ("a","bc") from colliding.
    """
    digest = hashlib.sha256()
    for part in (tenant_id, owner_id):
        raw = part.encode()
        digest.update(len(raw).to_bytes(4, "big"))
        digest.update(raw)
    digest.update(text.encode())
    return digest.hexdigest()


def has_acl(payload: dict[str, Any]) -> bool:
    """True when every ACL field is present (used to refuse un-stamped writes)."""
    return all(field in payload for field in ACL_FIELDS)


def is_authorized(payload: dict[str, Any], principal: Principal) -> bool:
    """Python mirror of :func:`build_qdrant_filter`; the two must always agree."""
    tenant = payload.get("tenant_id")
    if not isinstance(tenant, str) or tenant != principal.tenant_id:
        return False
    if payload.get("quarantined") is True:  # held back by the injection screen
        return False

    level = payload.get("classification")
    if isinstance(level, bool) or not isinstance(level, int) or level > principal.clearance:
        return False

    owner = payload.get("owner_id")
    if isinstance(owner, str) and owner and owner == principal.user_id:
        return True
    if payload.get("acl_public") is True:
        return True
    groups = payload.get("acl_groups")
    if isinstance(groups, list):
        return any(isinstance(g, str) and g in principal.groups for g in groups)
    return False


def authorize(
    chunks: list[dict[str, Any]], principal: Principal
) -> tuple[list[dict[str, Any]], int]:
    """Keep only chunks the principal may read. Returns (allowed, dropped_count)."""
    allowed = [c for c in chunks if is_authorized(c.get("metadata") or {}, principal)]
    return allowed, len(chunks) - len(allowed)


def build_qdrant_filter(principal: Principal, source_in: list[str] | None = None) -> Any:
    """Server-side pre-filter for *principal*. ``source_in`` only narrows further."""
    from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue, Range

    access = [
        FieldCondition(key="owner_id", match=MatchValue(value=principal.user_id)),
        FieldCondition(key="acl_public", match=MatchValue(value=True)),
    ]
    if principal.groups:
        access.append(FieldCondition(key="acl_groups", match=MatchAny(any=sorted(principal.groups))))

    must: list[Any] = [
        FieldCondition(key="tenant_id", match=MatchValue(value=principal.tenant_id)),
        FieldCondition(key="classification", range=Range(lte=principal.clearance)),
        Filter(should=access),
    ]
    if source_in:
        must.append(FieldCondition(key="source", match=MatchAny(any=list(source_in))))
    quarantined = FieldCondition(key="quarantined", match=MatchValue(value=True))
    return Filter(must=must, must_not=[quarantined])
