"""Per-tenant policy: the ceiling that user scopes can never exceed.

Effective permission is always the AND of the tenant policy, the user's scope
and the request flag. Unknown tenants get the most restrictive policy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import yaml

from online_pipeline.auth.principal import SCOPE_WEB, Principal

logger = logging.getLogger(__name__)

_ANSWER_POLICIES = {"warn", "abstain"}


@dataclass(frozen=True)
class TenantPolicy:
    allow_web_search: bool = False
    allow_url_ingest: bool = False
    url_allowlist: tuple[str, ...] = ()
    max_classification: int = 1
    require_verified_sources: bool = True
    ungrounded_answer_policy: str = "abstain"
    numeric_mismatch_policy: str = "abstain"
    refuse_investment_advice: bool = True
    answer_disclaimer: str = ""


RESTRICTIVE = TenantPolicy()


class PolicyError(ValueError):
    """The tenant policy file is malformed."""


class TenantPolicyStore:
    """Tenant id → :class:`TenantPolicy`, loaded once at startup."""

    def __init__(
        self,
        default: TenantPolicy = RESTRICTIVE,
        tenants: dict[str, TenantPolicy] | None = None,
    ) -> None:
        self._default = default
        self._tenants = dict(tenants or {})

    def for_tenant(self, tenant_id: str) -> TenantPolicy:
        return self._tenants.get(tenant_id, self._default)

    @classmethod
    def load(cls, path: str | Path) -> TenantPolicyStore:
        path = Path(path)
        if not path.exists():
            logger.warning("Tenant policy file %s not found; every tenant is restricted.", path)
            return cls()
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise PolicyError(f"{path}: invalid YAML: {exc}") from exc
        if not isinstance(raw, dict):
            raise PolicyError(f"{path}: top level must be a mapping")

        unknown = set(raw) - {"default", "tenants"}
        if unknown:
            raise PolicyError(f"{path}: unknown top-level keys {sorted(unknown)}")

        default = _build(RESTRICTIVE, raw.get("default") or {}, "default")
        tenants_raw = raw.get("tenants") or {}
        if not isinstance(tenants_raw, dict):
            raise PolicyError(f"{path}: 'tenants' must be a mapping")
        # Tenants inherit from the file's default block, then override.
        tenants = {
            str(name): _build(default, values or {}, f"tenants.{name}")
            for name, values in tenants_raw.items()
        }
        return cls(default=default, tenants=tenants)


def _build(base: TenantPolicy, values: dict[str, Any], where: str) -> TenantPolicy:
    if not isinstance(values, dict):
        raise PolicyError(f"{where}: must be a mapping")
    known = {f.name for f in fields(TenantPolicy)}
    unknown = set(values) - known
    if unknown:
        raise PolicyError(f"{where}: unknown keys {sorted(unknown)}")

    merged = dict(values)
    for key in ("allow_web_search", "allow_url_ingest", "require_verified_sources", "refuse_investment_advice"):
        if key in merged and not isinstance(merged[key], bool):
            raise PolicyError(f"{where}.{key}: must be true or false")
    if "url_allowlist" in merged:
        allow = merged["url_allowlist"]
        if not isinstance(allow, list) or not all(isinstance(h, str) and h for h in allow):
            raise PolicyError(f"{where}.url_allowlist: must be a list of host names")
        merged["url_allowlist"] = tuple(h.strip().lower().rstrip(".") for h in allow)
    if "max_classification" in merged:
        level = merged["max_classification"]
        if isinstance(level, bool) or not isinstance(level, int) or not 0 <= level <= 3:
            raise PolicyError(f"{where}.max_classification: must be an integer 0-3")
    if "answer_disclaimer" in merged:
        text = merged["answer_disclaimer"]
        if not isinstance(text, str) or len(text) > 500:
            raise PolicyError(f"{where}.answer_disclaimer: must be text of at most 500 characters")
    for key in ("ungrounded_answer_policy", "numeric_mismatch_policy"):
        if key in merged and merged[key] not in _ANSWER_POLICIES:
            raise PolicyError(f"{where}.{key}: must be one of {sorted(_ANSWER_POLICIES)}")
    return replace(base, **merged)


# ── decisions ────────────────────────────────────────────────────────────────

# Denial reasons that come from access control (audited as policy.deny); the
# others are ordinary configuration or a user's own choice.
POLICY_DENIALS = frozenset({"tenant_policy", "missing_scope"})


def web_search_decision(
    *,
    policy: TenantPolicy,
    principal: Principal,
    request_flag: bool | None,
    global_enabled: bool,
) -> tuple[bool, str | None]:
    """Return (allowed, reason). A request can only opt out, never opt in."""
    if not policy.allow_web_search:
        return False, "tenant_policy"
    if not principal.has_scope(SCOPE_WEB):
        return False, "missing_scope"
    if not global_enabled:
        return False, "global_disabled"
    if request_flag is False:
        return False, "request_opt_out"
    return True, None


def url_host_allowed(policy: TenantPolicy, url: str) -> bool:
    """True when the tenant may ingest *url*: enabled and host on its allow-list.

    An empty allow-list means nothing is allowed, even with ingest enabled.
    """
    if not policy.allow_url_ingest or not policy.url_allowlist:
        return False
    try:
        parts = urlsplit(url.strip())
        host = (parts.hostname or "").lower().rstrip(".")
    except ValueError:
        return False
    if not host or parts.username or parts.password:
        return False
    return any(host == d or host.endswith("." + d) for d in policy.url_allowlist)
