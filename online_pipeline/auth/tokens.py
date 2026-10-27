"""Token validation: Okta RS256 access tokens, plus dev-only modes.

Only the algorithm pinned for the mode is accepted, so ``alg: none`` and
RS256→HS256 key-confusion tokens are rejected. Signing keys come from the
configured JWKS URL only; ``jku``/``x5u`` headers are never followed.
"""

from __future__ import annotations

import logging
from typing import Any

import jwt

from config.settings import Settings
from online_pipeline.auth.principal import (
    ALL_SCOPES,
    MAX_CLEARANCE,
    MIN_CLEARANCE,
    Principal,
)

logger = logging.getLogger(__name__)

_MAX_TOKEN_BYTES = 8192
_AUTH_MODES = {"okta", "dev_token", "disabled"}
_DEV_ISSUER = "dev-issuer"


class AuthError(Exception):
    """Authentication or token-claim failure with a stable audit reason code."""

    def __init__(self, status_code: int, code: str, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.code = code
        self.detail = detail


def validate_auth_config(cfg: Settings) -> None:
    """Refuse to start with an unsafe or incomplete auth/audit configuration."""
    if cfg.auth_mode not in _AUTH_MODES:
        raise RuntimeError(f"AUTH_MODE must be one of {sorted(_AUTH_MODES)}, got {cfg.auth_mode!r}.")
    dev = cfg.app_env == "dev"
    if cfg.auth_mode in {"disabled", "dev_token"} and not dev:
        raise RuntimeError(f"AUTH_MODE={cfg.auth_mode} is only allowed when APP_ENV=dev.")
    if cfg.auth_mode == "okta":
        if not cfg.okta_issuer.startswith("https://"):
            raise RuntimeError("OKTA_ISSUER must be an https:// URL when AUTH_MODE=okta.")
        if cfg.okta_jwks_url and not cfg.okta_jwks_url.startswith("https://"):
            raise RuntimeError("OKTA_JWKS_URL must be an https:// URL.")
        if not cfg.okta_audience:
            raise RuntimeError("OKTA_AUDIENCE must be set when AUTH_MODE=okta.")
    if cfg.auth_mode == "dev_token" and len(cfg.dev_jwt_secret) < 32:
        raise RuntimeError("DEV_JWT_SECRET must be at least 32 characters for AUTH_MODE=dev_token.")
    if not dev and not cfg.audit_hmac_key:
        raise RuntimeError("AUDIT_HMAC_KEY must be set outside APP_ENV=dev.")
    if not dev and not cfg.qdrant_api_key:
        raise RuntimeError("QDRANT_API_KEY must be set outside APP_ENV=dev: the vector store holds every tenant's data.")


class TokenValidator:
    """Turn a bearer token into a :class:`Principal`."""

    def __init__(self, cfg: Settings) -> None:
        self.cfg = cfg
        self.mode = cfg.auth_mode
        self._jwks: Any = None
        if self.mode == "okta":
            url = cfg.okta_jwks_url or cfg.okta_issuer.rstrip("/") + "/v1/keys"
            self._jwks = jwt.PyJWKClient(url, cache_jwk_set=True, lifespan=600, timeout=5)

    def validate(self, token: str | None) -> Principal:
        if self.mode == "disabled":
            return Principal(
                user_id="dev-user",
                tenant_id="dev",
                groups=frozenset({"dev"}),
                clearance=MAX_CLEARANCE,
                scopes=ALL_SCOPES,
            )
        if not token:
            raise AuthError(401, "missing_token", "Authentication required.")
        if len(token.encode()) > _MAX_TOKEN_BYTES:
            raise AuthError(401, "malformed", "Invalid token.")
        return self._principal_from_claims(self._decode(token))

    # ── internals ────────────────────────────────────────────────────────────

    def _decode(self, token: str) -> dict[str, Any]:
        cfg = self.cfg
        options = {"require": ["exp", "iss", "aud", "sub"]}
        try:
            if self.mode == "okta":
                key = self._jwks.get_signing_key_from_jwt(token).key
                return jwt.decode(
                    token, key, algorithms=["RS256"], audience=cfg.okta_audience,
                    issuer=cfg.okta_issuer, leeway=cfg.jwt_leeway_seconds, options=options,
                )
            return jwt.decode(
                token, cfg.dev_jwt_secret, algorithms=["HS256"], audience=cfg.okta_audience,
                issuer=_DEV_ISSUER, leeway=cfg.jwt_leeway_seconds, options=options,
            )
        except jwt.PyJWKClientConnectionError as exc:
            logger.error("JWKS endpoint unreachable: %s", exc)
            raise AuthError(503, "jwks_unavailable", "Authentication service unavailable.") from exc
        except jwt.ExpiredSignatureError as exc:
            raise AuthError(401, "token_expired", "Token expired.") from exc
        except jwt.InvalidAudienceError as exc:
            raise AuthError(401, "bad_audience", "Invalid token.") from exc
        except jwt.InvalidIssuerError as exc:
            raise AuthError(401, "bad_issuer", "Invalid token.") from exc
        except jwt.InvalidSignatureError as exc:
            raise AuthError(401, "bad_signature", "Invalid token.") from exc
        except jwt.MissingRequiredClaimError as exc:
            raise AuthError(401, "missing_claim", "Invalid token.") from exc
        except jwt.PyJWTError as exc:  # malformed, wrong alg, unknown kid, ...
            raise AuthError(401, "invalid_token", "Invalid token.") from exc

    def _principal_from_claims(self, claims: dict[str, Any]) -> Principal:
        cfg = self.cfg
        tenant = claims.get(cfg.okta_tenant_claim)
        # A missing or ambiguous tenant must never fall back to a default.
        if not isinstance(tenant, str) or not tenant.strip():
            raise AuthError(403, "no_tenant", "Token has no tenant.")

        return Principal(
            user_id=str(claims.get("uid") or claims["sub"]),
            tenant_id=tenant.strip(),
            groups=frozenset(_as_str_list(claims.get(cfg.okta_groups_claim))),
            clearance=_clearance(claims.get(cfg.okta_clearance_claim)),
            scopes=frozenset(_as_str_list(claims.get(cfg.okta_scopes_claim) or claims.get("scope"))),
        )


def _as_str_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return value.split()
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if isinstance(v, str) and v]
    return []


def _clearance(value: Any) -> int:
    # Unparseable clearance is treated as the lowest level, never the highest.
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        return MIN_CLEARANCE
    try:
        level = int(value)
    except ValueError:
        return MIN_CLEARANCE
    return max(MIN_CLEARANCE, min(MAX_CLEARANCE, level))
