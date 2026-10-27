"""Mint a short-lived HS256 token for local development (APP_ENV=dev, AUTH_MODE=dev_token)."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jwt  # noqa: E402

from config.settings import settings  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user", default="dev-user")
    parser.add_argument("--tenant", required=True)
    parser.add_argument("--groups", default="", help="comma separated")
    parser.add_argument("--clearance", type=int, default=1)
    parser.add_argument("--scopes", default="rag:query", help="comma separated")
    parser.add_argument("--ttl", type=int, default=3600, help="seconds")
    args = parser.parse_args()

    if settings.app_env != "dev" or settings.auth_mode != "dev_token":
        print("Refusing: requires APP_ENV=dev and AUTH_MODE=dev_token.", file=sys.stderr)
        return 1
    if len(settings.dev_jwt_secret) < 32:
        print("Refusing: DEV_JWT_SECRET must be at least 32 characters.", file=sys.stderr)
        return 1

    now = int(time.time())
    claims = {
        "iss": "dev-issuer",
        "aud": settings.okta_audience,
        "sub": args.user,
        "iat": now,
        "exp": now + args.ttl,
        settings.okta_tenant_claim: args.tenant,
        settings.okta_groups_claim: [g for g in args.groups.split(",") if g],
        settings.okta_clearance_claim: args.clearance,
        settings.okta_scopes_claim: [s for s in args.scopes.split(",") if s],
    }
    print(jwt.encode(claims, settings.dev_jwt_secret, algorithm="HS256"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
