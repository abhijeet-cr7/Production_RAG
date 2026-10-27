"""FastAPI dependencies: authenticate the caller and enforce scopes."""

from __future__ import annotations

import functools
import uuid
from typing import Callable

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from config.settings import settings
from online_pipeline.audit.logger import record_best_effort
from online_pipeline.auth.principal import Principal
from online_pipeline.auth.tokens import AuthError, TokenValidator

_bearer = HTTPBearer(auto_error=False)


@functools.lru_cache(maxsize=1)
def get_validator() -> TokenValidator:
    return TokenValidator(settings)


def _client_ip(request: Request) -> str | None:
    return request.client.host if request.client else None


def current_principal(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> Principal:
    request_id = uuid.uuid4().hex
    request.state.request_id = request_id
    token = credentials.credentials if credentials is not None else None
    try:
        principal = get_validator().validate(token)
    except AuthError as exc:
        record_best_effort(
            "auth.fail",
            decision="deny",
            reason=exc.code,
            request_id=request_id,
            path=request.url.path,
            client_ip=_client_ip(request),
        )
        headers = {"WWW-Authenticate": "Bearer"} if exc.status_code == 401 else None
        raise HTTPException(status_code=exc.status_code, detail=exc.detail, headers=headers) from exc
    request.state.principal = principal
    return principal


def require_scope(scope: str) -> Callable[..., Principal]:
    """Dependency factory: authenticated *and* holding *scope*."""

    def dependency(
        request: Request, principal: Principal = Depends(current_principal)
    ) -> Principal:
        if not principal.has_scope(scope):
            record_best_effort(
                "authz.fail",
                actor=principal,
                decision="deny",
                reason="missing_scope",
                request_id=getattr(request.state, "request_id", None),
                required_scope=scope,
                path=request.url.path,
            )
            raise HTTPException(status_code=403, detail=f"Missing required scope '{scope}'.")
        return principal

    return dependency
