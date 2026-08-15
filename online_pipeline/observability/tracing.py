"""LangSmith tracing: decorate pipeline stages, degrade to a no-op when disabled."""

from __future__ import annotations

import functools
import inspect
import logging
import os
from typing import Any, Callable, TypeVar

from config.settings import settings

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

_configured = False


def tracing_enabled() -> bool:
    """True when LangSmith tracing is switched on and an API key is present."""
    return bool(settings.langsmith_tracing and settings.langsmith_api_key)


def configure_tracing() -> bool:
    """Export the LangSmith environment once. Returns whether tracing is active."""
    global _configured
    if _configured:
        return tracing_enabled()
    _configured = True

    if not tracing_enabled():
        logger.info("LangSmith tracing disabled (set LANGSMITH_TRACING and LANGSMITH_API_KEY).")
        return False

    # Both the modern LANGSMITH_* and legacy LANGCHAIN_* names are read by the SDK.
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint
    os.environ["LANGCHAIN_ENDPOINT"] = settings.langsmith_endpoint
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project

    logger.info("LangSmith tracing enabled for project '%s'.", settings.langsmith_project)
    return True


def traced(name: str | None = None, run_type: str = "chain") -> Callable[[F], F]:
    """Record the wrapped callable as a LangSmith run.

    The LangSmith wrapper is resolved lazily on first call so that settings and
    imports are evaluated after application startup, and so that a missing
    ``langsmith`` install or disabled tracing costs nothing at runtime.
    """

    def decorator(func: F) -> F:
        resolved: list[Callable[..., Any]] = []

        def target() -> Callable[..., Any]:
            if not resolved:
                resolved.append(_wrap(func, name or func.__qualname__, run_type))
            return resolved[0]

        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                return await target()(*args, **kwargs)

            return async_wrapper  # type: ignore[return-value]

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return target()(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


def _wrap(func: Callable[..., Any], name: str, run_type: str) -> Callable[..., Any]:
    if not configure_tracing():
        return func
    try:
        from langsmith import traceable
    except ImportError:
        logger.warning("langsmith is not installed; skipping trace for '%s'.", name)
        return func
    try:
        return traceable(name=name, run_type=run_type)(func)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not attach LangSmith trace to '%s': %s", name, exc)
        return func
