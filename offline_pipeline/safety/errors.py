"""Error type shared by every untrusted-input guard."""

from __future__ import annotations


class UnsafeContent(Exception):
    """Input was refused by a security check.

    ``code`` is a stable machine-readable reason (safe to log and audit);
    ``detail`` is a short human message that never echoes the content itself.
    """

    def __init__(self, code: str, detail: str | None = None) -> None:
        super().__init__(detail or code)
        self.code = code
        self.detail = detail or code
