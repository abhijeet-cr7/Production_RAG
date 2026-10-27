"""HTML and XBRL extraction, run inside the sandbox worker.

HTML pages are sanitised (hidden text, scripts, frames removed) and turned into
blocks so tables survive. Inline XBRL pages additionally yield exact facts; plain
XBRL instance documents yield facts only. The XML is parsed with the hardened
parser (no DTD, entities or network).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from offline_pipeline.safety.errors import UnsafeContent
from offline_pipeline.safety.html_sanitizer import html_to_text, sanitize_html
from offline_pipeline.safety.upload_guard import read_text_bytes
from offline_pipeline.tables.blocks import blocks_to_json, blocks_to_text
from offline_pipeline.tables.html_blocks import html_to_blocks
from offline_pipeline.tables.xbrl import (
    facts_to_blocks,
    looks_like_ixbrl,
    parse_ixbrl,
    parse_xbrl_instance,
)


def extract_markup(path: Path, *, kind: str, max_chars: int) -> dict[str, Any]:
    data = path.read_bytes()
    if kind == "xbrl":
        return _extract_instance(data, max_chars)

    html = read_text_bytes(path, max_chars=max_chars)
    soup, report = sanitize_html(html)
    title = ""
    if soup.title is not None and soup.title.string:
        title = " ".join(soup.title.string.split())[:300]

    blocks = html_to_blocks(soup)
    metadata: dict[str, Any] = {
        "file_type": "html",
        "title": title,
        "hidden_elements_removed": report.hidden_elements_removed,
        "active_elements_removed": report.active_elements_removed,
    }
    if looks_like_ixbrl(data):
        try:
            facts, doc_meta = parse_ixbrl(data)
        except UnsafeContent as exc:
            if exc.code in {"xml_unsafe", "xbrl_too_many_facts"}:
                raise
            facts, doc_meta = [], {}
            metadata["xbrl_parse_failed"] = exc.code  # HTML is not always well-formed XML
        if facts:
            blocks.extend(facts_to_blocks(facts))
            metadata["xbrl_facts"] = len(facts)
        metadata.update({f"xbrl_{k}": v for k, v in doc_meta.items()})

    text = blocks_to_text(blocks) or html_to_text(soup)
    if len(text) > max_chars:
        raise UnsafeContent("text_too_large", "Extracted text is too large.")
    return {"text": text, "metadata": metadata, "blocks": blocks_to_json(blocks)}


def _extract_instance(data: bytes, max_chars: int) -> dict[str, Any]:
    # Parse first: a DTD/entity attack must be reported as such, not as "not XBRL".
    facts, doc_meta = parse_xbrl_instance(data)
    if not facts:
        raise UnsafeContent("xbrl_empty", "XBRL document contains no numeric facts.")
    blocks = facts_to_blocks(facts)
    text = blocks_to_text(blocks)
    if len(text) > max_chars:
        raise UnsafeContent("text_too_large", "Extracted text is too large.")
    metadata = {"file_type": "xbrl", "xbrl_facts": len(facts), **{f"xbrl_{k}": v for k, v in doc_meta.items()}}
    return {"text": text, "metadata": metadata, "blocks": blocks_to_json(blocks)}
