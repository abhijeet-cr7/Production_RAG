"""Cheap checks on an uploaded file before anything parses it."""

from __future__ import annotations

from pathlib import Path

from offline_pipeline.safety.errors import UnsafeContent

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".html", ".htm", ".xml", ".xbrl"}


def check_upload(path: Path, suffix: str) -> str:
    """Confirm the bytes match the extension. Returns the kind: pdf, docx or txt."""
    suffix = suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise UnsafeContent("unsupported_type", f"Unsupported file type '{suffix}'.")
    with open(path, "rb") as handle:
        head = handle.read(8192)
    if not head:
        raise UnsafeContent("empty_file", "File is empty.")

    if suffix == ".pdf":
        if b"%PDF-" not in head[:1024]:
            raise UnsafeContent("magic_mismatch", "File content is not a PDF.")
        return "pdf"
    if suffix == ".docx":
        if not head.startswith(b"PK\x03\x04"):
            raise UnsafeContent("magic_mismatch", "File content is not a DOCX.")
        return "docx"
    if b"\x00" in head:
        raise UnsafeContent("magic_mismatch", "File content is binary, not text.")
    if suffix in (".html", ".htm"):
        if not _markup(head):
            raise UnsafeContent("magic_mismatch", "File content is not HTML.")
        return "html"
    if suffix in (".xml", ".xbrl"):
        if not _markup(head):
            raise UnsafeContent("magic_mismatch", "File content is not XML.")
        return "xbrl"
    return "txt"


def _markup(head: bytes) -> bool:
    stripped = head.lstrip(b"\xef\xbb\xbf \t\r\n")
    return stripped.startswith(b"<") and b"%PDF-" not in head[:1024]


def extract_text_file(path: Path, *, max_chars: int) -> dict:
    """Read a .txt as UTF-8, rejecting binary content and oversized text."""
    raw = path.read_bytes()
    if b"\x00" in raw:
        raise UnsafeContent("txt_not_text", "File content is binary, not text.")
    text = raw.decode("utf-8-sig", errors="replace")
    if text.count("�") > max(8, len(text) // 50):
        raise UnsafeContent("txt_not_text", "File is not valid UTF-8 text.")
    if len(text) > max_chars:
        raise UnsafeContent("text_too_large", "Extracted text is too large.")
    return {"text": text, "metadata": {"file_type": "txt"}}


def read_text_bytes(path: Path, *, max_chars: int) -> str:
    """Decode a markup file as UTF-8 with a hard size ceiling (used by the worker)."""
    raw = path.read_bytes()
    if len(raw) > max_chars * 4:
        raise UnsafeContent("text_too_large", "Document is too large.")
    return raw.decode("utf-8", errors="replace")
