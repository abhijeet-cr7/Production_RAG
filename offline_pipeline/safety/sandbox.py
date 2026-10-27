"""Run the untrusted-file parser in a disposable, resource-limited subprocess."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from offline_pipeline.safety.errors import UnsafeContent
from offline_pipeline.safety.upload_guard import check_upload

_WORKER = Path(__file__).with_name("extract_worker.py")
_MAX_RESULT_BYTES = 64 * 1024 * 1024


def _rss_mb(pid: int) -> float:
    """Resident memory of *pid* in MB (portable: works on macOS and Linux)."""
    try:
        out = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True, timeout=2
        ).stdout.strip()
        return int(out) / 1024 if out else 0.0
    except (subprocess.SubprocessError, ValueError, OSError):
        return 0.0


def _kill(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        proc.kill()
    proc.wait()


def safe_extract_file(
    path: str | Path,
    suffix: str,
    *,
    timeout: float = 120.0,
    cpu_seconds: int = 110,
    max_memory_mb: int = 1024,
    max_pages: int = 500,
    max_chars: int = 5_000_000,
    ocr_max_pages: int = 20,
) -> dict[str, Any]:
    """Validate then parse *path* in a sandbox; returns ``{"text", "metadata"}``.

    Raises :class:`UnsafeContent` for anything unsafe, malformed, too big, too
    slow or too hungry. Output goes to a size-limited temp file rather than a
    pipe so a hostile child cannot flood the parent's memory.
    """
    path = Path(path)
    kind = check_upload(path, suffix)
    return run_worker(
        kind, path, timeout=timeout, cpu_seconds=cpu_seconds, max_memory_mb=max_memory_mb,
        max_pages=max_pages, max_chars=max_chars, ocr_max_pages=ocr_max_pages,
    )


def run_worker(
    kind: str,
    path: Path,
    *,
    timeout: float = 120.0,
    cpu_seconds: int = 110,
    max_memory_mb: int = 1024,
    max_pages: int = 500,
    max_chars: int = 5_000_000,
    ocr_max_pages: int = 20,
) -> dict[str, Any]:
    """Parse *path* as *kind* (pdf, docx, txt, html, xbrl) inside the sandbox."""
    cmd = [
        sys.executable, "-I", str(_WORKER), kind, str(path),
        str(cpu_seconds), str(max_memory_mb), str(max_pages), str(max_chars), str(ocr_max_pages),
    ]
    # Minimal environment: no API keys, no tokens, nothing the parser could leak.
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "LANG": "C.UTF-8"}

    with tempfile.TemporaryFile() as out:
        proc = subprocess.Popen(
            cmd, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.DEVNULL,
            env=env, cwd=tempfile.gettempdir(), start_new_session=True, close_fds=True,
        )
        deadline = time.monotonic() + timeout
        next_mem_check = 0.0
        while proc.poll() is None:
            now = time.monotonic()
            if now > deadline:
                _kill(proc)
                raise UnsafeContent("parser_timeout", "Parsing took too long.")
            if now >= next_mem_check:
                next_mem_check = now + 0.25
                if _rss_mb(proc.pid) > max_memory_mb:
                    _kill(proc)
                    raise UnsafeContent("parser_memory", "Parsing used too much memory.")
            time.sleep(0.02)

        if proc.returncode != 0:
            raise UnsafeContent("parser_crashed", "The document could not be parsed safely.")
        out.seek(0, os.SEEK_END)
        if out.tell() > _MAX_RESULT_BYTES:
            raise UnsafeContent("text_too_large", "Extracted text is too large.")
        out.seek(0)
        raw = out.read()

    try:
        result = json.loads(raw.decode("utf-8", errors="strict").strip().splitlines()[-1])
    except (ValueError, IndexError, UnicodeDecodeError) as exc:
        raise UnsafeContent("parser_crashed", "The document could not be parsed safely.") from exc
    if not isinstance(result, dict):
        raise UnsafeContent("parser_crashed", "The document could not be parsed safely.")
    if not result.get("ok"):
        raise UnsafeContent(str(result.get("code", "parser_error")), str(result.get("detail", "Rejected.")))
    doc = result.get("doc")
    if not isinstance(doc, dict) or not isinstance(doc.get("text"), str):
        raise UnsafeContent("parser_crashed", "The document could not be parsed safely.")
    if "blocks" in doc and not (isinstance(doc["blocks"], list) and all(isinstance(b, dict) for b in doc["blocks"])):
        raise UnsafeContent("parser_crashed", "The document could not be parsed safely.")
    return doc
