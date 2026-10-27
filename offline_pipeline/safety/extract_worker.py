"""Sandbox worker: parse ONE untrusted file under hard resource limits.

Run as ``python -I extract_worker.py <kind> <path> <cpu_s> <mem_mb> <max_pages>
<max_chars> <ocr_pages>``. It limits itself first (CPU, address space where the
OS enforces it, file size, no core dumps), then parses, then prints exactly one
JSON line. The parent treats any other outcome (crash, signal, timeout, memory
kill, malformed output) as a rejection, so a parser exploit or bomb can at worst
cost one disposable process.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Isolated mode drops the script directory from sys.path; our own package root is
# trusted, the (untrusted) upload lives elsewhere and is never importable.
ROOT = str(Path(__file__).resolve().parents[2])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _limit(cpu_seconds: int, mem_mb: int) -> None:
    import resource

    def cap(which: int, soft: int) -> None:
        try:
            _, hard = resource.getrlimit(which)
            limit = soft if hard == resource.RLIM_INFINITY else min(soft, hard)
            resource.setrlimit(which, (limit, limit))  # hard too: cannot be raised back
        except (ValueError, OSError):
            pass  # not supported/enforced on this OS; the parent watchdog backs it up

    cap(resource.RLIMIT_CPU, cpu_seconds)
    cap(resource.RLIMIT_AS, mem_mb * 1024 * 1024)
    cap(resource.RLIMIT_FSIZE, 128 * 1024 * 1024)
    cap(resource.RLIMIT_CORE, 0)


def main(argv: list[str]) -> int:
    kind, path = argv[1], Path(argv[2])
    cpu, mem_mb, max_pages, max_chars, ocr_pages = (int(v) for v in argv[3:8])
    _limit(cpu, mem_mb)

    from offline_pipeline.safety.errors import UnsafeContent

    try:
        if kind == "pdf":
            from offline_pipeline.safety.pdf_guard import extract_pdf

            doc = extract_pdf(path, max_pages=max_pages, max_chars=max_chars, ocr_max_pages=ocr_pages)
        elif kind == "docx":
            from offline_pipeline.safety.docx_guard import extract_docx

            doc = extract_docx(path, max_chars=max_chars)
        elif kind == "txt":
            from offline_pipeline.safety.upload_guard import extract_text_file

            doc = extract_text_file(path, max_chars=max_chars)
        else:
            raise UnsafeContent("unsupported_type", "Unsupported file type.")
        result = {"ok": True, "doc": doc}
    except UnsafeContent as exc:
        result = {"ok": False, "code": exc.code, "detail": exc.detail}
    except MemoryError:
        result = {"ok": False, "code": "parser_memory", "detail": "Parser exceeded its memory limit."}
    except RecursionError:
        result = {"ok": False, "code": "parser_recursion", "detail": "Document nesting is too deep."}
    except Exception as exc:  # noqa: BLE001
        result = {"ok": False, "code": "parser_error", "detail": f"Parser failed ({type(exc).__name__})."}

    sys.stdout.write(json.dumps(result, ensure_ascii=True) + "\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    os._exit(main(sys.argv))
