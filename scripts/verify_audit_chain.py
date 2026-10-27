"""Verify the audit log hash chain. Exit 0 if intact, 1 if tampered or unreadable."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings  # noqa: E402
from online_pipeline.audit.logger import verify_chain  # noqa: E402


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else settings.audit_log_path
    key = (settings.audit_hmac_key or "dev-audit-key-not-for-production").encode()
    result = verify_chain(path, key)
    if result.ok:
        print(f"OK: {result.events} events, chain intact ({path}).")
        return 0
    print(f"TAMPERED at seq {result.bad_seq}: {result.reason} ({result.events} events verified before it).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
