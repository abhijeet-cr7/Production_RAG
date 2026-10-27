"""Give documents indexed before access control existed an owner (they are unreadable otherwise).

Points with no ACL are denied to everyone. This assigns them to one tenant and owner, private
to that owner, classification 1 (internal) and trust tier ``unverified`` (their origin is
unknown). Points that already have a complete ACL are never touched, so re-running is safe.

    python scripts/backfill_acl.py --tenant legacy --owner admin@acme.com           # dry run
    python scripts/backfill_acl.py --tenant legacy --owner admin@acme.com --apply
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings  # noqa: E402
from online_pipeline.audit.logger import record_best_effort  # noqa: E402
from online_pipeline.lifecycle import backfill_acl  # noqa: E402
from vector_db.client import VectorDBClient  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tenant", required=True, help="tenant id the legacy documents will belong to")
    parser.add_argument("--owner", required=True, help="user id that will own them")
    parser.add_argument("--apply", action="store_true", help="write the changes (default: dry run)")
    args = parser.parse_args()

    report = backfill_acl(VectorDBClient(), tenant_id=args.tenant, owner_id=args.owner, apply=args.apply)
    mode = "APPLIED" if args.apply else "DRY RUN (nothing written; add --apply)"
    print(f"{mode}: scanned {report.scanned} points, {report.needing_acl} lack an ACL, {report.updated} updated.")
    if args.apply:
        record_best_effort(
            "admin.backfill_acl", actor="script", tenant=args.tenant,
            owner=args.owner, scanned=report.scanned, updated=report.updated,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
