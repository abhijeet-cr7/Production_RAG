"""Delete documents older than the retention period, skipping any under legal hold.

    python scripts/purge_expired.py --days 2555              # dry run: list what would go
    python scripts/purge_expired.py --days 2555 --apply
    RETENTION_DAYS=2555 python scripts/purge_expired.py --apply   # days from settings

Each deletion removes the chunks and the stored original (unless another document uses the
same bytes) and writes a ``doc.delete`` audit event. Run it from cron or a scheduled job.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings  # noqa: E402
from online_pipeline.lifecycle import purge_expired  # noqa: E402
from online_pipeline.provenance.rawstore import RawStore  # noqa: E402
from vector_db.client import VectorDBClient  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--days", type=int, default=settings.retention_days, help="retention period in days")
    parser.add_argument("--apply", action="store_true", help="delete (default: dry run)")
    args = parser.parse_args()
    if args.days <= 0:
        print("Nothing to do: set --days or RETENTION_DAYS to a positive number.")
        return 0

    raw = RawStore(settings.raw_store_dir) if settings.raw_store_enabled else None
    report = purge_expired(VectorDBClient(), raw, days=args.days, apply=args.apply)
    mode = "APPLIED" if args.apply else "DRY RUN (nothing deleted; add --apply)"
    print(f"{mode}: examined {report.examined}, expired {report.expired}, "
          f"held {report.held}, deleted {report.deleted}.")
    for doc_id in report.doc_ids[:50]:
        print(f"  {'deleted' if args.apply else 'would delete'} {doc_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
