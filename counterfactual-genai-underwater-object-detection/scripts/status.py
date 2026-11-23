#!/usr/bin/env python3
"""Read project run ledgers and checkpoint metadata without torch or CUDA."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from uac.runtime import EventLedger, PROJECT_ROOT, project_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", default="runs/events.jsonl")
    parser.add_argument("--tail", type=int, default=12)
    parser.add_argument("--repair-tail", action="store_true",
                        help="Quarantine and remove only an incomplete final JSONL append")
    args = parser.parse_args()
    ledger = EventLedger(args.ledger)
    try:
        rows = ledger.read(repair_tail=args.repair_tail)
    except ValueError as exc:
        print(json.dumps({"status": "ledger_error", "error": str(exc)}, indent=2))
        return 2
    snapshots = []
    for path in sorted(PROJECT_ROOT.glob("checkpoints/**/latest.json")):
        try:
            value = json.loads(path.read_text())
            snapshots.append({"path": str(path.relative_to(PROJECT_ROOT)),
                              "latest": value.get("latest"), "previous": value.get("previous"),
                              "global_step": value.get("entries", [{}])[0].get("global_step")})
        except (ValueError, OSError, IndexError):
            snapshots.append({"path": str(path.relative_to(PROJECT_ROOT)), "status": "unreadable_index"})
    print(json.dumps({"project": str(ROOT), "event_count": len(rows),
                      "recent_events": rows[-max(0, args.tail):] if args.tail > 0 else [],
                      "checkpoint_indices": snapshots,
                      "note": "Read-only metadata status; training restore verifies payload checksums."}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
