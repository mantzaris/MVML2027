"""Validate an explicitly mapped release CSV, create partitions, and audit them."""
import argparse
from pathlib import Path

from underwater_reasoning.data import inspect_manifest, assign_splits, audit_splits, purge_temporal_conflicts
from underwater_reasoning.io import atomic_json

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--manifest", required=True, type=Path)
parser.add_argument("--audio-root", required=True, type=Path)
parser.add_argument("--output", type=Path, default=Path("data/private/prepared"))
parser.add_argument("--seed", type=int, default=17)
parser.add_argument("--max-per-group", type=int, default=100)
args = parser.parse_args()
rows = inspect_manifest(args.manifest, args.audio_root)
rows = assign_splits(rows, args.seed, args.max_per_group)
rows, exclusions = purge_temporal_conflicts(rows)
audit = audit_splits(rows)
atomic_json(args.output / "manifest.json", rows)
atomic_json(args.output / "exclusions.json", exclusions)
atomic_json(args.output / "audit.json", audit)
print("PASS" if audit["passed"] else "BLOCKED", audit["counts"])
raise SystemExit(0 if audit["passed"] else 2)
