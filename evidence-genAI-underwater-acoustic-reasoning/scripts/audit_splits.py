import argparse
import json
from pathlib import Path
from underwater_reasoning.data import audit_splits
from underwater_reasoning.io import atomic_json

parser = argparse.ArgumentParser()
parser.add_argument("manifest", type=Path)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
report = audit_splits(json.loads(args.manifest.read_text()))
atomic_json(args.output, report)
print("PASS" if report["passed"] else "BLOCKED", report["counts"])
raise SystemExit(0 if report["passed"] else 2)
