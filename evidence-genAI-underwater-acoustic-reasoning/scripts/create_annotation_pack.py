import argparse
import json
from pathlib import Path

from underwater_reasoning.annotation import create_pack
from underwater_reasoning.data import audit_splits, sample_annotations

parser = argparse.ArgumentParser()
parser.add_argument("--manifest", type=Path, required=True)
parser.add_argument("--audio-root", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--private-key", type=Path, required=True)
parser.add_argument("--config", type=Path, default=Path("configs/study.json"))
args = parser.parse_args()
rows = json.loads(args.manifest.read_text())
if not audit_splits(rows)["passed"]:
    raise SystemExit("Blocked: source integrity must pass before formal annotation sampling")
config = json.loads(args.config.read_text())
selected = sample_annotations(rows, config["annotation_counts"], config["split_seed"],
                              config["annotation_max_per_group"], config["annotation_min_separation_seconds"])
result = create_pack(selected, args.audio_root, args.output, args.private_key)
print("Created unannotated pack", result["pack_id"], "with", len(selected), "clips")
