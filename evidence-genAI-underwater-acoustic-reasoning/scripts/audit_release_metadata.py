"""Reproduce aggregate provenance and fold-capacity findings; never model outcomes."""
import csv
import io
import json
from collections import defaultdict
from pathlib import Path

from probe_release import RemoteSplitZip
from underwater_reasoning.io import atomic_json

root = Path("data/raw/release_probe")
classes = ["background", "cargo", "passengership", "tanker", "tug"]
groups = defaultdict(dict)
all_vessels = {}
vessel_labels = defaultdict(set)
summary = {}
for split in ["train", "validation", "test"]:
    with (root / f"metadata_5s_{split}.csv").open() as handle:
        rows = [r for r in csv.DictReader(handle) if r["label"] in classes]
    all_vessels[split] = {r["MMSI"] for r in rows if r["label"] != "background"}
    for row in rows:
        if row["label"] != "background": vessel_labels[row["MMSI"]].add(row["label"])
    counts = {}
    for label in classes:
        subset = [r for r in rows if r["label"] == label]
        ids = {r["date"] if label == "background" else r["MMSI"] for r in subset}
        groups[label][split] = ids
        counts[label] = {"clips": len(subset), "days" if label == "background" else "vessels": len(ids),
                         "missing_hydrophone": sum(not r["hydrophone"] for r in subset),
                         "shorter_than_5s": sum(float(r["duration_sec"]) < 4.999 for r in subset)}
    summary[split] = counts
summary["cross_split_group_overlap"] = {
    c: {a + "_" + b: len(groups[c][a] & groups[c][b])
        for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]}
    for c in classes}
summary["all_vessel_cross_split_overlap"] = {a + "_" + b: len(all_vessels[a] & all_vessels[b])
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]}
summary["vessels_with_conflicting_class_labels"] = sum(len(v) > 1 for v in vessel_labels.values())
summary["blocking_fields"] = [
    "No parent recording ID column", "No deployment ID column; hydrophone device IDs are not deployment IDs",
    "Background hydrophone fields missing",
    "Only date field for time; documented filenames contain clip times but do not establish parent/deployment identity"]
atomic_json(Path("sources/grouping_audit.json"), summary)
archive = RemoteSplitZip()
entries = json.loads(Path(".cache/release_index.json").read_text())
feasibility = []
for entry in entries:
    if not entry["name"].endswith("metadata_5s_test.csv"):
        continue
    rows = list(csv.DictReader(io.StringIO(archive.extract_bytes(entry).decode("utf-8-sig"))))
    vessels = {r["MMSI"] for r in rows if r["label"] == "passengership"}
    feasibility.append({"fold": entry["name"].split("/")[-2], "passenger_test_vessels": len(vessels),
                        "maximum_passenger_annotations_at_cap3": 3 * len(vessels),
                        "needed_balanced_passenger_annotations": 48, "capacity_sufficient": 3 * len(vessels) >= 48})
atomic_json(Path("sources/fold_annotation_feasibility.json"), sorted(feasibility, key=lambda r: r["fold"]))
print("Aggregate audits saved; main-study provenance and annotation capacity remain blocked.")
