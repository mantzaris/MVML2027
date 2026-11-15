"""Strict canonical metadata ingestion and source-disjoint split preparation.

Release-specific column/label mapping must be verified separately. This module
never invents identity from a path and never silently repairs an official fold.
"""
from __future__ import annotations

import csv
import hashlib
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

from .io import canonical_hash, file_hash, safe_relative

LABELS = ("background", "cargo", "passenger", "tanker", "tug_tow")
PARTITIONS = ("acoustic_fit", "policy_fit", "development", "calibration", "test")
REQUIRED = ("clip_id", "audio_path", "label", "official_split", "mmsi",
            "deployment_id", "parent_recording_id", "start_utc")


def timestamp(value: str) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("Timestamp must include its verified UTC offset")
    return result.astimezone(timezone.utc)


def source_group(row: dict) -> str:
    if not row.get("deployment_id") or not row.get("parent_recording_id"):
        raise ValueError("Verified deployment and parent recording IDs are required")
    start = timestamp(row["start_utc"])
    if row["label"] == "background":
        if row.get("mmsi"):
            raise ValueError("Background with a vessel identity requires manual resolution")
        return "background:" + row["deployment_id"] + ":" + start.date().isoformat()
    mmsi = row.get("mmsi", "")
    if len(mmsi) != 9 or not mmsi.isdigit() or set(mmsi) == {"0"}:
        raise ValueError("A verified nine-digit MMSI is required for vessel clips")
    return "vessel:" + mmsi


def inspect_manifest(csv_path: Path, audio_root: Path) -> list[dict]:
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(REQUIRED) - set(reader.fieldnames or ())
        if missing:
            raise ValueError("Missing verified metadata columns: " + ", ".join(sorted(missing)))
        rows = list(reader)
    if not rows:
        raise ValueError("Manifest is empty")
    output, seen = [], set()
    for raw in rows:
        row = {key: raw[key].strip() for key in REQUIRED}
        if not row["clip_id"] or row["clip_id"] in seen:
            raise ValueError("Clip IDs must be present and unique")
        seen.add(row["clip_id"])
        if row["label"] not in LABELS or row["official_split"] not in {"train", "validation", "test"}:
            raise ValueError("Unknown label/split; verify the release mapping first")
        row["source_group"] = source_group(row)
        path = safe_relative(audio_root, row["audio_path"])
        waveform, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        if sample_rate != 32000 or waveform.shape != (160000, 1):
            raise ValueError("Expected verified native mono 32 kHz, 5 s audio; resolve header/channel differences")
        if not np.isfinite(waveform).all():
            raise ValueError("Non-finite audio samples")
        digest = hashlib.sha256(str(sample_rate).encode() + waveform.astype("<f4").tobytes()).hexdigest()
        row.update(sample_rate=sample_rate, duration_seconds=len(waveform) / sample_rate,
                   channels=1, waveform_sha256=digest, file_sha256=file_hash(path))
        output.append(row)
    return output


def assign_splits(rows: list[dict], seed: int = 17, max_per_group: int = 100) -> list[dict]:
    if max_per_group < 1:
        raise ValueError("Group cap must be positive")
    groups = defaultdict(list)
    for row in rows:
        groups[row["source_group"]].append(row)
    strata = defaultdict(list)
    for group, members in groups.items():
        splits = {r["official_split"] for r in members}
        labels = {r["label"] for r in members}
        if len(splits) != 1 or len(labels) != 1:
            raise ValueError("Source spans official partitions or inconsistent class labels: " + group)
        strata[(next(iter(splits)), next(iter(labels)))].append(group)
    assigned = {}
    for (official, label), group_ids in sorted(strata.items()):
        ordered = sorted(group_ids, key=lambda g: canonical_hash([seed, official, label, g]))
        if official == "test":
            assigned.update({g: "test" for g in ordered})
            continue
        if len(ordered) < 2:
            raise ValueError("Insufficient independent groups to subdivide " + official + "/" + label)
        ratio = 0.75 if official == "train" else 0.5
        cutoff = min(len(ordered) - 1, max(1, round(len(ordered) * ratio)))
        left, right = (("acoustic_fit", "policy_fit") if official == "train"
                       else ("development", "calibration"))
        assigned.update({g: left if i < cutoff else right for i, g in enumerate(ordered)})
    output = []
    for group, members in sorted(groups.items()):
        # Preserve every official test clip for the classifier evaluation.
        selected = sorted(members, key=lambda r: canonical_hash([seed, r["clip_id"]]))
        if assigned[group] != "test":
            selected = selected[:max_per_group]
        for row in selected:
            output.append(dict(row, partition=assigned[group]))
    return sorted(output, key=lambda r: r["clip_id"])


def audit_splits(rows: list[dict], guard_seconds: float = 60) -> dict:
    if guard_seconds < 0:
        raise ValueError("Temporal guard cannot be negative")
    problems = []
    for field in ("source_group", "parent_recording_id", "waveform_sha256"):
        assignments = defaultdict(set)
        for row in rows:
            if not row.get(field):
                problems.append({"kind": "missing", "field": field, "clip_id": row["clip_id"]})
            assignments[row.get(field)].add(row["partition"])
        for value, parts in assignments.items():
            if len(parts) > 1:
                problems.append({"kind": "cross_partition_" + field, "id": value,
                                 "partitions": sorted(parts)})
    deployments = defaultdict(list)
    ids = Counter()
    for row in rows:
        ids[row["clip_id"]] += 1
        if row["partition"] not in PARTITIONS:
            problems.append({"kind": "invalid_partition", "clip_id": row["clip_id"]})
        if source_group(row) != row["source_group"]:
            problems.append({"kind": "incorrect_group", "clip_id": row["clip_id"]})
        start = timestamp(row["start_utc"]).timestamp()
        deployments[row["deployment_id"]].append((start, row))
    for clip_id, count in ids.items():
        if count != 1:
            problems.append({"kind": "duplicate_clip_id", "clip_id": clip_id})
    for deployment, items in deployments.items():
        active = []
        for start, row in sorted(items, key=lambda x: (x[0], x[1]["clip_id"])):
            active = [(end, other) for end, other in active if end + guard_seconds > start]
            for end, other in active:
                if row["partition"] != other["partition"]:
                    problems.append({"kind": "temporal_guard", "deployment_id": deployment,
                                     "clips": [other["clip_id"], row["clip_id"]]})
            active.append((start + row["duration_seconds"], row))
    counts = {part: {"clips": sum(r["partition"] == part for r in rows),
                     "groups": len({r["source_group"] for r in rows if r["partition"] == part}),
                     "classes": dict(Counter(r["label"] for r in rows if r["partition"] == part))}
              for part in PARTITIONS}
    return {"passed": bool(rows) and not problems, "problems": problems, "counts": counts,
            "manifest_sha256": canonical_hash(rows), "guard_seconds": guard_seconds}


def purge_temporal_conflicts(rows: list[dict], guard_seconds: float = 60) -> tuple[list[dict], list[dict]]:
    """Preserve test, then calibration/development/policy/acoustic in that order.

    Does not resolve identity/hash/parent conflicts; those still fail the audit.
    All exclusions are returned for review before the protocol is frozen.
    """
    audit = audit_splits(rows, guard_seconds)
    rank = {part: i for i, part in enumerate(PARTITIONS)}
    by_id = {r["clip_id"]: r for r in rows}
    conflicts = defaultdict(set)
    for issue in audit["problems"]:
        if issue["kind"] == "temporal_guard":
            a, b = issue["clips"]
            conflicts[a].add(b)
            conflicts[b].add(a)
    kept, removed = set(), []
    ordered = sorted(rows, key=lambda r: (-rank[r["partition"]], r["clip_id"]))
    for row in ordered:
        blockers = conflicts[row["clip_id"]] & kept
        if blockers:
            removed.append({"clip_id": row["clip_id"], "reason": "temporal_guard",
                            "retained_conflicts": sorted(blockers)})
        else:
            kept.add(row["clip_id"])
    return [by_id[k] for k in sorted(kept)], removed


def sample_annotations(rows: list[dict], counts: dict[str, int], seed: int = 17,
                       max_per_group: int = 3, min_separation: float = 600) -> list[dict]:
    selected = []
    for partition, total in counts.items():
        if total % len(LABELS):
            raise ValueError("Class-balanced annotation counts must be divisible by five")
        target = total // len(LABELS)
        for label in LABELS:
            candidates = defaultdict(list)
            for row in rows:
                if row["partition"] == partition and row["label"] == label:
                    candidates[row["source_group"]].append(row)
            chosen_times = defaultdict(list)
            for values in candidates.values():
                values.sort(key=lambda r: canonical_hash([seed, r["clip_id"]]))
            ordered_groups = sorted(candidates, key=lambda g: canonical_hash([seed, partition, label, g]))
            current = []
            for round_index in range(max_per_group):
                for group in ordered_groups:
                    if len(current) == target:
                        break
                    for row in candidates[group]:
                        instant = timestamp(row["start_utc"]).timestamp()
                        if row not in current and all(abs(instant - t) >= min_separation for t in chosen_times[group]):
                            current.append(row)
                            chosen_times[group].append(instant)
                            break
                if len(current) == target:
                    break
            if len(current) != target:
                raise ValueError(f"Insufficient independent annotation candidates for {partition}/{label}: "
                                 f"{len(current)}/{target}; do not silently relax constraints")
            selected.extend(current)
    return selected
