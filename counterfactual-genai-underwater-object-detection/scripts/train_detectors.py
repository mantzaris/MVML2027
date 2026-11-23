#!/usr/bin/env python3
"""Restartable real-data detector fitting and development-only model selection.

No test/calibration feature or label access occurs. Per-view and every acquired
primary-view subset get separate numeric heads. Models share the frozen real
train/development cohorts; scaling fits training only. Strong CNN and frozen
speech-foundation transfer baselines are evaluated as separate comparators.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
import torch
from uac.models import FeatureDetector, SpectralDetector
from uac.training_v2 import seed_all, train_supervised
from uac.runtime import RunLock, EventLedger, atomic_json, fingerprint, source_fingerprint, sha256_file, project_path, utc_now
from extract_real_features import _safe_gpu
from extract_foundation import load_index, load_shard

FEATURE_INDICES = {
    "fourier": list(range(0, 32)), "stft": list(range(32, 96)), "wavelet": list(range(96, 152)),
    "fourier_stft": list(range(0, 96)), "fourier_wavelet": list(range(0, 32)) + list(range(96, 152)),
    "stft_wavelet": list(range(32, 152)), "main": list(range(152)), "hht_exploratory": list(range(184)),
}
PRIMARY_METHODS = tuple(name for name in FEATURE_INDICES if name != "hht_exploratory") + ("cnn", "foundation")
METHODS = tuple(FEATURE_INDICES) + ("cnn", "foundation")


def load_split_tensors(split, root, include_audio=False):
    index, index_hash = load_index(split, root)
    features, labels, audio, ids, rows = [], [], [], [], []
    for record in index["shards"]:
        shard = load_shard(record)
        features.append(shard["features"])
        labels.append(shard["labels"])
        ids.extend(shard["ids"])
        rows.extend(shard["rows"])
        if include_audio:
            audio.append(shard["audio"])
    if len(ids) != len(set(ids)) or len(ids) != index["expected_rows"]:
        raise RuntimeError("Duplicate or missing source identities in feature shards")
    return {"features": torch.cat(features).to("cuda"), "labels": torch.cat(labels).to("cuda"),
            "audio": torch.cat(audio).to("cuda") if include_audio else None,
            "ids": ids, "rows": rows, "index": index, "index_sha256": index_hash}


def update_selection(records, requested_seeds, path, data_fingerprint, pilot):
    eligible, incomplete = [], []
    for method in PRIMARY_METHODS:
        chosen = [row for row in records if row["method"] == method and row["seed"] in requested_seeds]
        available = {row["seed"] for row in chosen}
        if available == set(requested_seeds):
            eligible.append({"method": method, "seed_mean_development_ap": sum(row["dev_ap"] for row in chosen) / len(chosen),
                             "seeds": sorted(available), "per_seed_ap": {str(row["seed"]): row["dev_ap"] for row in chosen}})
        else:
            incomplete.append({"method": method, "missing_seeds": sorted(set(requested_seeds) - available)})
    eligible.sort(key=lambda row: (-row["seed_mean_development_ap"], row["method"]))
    record = {"schema_version": 1, "status": "pilot" if pilot else ("development_complete" if not incomplete else "partial"),
              "selection_basis": "highest seed-mean AP on development, excluding exploratory HHT; lexical tie break",
              "data_fingerprint": data_fingerprint, "requested_seeds": requested_seeds,
              "ranked_complete_methods": eligible, "incomplete_methods": incomplete,
              "provisional_best_method": eligible[0]["method"] if eligible else None,
              "locked_comparator": None, "test_examined": False, "updated_utc": utc_now(),
              "note": "This is development ranking. Final comparator and protocol are locked separately before test."}
    atomic_json(path, record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--real-dir", default="data/features/real")
    parser.add_argument("--foundation-dir", default="data/features/foundation")
    parser.add_argument("--feature-batch-size", type=int, default=256)
    parser.add_argument("--cnn-batch-size", type=int, default=32)
    parser.add_argument("--pilot", action="store_true", help="At most two epochs, separate pilot artifact namespace")
    args = parser.parse_args()
    if min(args.feature_batch_size, args.cnn_batch_size) < 1 or len(set(args.methods)) != len(args.methods):
        parser.error("Positive batch sizes and distinct methods are required")
    _safe_gpu()
    research = json.loads((PROJECT / "configs/research.json").read_text())
    seeds = args.seeds or ([17] if args.pilot else research["seeds"])
    if len(set(seeds)) != len(seeds):
        parser.error("Seeds must be distinct")
    code_hash = source_fingerprint(["scripts/train_detectors.py", "src/uac/models.py", "src/uac/training_v2.py"])
    namespace = "detector_pilot" if args.pilot else "detectors"
    index_path = PROJECT / "manifests" / ("detector_pilot_index.json" if args.pilot else "detector_index.json")
    selection_path = PROJECT / "manifests" / ("detector_pilot_selection.json" if args.pilot else "development_selection.json")
    with RunLock("runs/training.lock"):
        ledger = EventLedger()
        needs_real = any(method != "foundation" for method in args.methods)
        real_train = load_split_tensors("train", args.real_dir, include_audio="cnn" in args.methods) if needs_real else None
        real_dev = load_split_tensors("development", args.real_dir, include_audio="cnn" in args.methods) if needs_real else None
        foundation_train = load_split_tensors("train", args.foundation_dir) if "foundation" in args.methods else None
        foundation_dev = load_split_tensors("development", args.foundation_dir) if "foundation" in args.methods else None
        available = [x for x in (real_train, real_dev, foundation_train, foundation_dev) if x is not None]
        cohorts = {x["index"]["data_fingerprint"] for x in available}
        if len(cohorts) != 1:
            raise RuntimeError("Train/development representation caches have different cohort fingerprints")
        data_fingerprint = next(iter(cohorts))
        if real_train is not None and foundation_train is not None:
            if real_train["ids"] != foundation_train["ids"] or real_dev["ids"] != foundation_dev["ids"]:
                raise RuntimeError("Foundation and acoustic baselines do not share identical cohort ordering")
        if index_path.exists():
            index = json.loads(index_path.read_text())
            if index.get("data_fingerprint") != data_fingerprint:
                raise RuntimeError("Existing detector index is for a different cohort; do not overwrite")
            records = index["records"]
        else:
            records = []
        for method in args.methods:
            is_foundation = method == "foundation"
            train, dev = (foundation_train, foundation_dev) if is_foundation else (real_train, real_dev)
            kind = "foundation" if is_foundation else ("cnn" if method == "cnn" else "feature")
            indices = list(range(1536)) if is_foundation else (FEATURE_INDICES[method] if kind == "feature" else [])
            input_dim = len(indices) if kind != "cnn" else train["audio"].shape[-1]
            train_x = train["audio"] if kind == "cnn" else train["features"][:, indices]
            dev_x = dev["audio"] if kind == "cnn" else dev["features"][:, indices]
            training_data_hash = fingerprint({"cohort": data_fingerprint, "train_index_sha256": train["index_sha256"],
                                              "development_index_sha256": dev["index_sha256"]})
            for seed in seeds:
                seed_all(seed)
                if kind == "cnn":
                    model = SpectralDetector(sample_rate=16000, channels=32).cuda()
                    epochs, patience, batch_size, lr = research["cnn_epochs"], research["cnn_patience"], args.cnn_batch_size, 3e-4
                else:
                    model = FeatureDetector(input_dim).cuda()
                    model.fit_scaling(train_x)
                    epochs, patience, batch_size, lr = research["feature_epochs"], research["feature_patience"], args.feature_batch_size, 1e-3
                if args.pilot:
                    epochs, patience = min(epochs, 2), min(patience, 2)
                name = f"{namespace}/{method}_seed{seed}"
                config = {"method": method, "kind": kind, "feature_indices": indices, "input_dim": input_dim,
                          "sample_rate": 16000, "channels": 32 if kind == "cnn" else None,
                          "width": 128 if kind != "cnn" else None, "script_code_fingerprint": code_hash,
                          "cohort_data_fingerprint": data_fingerprint,
                          "train_index_sha256": train["index_sha256"], "development_index_sha256": dev["index_sha256"],
                          "pilot": args.pilot, "fit_partitions": ["train"], "selection_partition": "development"}
                ledger.append("detector_training_started", method=method, seed=seed, kind=kind, data_fingerprint=data_fingerprint)
                summary = train_supervised(model, train_x, train["labels"], dev_x, dev["labels"],
                                           name=name, config=config, data_hash=training_data_hash,
                                           epochs=epochs, batch_size=batch_size, lr=lr, patience=patience, seed=seed)
                checkpoint = f"checkpoints/{name}/best.pt"
                best = torch.load(project_path(checkpoint), map_location="cpu", weights_only=False)
                record = {"method": method, "seed": seed, "kind": kind, "feature_indices": indices,
                          "input_dim": input_dim, "checkpoint": checkpoint,
                          "checkpoint_sha256": sha256_file(checkpoint), "dev_ap": float(best["dev_ap"]),
                          "config": best["config"], "data_fingerprint": data_fingerprint,
                          "training_data_hash": training_data_hash, "completed_utc": utc_now(),
                          "epochs_completed": len(summary["history"]), "pilot": args.pilot,
                          "primary_eligible": method != "hht_exploratory"}
                records = [row for row in records if (row["method"], row["seed"]) != (method, seed)] + [record]
                records.sort(key=lambda row: (row["method"], row["seed"]))
                atomic_json(index_path, {"schema_version": 1, "data_fingerprint": data_fingerprint,
                                        "records": records, "updated_utc": utc_now(), "test_examined": False,
                                        "feature_group_slices": {"fourier": [0, 32], "stft": [32, 96], "wavelet": [96, 152], "hht": [152, 184]}})
                update_selection(records, research["seeds"] if not args.pilot else seeds, selection_path, data_fingerprint, args.pilot)
                ledger.append("detector_training_complete", method=method, seed=seed, dev_ap=record["dev_ap"], checkpoint=checkpoint)
                print(json.dumps({"method": method, "seed": seed, "dev_ap": record["dev_ap"],
                                  "epochs": record["epochs_completed"], "checkpoint": checkpoint}), flush=True)
                del model, best
                torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as exc:
        print(json.dumps({"status": "error", "error": str(exc), "cpu_fallback": False}), file=sys.stderr)
        raise SystemExit(2)
