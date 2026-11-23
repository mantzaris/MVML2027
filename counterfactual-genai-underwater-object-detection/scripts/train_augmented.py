#!/usr/bin/env python3
"""B4: real plus matched-volume synthetic training, real-only development.

Use every real-training scene and all 20,000 controlled synthetic training scenes.
Synthetic volume matches configured full flow training (5,000 steps x batch 4),
not identical draws. Cap maximum optimizer updates approximately at the real-only
150-epoch feature baseline by reducing mixed-data epochs. Actual updates and
selection passes are recorded; equal realized compute is not claimed. No
synthetic development/test or real calibration/test examples are read.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
import torch
from uac.models import FeatureDetector
from uac.training_v2 import seed_all, train_supervised
from uac.runtime import RunLock, EventLedger, atomic_json, fingerprint, project_path, sha256_file, source_fingerprint, utc_now
from extract_real_features import _safe_gpu
from train_detectors import load_split_tensors, PRIMARY_METHODS


def update_ranking(records, seeds, data_fingerprint):
    methods = tuple(PRIMARY_METHODS) + ("augmentation",)
    ranked, incomplete = [], []
    for method in methods:
        chosen = [row for row in records if row["method"] == method and row["seed"] in seeds and row.get("primary_eligible", True)]
        available = {row["seed"] for row in chosen}
        if available != set(seeds):
            incomplete.append({"method": method, "missing_seeds": sorted(set(seeds) - available)})
            continue
        ranked.append({"method": method, "seed_mean_development_ap": sum(row["dev_ap"] for row in chosen) / len(chosen),
                       "seeds": sorted(available), "per_seed_ap": {str(row["seed"]): row["dev_ap"] for row in chosen}})
    ranked.sort(key=lambda row: (-row["seed_mean_development_ap"], row["method"]))
    atomic_json("manifests/development_selection.json", {"schema_version": 1,
        "status": "development_complete" if not incomplete else "partial",
        "selection_basis": "highest seed-mean real-development AP including synthetic augmentation; exploratory HHT excluded; lexical tie break",
        "data_fingerprint": data_fingerprint, "requested_seeds": seeds,
        "ranked_complete_methods": ranked, "incomplete_methods": incomplete,
        "provisional_best_method": ranked[0]["method"] if ranked else None,
        "locked_comparator": None, "test_examined": False, "updated_utc": utc_now(),
        "note": "Development ranking only. Final parent freeze must consider all primary-eligible detector-index records."})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-dir", default="data/features/real")
    parser.add_argument("--synthetic-dir", default="data/features/synthetic")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 29, 43])
    args = parser.parse_args()
    if args.batch_size < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Positive batch size and distinct seeds are required")
    if project_path("runs/test_freeze.json").is_file():
        raise RuntimeError("Augmentation baseline fitting is locked after test freeze")
    _safe_gpu()
    research = json.loads(project_path("configs/research.json").read_text())
    config_hash = sha256_file("configs/synthetic_features.json")
    code_hash = source_fingerprint(["scripts/train_augmented.py", "src/uac/models.py", "src/uac/training_v2.py"])
    indices = list(range(152))
    with RunLock("runs/training.lock"):
        real_train = load_split_tensors("train", args.real_dir)
        real_dev = load_split_tensors("development", args.real_dir)
        synthetic_train = load_split_tensors("train", args.synthetic_dir)
        if real_train["index"]["data_fingerprint"] != real_dev["index"]["data_fingerprint"]:
            raise RuntimeError("Real train/development cohort fingerprints disagree")
        if synthetic_train["index"]["data_fingerprint"] != config_hash:
            raise RuntimeError("Synthetic training cache is not from the frozen synthetic feature configuration")
        if any(row.get("origin") != "fully_synthetic" or row.get("split") != "train" for row in synthetic_train["rows"]):
            raise RuntimeError("Augmentation may use only fully synthetic training rows")
        if any(row.get("split") != "train" for row in real_train["rows"]) or any(row.get("split") != "development" for row in real_dev["rows"]):
            raise RuntimeError("Real fitting or selection partition mismatch")
        if set(real_train["ids"]) & set(real_dev["ids"]) or set(real_train["ids"]) & set(synthetic_train["ids"]):
            raise RuntimeError("Duplicate or overlapping real/synthetic scene identities")
        n_real, n_synthetic = len(real_train["labels"]), len(synthetic_train["labels"])
        flow_volume = research["generator_steps"] * research["generator_batch_size"]
        if n_synthetic != 20000 or n_synthetic != flow_volume:
            raise RuntimeError("Matched-volume baseline requires all20,000 scenes and matching configured full flow volume")
        maximum_real_epochs = research["feature_epochs"]
        n_total = n_real + n_synthetic
        epochs = math.ceil(maximum_real_epochs * n_real / n_total)
        # Preserve roughly the same number of stale optimizer updates permitted
        # by real-only patience; a mixed epoch contains more optimizer steps.
        patience = max(2, math.ceil(research["feature_patience"] * n_real / n_total))
        real_steps_per_epoch = math.ceil(n_real / args.batch_size)
        mixed_steps_per_epoch = math.ceil(n_total / args.batch_size)
        maximum_reference_updates = maximum_real_epochs * real_steps_per_epoch
        maximum_augmented_updates = epochs * mixed_steps_per_epoch
        train_x = torch.cat((real_train["features"][:, indices], synthetic_train["features"][:, indices]))
        train_y = torch.cat((real_train["labels"], synthetic_train["labels"]))
        dev_x, dev_y = real_dev["features"][:, indices], real_dev["labels"]
        real_fingerprint = real_train["index"]["data_fingerprint"]
        data_hash = fingerprint({"real_cohort": real_fingerprint,
                                "real_train_index_sha256": real_train["index_sha256"],
                                "real_development_index_sha256": real_dev["index_sha256"],
                                "synthetic_train_index_sha256": synthetic_train["index_sha256"],
                                "synthetic_features_config_sha256": config_hash})
        index_path = project_path("manifests/detector_index.json")
        if index_path.is_file():
            index = json.loads(index_path.read_text())
            if index["data_fingerprint"] != real_fingerprint:
                raise RuntimeError("Existing detector index belongs to another real cohort")
            records = index["records"]
        else:
            records = []
        settings = {"method": "augmentation", "kind": "feature", "feature_indices": indices,
                    "input_dim": 152, "width": 128, "script_code_fingerprint": code_hash,
                    "real_cohort_data_fingerprint": real_fingerprint,
                    "real_train_index_sha256": real_train["index_sha256"],
                    "real_development_index_sha256": real_dev["index_sha256"],
                    "synthetic_train_index_sha256": synthetic_train["index_sha256"],
                    "synthetic_features_config_sha256": config_hash,
                    "synthetic_seed_namespace": synthetic_train["index"]["config"]["namespaces"]["train"],
                    "n_real_training": n_real, "n_synthetic_training": n_synthetic,
                    "configured_flow_training_scenes": flow_volume, "draw_identity_matches_flow": False,
                    "maximum_real_reference_epochs": maximum_real_epochs,
                    "maximum_real_reference_updates": maximum_reference_updates,
                    "maximum_augmented_epochs": epochs, "maximum_augmented_updates": maximum_augmented_updates,
                    "update_budget_ratio": maximum_augmented_updates / maximum_reference_updates,
                    "patience_epochs": patience, "patience_basis": "ceil(real patience epochs times n_real/n_total), minimum2",
                    "sampling": "uniform mixed-row permutation each epoch; no class reweighting",
                    "training_positive_fraction": float(train_y.mean()),
                    "development_source": "real development only", "primary_eligible": True,
                    "limitation": "Maximum-update matching is approximate; actual early stopping and development-check frequency can differ"}
        ledger = EventLedger()
        for seed in args.seeds:
            seed_all(seed)
            model = FeatureDetector(152).cuda()
            model.fit_scaling(train_x)
            name = f"detectors/augmentation_seed{seed}"
            summary = train_supervised(model, train_x, train_y, dev_x, dev_y, name=name,
                                       config=settings, data_hash=data_hash, epochs=epochs,
                                       batch_size=args.batch_size, lr=1e-3, patience=patience, seed=seed)
            checkpoint = f"checkpoints/{name}/best.pt"
            best = torch.load(project_path(checkpoint), map_location="cpu", weights_only=False)
            actual_epochs = len(summary["history"])
            actual_updates = actual_epochs * mixed_steps_per_epoch
            latest = json.loads(project_path(f"checkpoints/{name}/resume/latest.json").read_text())
            if latest["entries"][0]["global_step"] != actual_updates:
                raise RuntimeError("Augmented training checkpoint update count differs from completed-epoch accounting")
            record = {"method": "augmentation", "seed": seed, "kind": "feature", "feature_indices": indices,
                      "input_dim": 152, "checkpoint": checkpoint, "checkpoint_sha256": sha256_file(checkpoint),
                      "dev_ap": float(best["dev_ap"]), "config": best["config"],
                      "data_fingerprint": real_fingerprint, "training_data_hash": data_hash,
                      "completed_utc": utc_now(), "epochs_completed": actual_epochs,
                      "optimizer_updates": actual_updates, "maximum_optimizer_updates": maximum_augmented_updates,
                      "early_stopped": actual_epochs < epochs, "pilot": False, "primary_eligible": True}
            records = [row for row in records if (row["method"], row["seed"]) != ("augmentation", seed)] + [record]
            records.sort(key=lambda row: (row["method"], row["seed"]))
            atomic_json(index_path, {"schema_version": 1, "data_fingerprint": real_fingerprint,
                                    "records": records, "updated_utc": utc_now(), "test_examined": False,
                                    "feature_group_slices": {"fourier": [0, 32], "stft": [32, 96], "wavelet": [96, 152], "hht": [152, 184]}})
            update_ranking(records, research["seeds"], real_fingerprint)
            ledger.append("augmentation_baseline_complete", seed=seed, dev_ap=record["dev_ap"],
                          optimizer_updates=actual_updates, checkpoint=checkpoint)
            print(json.dumps({"method": "augmentation", "seed": seed, "dev_ap": record["dev_ap"],
                              "actual_updates": actual_updates, "maximum_updates": maximum_augmented_updates,
                              "reference_maximum_updates": maximum_reference_updates,
                              "epochs": actual_epochs, "checkpoint": checkpoint}), flush=True)
            del model, best
        atomic_json("manifests/augmentation_protocol.json", {"settings": settings, "training_data_fingerprint": data_hash,
                    "records": [row for row in records if row["method"] == "augmentation"],
                    "statistical_selection": "real-development AP; no held-out synthetic or real outcomes accessed"})
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as exc:
        print(json.dumps({"status": "error", "error": str(exc), "cpu_fallback": False}), file=sys.stderr)
        raise SystemExit(2)
