#!/usr/bin/env python3
"""Focused, frozen-protocol synthetic representation study for the supplement.

Two stages: train on train/development only, then evaluate both held-out simulator
partitions after a freeze containing every model checkpoint digest. Learning,
feature extraction and inference are CUDA; CPU statistics and durable I/O are
permitted. Presence AP is not temporal AP and the synthetic source types are not
vessel classes. No synthetic outcome is a substitute for the main real study.
"""
from __future__ import annotations
import argparse
import io
import json
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
import numpy as np
import torch
from uac.dsp import extract_features
from uac.models import FeatureDetector
from uac.synthetic import generate_batch
from uac.training import train_supervised, seed_all
from uac.metrics import binary_average_precision, paired_group_bootstrap
from uac.runtime import (RunLock, StopRequested, EventLedger, atomic_bytes, atomic_json,
                         fingerprint, project_path, sha256_file, source_fingerprint, utc_now)
from extract_real_features import _safe_gpu
from extract_foundation import load_index, load_shard

SLICES = {"fourier": (0, 32), "stft": (32, 96), "wavelet": (96, 152), "hht": (152, 184)}
MODEL_INDEX = "manifests/supplemental_model_index.json"
COST_PATH = "manifests/supplemental_feature_cost.json"
RESULT_PATH = "results/tables/supplemental_views.json"


def indices_for(groups):
    return [i for group in groups for i in range(*SLICES[group])]


def load_controlled_split(split, config):
    index, index_hash = load_index(split, config["cache_root"])
    parts, targets, rows, ids = [], [], [], []
    for record in index["shards"]:
        shard = load_shard(record)
        parts.append(shard["features"])
        targets.append(shard["labels"])
        rows.extend(shard["rows"])
        ids.extend(shard["ids"])
    x = torch.cat(parts).to(device="cuda", dtype=torch.float32)
    y = torch.cat(targets).to(device="cuda", dtype=torch.float32)
    if x.shape != (config["expected_scene_counts"][split], 184) or len(ids) != len(set(ids)):
        raise RuntimeError("Unexpected synthetic cache dimensions or repeated scene identities")
    if len(rows) != len(y) or any(row.get("origin") != "fully_synthetic" for row in rows):
        raise RuntimeError("Supplement requires fully synthetic metadata for every scene")
    if not bool(torch.isfinite(x).all()) or not bool(((y == 0) | (y == 1)).all()):
        raise RuntimeError("Invalid synthetic feature values or binary labels")
    counts = np.asarray([row["source_count"] for row in rows], dtype=np.int64)
    types = np.asarray([row["source_types"] for row in rows], dtype=np.int64)
    if types.shape != (len(rows), 2) or not np.array_equal((counts > 0).astype(float), y.cpu().numpy()):
        raise RuntimeError("Synthetic source metadata is inconsistent with presence labels")
    family_values = {row["family"] for row in rows}
    if len(family_values) != 1:
        raise RuntimeError("A supplemental evaluation partition must have one declared simulator family")
    return {"x": x, "y": y, "ids": ids, "rows": rows, "source_count": counts,
            "source_types": types, "family": next(iter(family_values)),
            "index": index, "index_sha256": index_hash}


def measure_feature_cost(config):
    settings = config["feature_timing"]
    source_hash = source_fingerprint(["src/uac/dsp.py", "src/uac/synthetic.py", "scripts/supplemental_views.py"])
    expected = {"settings": settings, "source_fingerprint": source_hash,
                "methods": config["methods"], "samples": config["num_samples"]}
    path = project_path(COST_PATH)
    if path.is_file():
        record = json.loads(path.read_text())
        if record.get("fingerprint") == fingerprint(expected):
            return record
    batch = generate_batch(settings["batch_size"], config["num_samples"], config["sample_rate"],
                           seed=settings["seed"], family=settings["family"])
    costs = {}
    with torch.inference_mode():
        for method, groups in config["methods"].items():
            for _ in range(settings["warmup_calls"]):
                extract_features(batch["audio"], config["sample_rate"], groups=groups)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            samples = []
            for _ in range(settings["measured_calls"]):
                start = time.perf_counter()
                extract_features(batch["audio"], config["sample_rate"], groups=groups)
                torch.cuda.synchronize()
                samples.append(1000 * (time.perf_counter() - start) / settings["batch_size"])
            costs[method] = {"feature_cost_ms": float(np.median(samples)),
                             "p10_ms": float(np.quantile(samples, .1)), "p90_ms": float(np.quantile(samples, .9)),
                             "measured_calls": len(samples), "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 1024 ** 2}
    record = {"schema_version": 1, "fingerprint": fingerprint(expected), **expected,
              "costs": costs, "recorded_utc": utc_now(), "scope": settings["scope"]}
    atomic_json(path, record)
    return record


def train(config, config_path):
    if project_path("runs/test_freeze.json").is_file():
        raise RuntimeError("Supplemental model fitting is locked after test freeze")
    train_data = load_controlled_split(config["training_split"], config)
    dev_data = load_controlled_split(config["development_split"], config)
    if set(train_data["ids"]) & set(dev_data["ids"]):
        raise RuntimeError("Synthetic train/development scene identities overlap")
    cache_config_hash = sha256_file(config["cache_config"])
    config_hash = sha256_file(config_path)
    code_hash = source_fingerprint(["scripts/supplemental_views.py", "src/uac/models.py", "src/uac/training.py"])
    data_hash = fingerprint({"synthetic_config_sha256": cache_config_hash,
                             "train_index_sha256": train_data["index_sha256"],
                             "development_index_sha256": dev_data["index_sha256"]})
    path = project_path(MODEL_INDEX)
    if path.is_file():
        index = json.loads(path.read_text())
        if index["training_data_fingerprint"] != data_hash or index["config_sha256"] != config_hash or index["code_fingerprint"] != code_hash:
            raise RuntimeError("Existing supplemental model index is incompatible; preserve it and use a new protocol namespace")
        records = index["records"]
    else:
        records = []
    cost = measure_feature_cost(config)
    ledger = EventLedger()
    for method, groups in config["methods"].items():
        indices = indices_for(groups)
        for seed in config["seeds"]:
            old = [record for record in records if record["method"] == method and record["seed"] == seed]
            if old and project_path(old[0]["checkpoint"]).is_file() and sha256_file(old[0]["checkpoint"]) == old[0]["checkpoint_sha256"]:
                print(json.dumps({"stage": "already_trained", "method": method, "seed": seed}), flush=True)
                continue
            seed_all(seed)
            model = FeatureDetector(len(indices), width=config["model_width"]).cuda()
            train_x = train_data["x"][:, indices]
            dev_x = dev_data["x"][:, indices]
            model.fit_scaling(train_x)
            name = f"supplemental/{method}_seed{seed}"
            settings = {"method": method, "groups": groups, "feature_indices": indices,
                        "input_dim": len(indices), "width": config["model_width"],
                        "config_sha256": config_hash, "script_code_fingerprint": code_hash,
                        "synthetic_features_config_sha256": cache_config_hash,
                        "primary_or_exploratory": "exploratory_HHT" if "hht" in groups else "validated_transforms"}
            summary = train_supervised(model, train_x, train_data["y"], dev_x, dev_data["y"], name=name,
                                       config=settings, data_hash=data_hash, epochs=config["epochs"],
                                       batch_size=config["batch_size"], lr=config["learning_rate"],
                                       patience=config["patience"], seed=seed)
            checkpoint = f"checkpoints/{name}/best.pt"
            best = torch.load(project_path(checkpoint), map_location="cpu", weights_only=False)
            record = {"method": method, "seed": seed, "groups": groups, "feature_indices": indices,
                      "input_dim": len(indices), "checkpoint": checkpoint, "checkpoint_sha256": sha256_file(checkpoint),
                      "development_ap": float(best["dev_ap"]), "epochs_completed": len(summary["history"]),
                      "config": best["config"], "hht_status": config["hht_status"] if "hht" in groups else "not_used"}
            records = [r for r in records if (r["method"], r["seed"]) != (method, seed)] + [record]
            records.sort(key=lambda r: (r["method"], r["seed"]))
            index = {"schema_version": 1, "status": "training", "config_sha256": config_hash,
                     "synthetic_features_config_sha256": cache_config_hash,
                     "training_data_fingerprint": data_hash, "code_fingerprint": code_hash,
                     "train_index_sha256": train_data["index_sha256"], "development_index_sha256": dev_data["index_sha256"],
                     "feature_cost_sha256": sha256_file(COST_PATH), "records": records, "test_examined": False}
            atomic_json(path, index)
            ledger.append("supplemental_model_trained", method=method, seed=seed, development_ap=record["development_ap"], checkpoint=checkpoint)
            print(json.dumps({"stage": "trained", "method": method, "seed": seed, "development_ap": record["development_ap"]}), flush=True)
            del model, train_x, dev_x, best
    expected = {(method, seed) for method in config["methods"] for seed in config["seeds"]}
    if {(record["method"], record["seed"]) for record in records} != expected:
        raise RuntimeError("Supplemental model grid is incomplete")
    # Keep the completed index immutable on healthy re-entry, so its frozen hash
    # is stable. An unfinished run commits completion only after all checkpoints.
    current = json.loads(path.read_text())
    if current.get("status") != "complete":
        current["status"] = "complete"
        current["completed_utc"] = utc_now()
        atomic_json(path, current)
    print(json.dumps({"stage": "training_complete", "models": len(records),
                      "model_index": MODEL_INDEX, "freeze_artifacts_required": [str(config_path.relative_to(PROJECT)),
                          "scripts/supplemental_views.py", MODEL_INDEX, COST_PATH] + [record["checkpoint"] for record in records]}), flush=True)


def _require_frozen_artifacts(config, config_path):
    freeze_path = project_path("runs/test_freeze.json")
    if not freeze_path.is_file():
        raise RuntimeError("Supplemental held-out evaluation requires runs/test_freeze.json")
    freeze = json.loads(freeze_path.read_text())
    cache_hash = sha256_file(config["cache_config"])
    if freeze.get("synthetic_features_config_sha256") != cache_hash:
        raise RuntimeError("Synthetic feature configuration is not the frozen one")
    index = json.loads(project_path(MODEL_INDEX).read_text())
    if index.get("status") != "complete" or index["config_sha256"] != sha256_file(config_path) or index["synthetic_features_config_sha256"] != cache_hash:
        raise RuntimeError("Supplemental model index is incomplete or incompatible")
    current_code = source_fingerprint(["scripts/supplemental_views.py", "src/uac/models.py", "src/uac/training.py"])
    if index["code_fingerprint"] != current_code:
        raise RuntimeError("Supplemental fitting/inference code changed after model fitting")
    artifacts = freeze.get("artifacts", {})
    required = [str(config_path.relative_to(PROJECT)), "scripts/supplemental_views.py", MODEL_INDEX, COST_PATH]
    required += [record["checkpoint"] for record in index["records"]]
    for name in required:
        entry = artifacts.get(name)
        expected = entry.get("sha256") if isinstance(entry, dict) else entry
        if expected is None or expected != sha256_file(name):
            raise RuntimeError(f"Required supplemental artifact absent or changed after freeze: {name}")
    expected_grid = {(method, seed) for method in config["methods"] for seed in config["seeds"]}
    if {(row["method"], row["seed"]) for row in index["records"]} != expected_grid:
        raise RuntimeError("Frozen model index is missing required supplemental methods/seeds")
    return index, sha256_file(freeze_path)


def _regimes(data):
    negative = data["source_count"] == 0
    for name, source_type in (("harmonic_am", 0), ("chirp_dominant", 1), ("decaying_am", 2)):
        positive = (data["source_count"] == 1) & (data["source_types"][:, 0] == source_type)
        yield name, positive | negative
    yield "overlapping_sources", (data["source_count"] == 2) | negative


def _predict(model, x, batch_size):
    with torch.inference_mode():
        result = torch.cat([model(x[start:start + batch_size]) for start in range(0, len(x), batch_size)])
    if not bool(torch.isfinite(result).all()):
        raise RuntimeError("Nonfinite supplemental prediction")
    return result


def _numerics(data):
    x = data["x"]
    return {"n_scenes": len(x), "mean_flagged_mode_fraction": float(x[:, 181].mean()),
            "scene_fraction_with_any_flagged_mode": float((x[:, 181] > 0).float().mean()),
            "mean_valid_if_fraction": float(x[:, [155, 159, 163, 167, 171, 175]].mean()),
            "max_reconstruction_relative_rms": float(x[:, 178].max()),
            "mean_extracted_mode_fraction": float(x[:, 179].mean()),
            "interpretation": "Bounded-EMD diagnostics; good reconstruction does not demonstrate converged physical IMFs"}


def evaluate(config, config_path):
    model_index, freeze_hash = _require_frozen_artifacts(config, config_path)
    cost = json.loads(project_path(COST_PATH).read_text())
    ledger = EventLedger()
    output = {"schema_version": 1, "status": "running", "scope": config["scope"],
              "metric": "binary clip-presence average precision from raw-logit ranking; not temporal AP",
              "view_ablation": [], "regime_ablation": [], "missing_view": [], "comparisons": [], "hht_numerics": {},
              "provenance": {"config_sha256": sha256_file(config_path), "model_index_sha256": sha256_file(MODEL_INDEX),
                             "freeze_sha256": freeze_hash, "feature_cost_sha256": sha256_file(COST_PATH),
                             "synthetic_features_config_sha256": sha256_file(config["cache_config"]), "evaluation_indices": {}},
              "limitations": ["Fully synthetic presence classification only", "Synthetic source types are not vessel classes",
                  "Background scenes are reused across regime diagnostics; strata are not independent",
                  "AP prevalence differs across regimes; compare methods within a regime",
                  "HHT uses a bounded approximate decomposition with reported nonconvergence",
                  "Missing-view replacement is a diagnostic distribution shift, not a missingness-trained system",
                  "Feature timing excludes transfer, classifier inference, generation and language-agent cost"],
              "regime_definitions": config["regimes"], "feature_cost_scope": config["feature_timing"]["scope"]}
    with StopRequested() as stop:
        for split in config["evaluation_splits"]:
            data = load_controlled_split(split, config)
            labels = data["y"].cpu().numpy()
            output["provenance"]["evaluation_indices"][split] = data["index_sha256"]
            output["hht_numerics"][split] = _numerics(data)
            scores_by_model = {}
            for record in model_index["records"]:
                if stop.requested:
                    output["status"] = "paused"
                    atomic_json(RESULT_PATH, output)
                    return 75
                method, seed = record["method"], record["seed"]
                raw_path = project_path(f"results/raw/supplemental/{split}/{method}_seed{seed}.pt")
                receipt_path = raw_path.with_suffix(".json")
                key = fingerprint({"freeze_sha256": freeze_hash, "evaluation_index_sha256": data["index_sha256"],
                                   "checkpoint_sha256": record["checkpoint_sha256"], "config_sha256": sha256_file(config_path)})
                cached = None
                if receipt_path.is_file() and raw_path.is_file():
                    receipt = json.loads(receipt_path.read_text())
                    if receipt.get("key") == key and receipt.get("payload_sha256") == sha256_file(raw_path):
                        cached = torch.load(raw_path, map_location="cpu", weights_only=False)
                        if cached["ids"] != data["ids"]:
                            raise RuntimeError("Supplemental cached prediction identities differ")
                if cached is None:
                    if sha256_file(record["checkpoint"]) != record["checkpoint_sha256"]:
                        raise RuntimeError("Supplemental checkpoint checksum mismatch")
                    model = FeatureDetector(record["input_dim"], width=config["model_width"]).cuda()
                    payload = torch.load(project_path(record["checkpoint"]), map_location="cpu", weights_only=False)
                    model.load_state_dict(payload["model"])
                    model.eval()
                    x = data["x"][:, record["feature_indices"]]
                    clean = _predict(model, x, config["batch_size"])
                    corruptions = {}
                    for removed in record["groups"]:
                        lower, upper = SLICES[removed]
                        columns = [j for j, original in enumerate(record["feature_indices"]) if lower <= original < upper]
                        edited = x.clone()
                        edited[:, columns] = model.mean[columns]
                        corruptions[removed] = _predict(model, edited, config["batch_size"]).cpu()
                    cached = {"ids": data["ids"], "labels": data["y"].cpu(), "logits": clean.cpu(),
                              "missing_view_logits": corruptions, "method": method, "seed": seed, "split": split,
                              "family": data["family"], "key": key}
                    buffer = io.BytesIO()
                    torch.save(cached, buffer)
                    atomic_bytes(raw_path, buffer.getvalue())
                    atomic_json(receipt_path, {"key": key, "payload_sha256": sha256_file(raw_path),
                                              "payload": str(raw_path.relative_to(PROJECT)), "completed_utc": utc_now()})
                    del model, payload, x
                scores = cached["logits"].numpy()
                scores_by_model[(method, seed)] = scores
                base = {"method": method, "seed": seed, "split": split, "family": data["family"],
                        "origin": "fully_synthetic", "clip_duration_seconds": config["num_samples"] / config["sample_rate"],
                        "scene_manifest_sha256": data["index_sha256"],
                        "source_seed_namespace": data["index"]["config"]["namespaces"][split],
                        "n_scenes": len(labels), "n_positive": int(labels.sum()),
                        "n_negative": int((labels == 0).sum())}
                output["view_ablation"].append({**base, "ap": binary_average_precision(labels, scores),
                                                "feature_cost_ms": cost["costs"][method]["feature_cost_ms"],
                                                "timing_mode": "live_tensor_resident",
                                                "hht_status": record["hht_status"]})
                for name, selected in _regimes(data):
                    y = labels[selected]
                    output["regime_ablation"].append({**base, "regime": name, "n_scenes": int(selected.sum()),
                        "n_positive": int(y.sum()), "n_negative": int((y == 0).sum()),
                        "ap": binary_average_precision(y, scores[selected]) if len(y) and y.sum() > 0 else None,
                        "background_reuse": "all same-partition background scenes"})
                for removed, logits in cached["missing_view_logits"].items():
                    output["missing_view"].append({**base, "removed_view": removed,
                        "corruption": "train_mean_feature_imputation", "ap": binary_average_precision(labels, logits.numpy()),
                        "clean_ap": binary_average_precision(labels, scores), "retrained": False})
                atomic_json(RESULT_PATH, output)
                ledger.append("supplemental_evaluation_committed", split=split, method=method, seed=seed,
                              ap=output["view_ablation"][-1]["ap"], prediction_path=str(raw_path.relative_to(PROJECT)))
                print(json.dumps(output["view_ablation"][-1]), flush=True)
            for seed in config["seeds"]:
                reference = scores_by_model[("stft", seed)]
                for method in config["methods"]:
                    if method == "stft":
                        continue
                    interval = paired_group_bootstrap(labels, scores_by_model[(method, seed)], reference, data["ids"],
                        n_bootstrap=config["paired_bootstrap_replicates"], seed=config["paired_bootstrap_seed"] + seed)
                    output["comparisons"].append({"method": method, "reference": "stft", "seed": seed,
                        "split": split, "family": data["family"], **interval,
                        "scope": "Conditional on this simulator and scene-sampling distribution; exploratory intervals, no multiplicity-adjusted p-values"})
            atomic_json(RESULT_PATH, output)
    output["status"] = "complete"
    output["completed_utc"] = utc_now()
    atomic_json(RESULT_PATH, output)
    ledger.append("supplemental_views_complete", output=RESULT_PATH, comparisons=len(output["view_ablation"]))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("train", "evaluate"), required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/supplemental.json"))
    args = parser.parse_args()
    config_path = project_path(args.config)
    config = json.loads(config_path.read_text())
    if config["seeds"] != [17, 29, 43] or config["model_width"] != 128:
        raise RuntimeError("Unexpected supplemental model grid; amend and freeze a new protocol explicitly")
    _safe_gpu()
    with RunLock("runs/training.lock"):
        return train(config, config_path) if args.stage == "train" else evaluate(config, config_path)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as exc:
        print(json.dumps({"status": "error", "error": str(exc), "cpu_fallback": False}), file=sys.stderr)
        raise SystemExit(2)
