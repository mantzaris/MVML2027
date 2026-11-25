#!/usr/bin/env python3
"""Replay frozen real-flow classifiers on all held-out capture/stress cases.

Recomputes seed-37 SCR_FLOW, GMAE and DAE with the original 256-window batch
shape, frozen training-fitted preprocessing and deterministic corruptions.
Checks stored probabilities, paired row metadata, average precision and Brier.
This is an audit only: no training, refitting, threshold selection or tuning.
"""
import csv
import json
from pathlib import Path

import numpy as np
import torch

from measure_real_pipeline import (ROOT, SEED, PROBABILITY_TOLERANCE, completed_manifest,
                                   configure_cuda, frozen_preprocessing, load_model, predictions)
from real_data import CONDITIONS, TEST, condition_batch, input_hash, load_capture, sha, subset, windows
from real_metrics import metrics

METHODS = ["SCR_FLOW", "GMAE", "DAE"]
METRIC_TOLERANCE = 1e-8


@torch.inference_mode()
def collect(model, batch, batch_size):
    return torch.cat([predictions(model, subset(batch, ids))
                      for ids in torch.arange(len(batch["x"]), device="cuda").split(batch_size)])


@torch.inference_mode()
def main():
    manifest_path, manifest, data_manifest = completed_manifest()
    configure_cuda()
    preprocessing = frozen_preprocessing(manifest)
    models = {method: load_model(method, manifest) for method in METHODS}
    threshold = {record["method"]: float(record["threshold"]) for record in manifest["thresholds"]
                 if int(record["seed"]) == SEED}
    summary_path = ROOT / "results/real/summary.csv"
    with summary_path.open() as stream:
        records = list(csv.DictReader(stream))
    if len(records) != 675:
        raise AssertionError("Completed replay requires all 675 summary rows")
    summary = {(row["method"], row["condition"], int(row["scenario"])): row
               for row in records if int(row["seed"]) == SEED}
    rows = []
    for scenario in TEST:
        clean = windows(load_capture(scenario), scenario, preprocessing)
        for condition in CONDITIONS:
            data = condition_batch(clean, condition, preprocessing)
            data_digest = input_hash(data)
            if data_digest != manifest["case_hashes"][f"{condition}_{scenario}"]:
                raise AssertionError(f"Reconstructed captured inputs changed: {condition}/{scenario}")
            valid = data["valid"]
            labels = data["y"][valid]
            expected_metadata = dict(labels=labels.cpu().numpy(), row_id=data["row_id"][valid].cpu().numpy(),
                window=data["window"][:, None].expand_as(valid)[valid].cpu().numpy(),
                block=data["block"][:, None].expand_as(valid)[valid].cpu().numpy())
            for method, (model, checkpoint_path, checkpoint_digest) in models.items():
                raw_path = ROOT / "results/runs/real" / str(SEED) / f"{method}_{condition}_{scenario}.npz"
                raw_digest = sha(raw_path)
                if raw_digest != manifest["raw_hashes"][str(raw_path.relative_to(ROOT))]:
                    raise AssertionError(f"Raw archive changed: {raw_path}")
                probability = collect(model, data, manifest["settings"]["batch"])[valid]
                replayed = probability.cpu().numpy()
                with np.load(raw_path, allow_pickle=False) as saved:
                    for key, value in expected_metadata.items():
                        if not np.array_equal(value, saved[key]):
                            raise AssertionError(f"Replay metadata mismatch: {raw_path}/{key}")
                    if replayed.shape != saved["prob"].shape:
                        raise AssertionError(f"Probability shape changed: {raw_path}")
                    probability_error = float(np.max(np.abs(replayed - saved["prob"])))
                    saved_probability = torch.from_numpy(saved["prob"].copy()).to("cuda")
                if not np.isfinite(probability_error) or probability_error > PROBABILITY_TOLERANCE:
                    raise AssertionError(f"Probability replay mismatch: {raw_path}: {probability_error}")
                current = metrics(labels, probability, threshold[method])
                original = metrics(labels, saved_probability, threshold[method])
                summary_row = summary[(method, condition, scenario)]
                errors = dict(prob=probability_error)
                for key in ["ap", "brier"]:
                    errors[key] = abs(current[key] - original[key])
                    summary_error = abs(original[key] - float(summary_row[key]))
                    if (not np.isfinite(errors[key]) or not np.isfinite(summary_error)
                            or errors[key] > METRIC_TOLERANCE or summary_error > METRIC_TOLERANCE):
                        raise AssertionError(f"Metric replay or raw/summary mismatch: {raw_path}/{key}")
                rows.append(dict(method=method, scenario=scenario, condition=condition, flows=int(labels.numel()),
                    max_absolute_errors=errors, replayed_ap=current["ap"], replayed_brier=current["brier"],
                    checkpoint_path=str(checkpoint_path.relative_to(ROOT)), checkpoint_sha256=checkpoint_digest,
                    raw_path=str(raw_path.relative_to(ROOT)), raw_sha256=raw_digest, input_data_sha256=data_digest,
                    paired_metadata_equal=True))
            del data
        del clean
    expected_flows = data_manifest["totals"]["test"]["rows_retained"] * len(CONDITIONS) * len(METHODS)
    flow_evaluations = sum(row["flows"] for row in rows)
    if len(rows) != 45 or flow_evaluations != expected_flows:
        raise AssertionError("Replay did not cover every expected model/capture/condition/flow")
    output = dict(passed=True, training_seed=SEED, methods=METHODS, test_scenarios=TEST, conditions=CONDITIONS,
        hardware=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda,
        precision="FP32 inference; TF32 allowed; deterministic algorithms; original batch size 256 windows",
        probability_tolerance=PROBABILITY_TOLERANCE, metric_tolerance=METRIC_TOLERANCE,
        model_condition_replays=len(rows), model_flow_evaluations=flow_evaluations,
        scope=__doc__.strip(), no_training_or_refitting=True,
        source_manifest_sha256=sha(manifest_path), data_manifest_sha256=manifest["data_manifest_sha256"],
        summary_sha256=sha(summary_path), replay_script_sha256=sha(Path(__file__)),
        source_hashes={str(path.relative_to(ROOT)): sha(path) for path in
            [Path(__file__), ROOT/"scripts/measure_real_pipeline.py", ROOT/"src/real_data.py",
             ROOT/"src/real_models.py", ROOT/"src/real_metrics.py"]},
        maximum_absolute_errors={key: max(row["max_absolute_errors"][key] for row in rows) for key in ["prob", "ap", "brier"]},
        rows=rows)
    path = ROOT / "results/real/replay.json"
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(dict(passed=True, replays=len(rows), model_flow_evaluations=flow_evaluations,
                         maximum_absolute_errors=output["maximum_absolute_errors"])))


if __name__ == "__main__":
    main()
