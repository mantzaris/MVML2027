#!/usr/bin/env python3
"""Recover frozen validation thresholds and an inspectable synthetic decision trace.

Run only after the complete study:
    python3 scripts/run_gpu.py -- scripts/explain_example.py

Truth is used to choose an illustrative median-regret example and in a separate
evaluator-only section, never to alter model predictions or choose thresholds.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from run_study import ROOT, Model, expected_regret, generate, predict, subset, utilities
from regen import topk_gate, serialize_reports

SEEDS = [11, 23, 37, 51, 71]
KEYS = ["B2", "P", "no_provenance", "no_semantic", "dense_gate"]
GRID = [0.0, 0.1, 0.3, 0.5]
BATCH = 256


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_model(seed, key):
    checkpoint = ROOT / "results/checkpoints" / f"{seed}_{key}.pt"
    model = Model(kind="dense" if key in ("B2", "dense_gate") else "hdc").cuda()
    model.load_state_dict(torch.load(checkpoint, map_location="cuda", weights_only=True))
    model.eval()
    return model, checkpoint


@torch.inference_mode()
def recover_thresholds(validation):
    output = {
        "procedure": "Reconstructed after the run from frozen checkpoints using validation only.",
        "validation_seed": 22027,
        "validation_snapshots": 2000,
        "batch_size": BATCH,
        "temperature": 0.1,
        "threshold_grid": GRID,
        "aggregation": "Mean of batch-mean expected regrets, matching the original runner exactly.",
        "tie_breaking": "First minimum in ascending threshold grid, matching numpy.argmin.",
        "source_manifest_sha256": sha256(ROOT / "results/study_manifest.json"),
        "rows": [],
    }
    ids = torch.arange(2000, device="cuda").split(BATCH)
    for seed in SEEDS:
        for key in KEYS:
            model, checkpoint = load_model(seed, key)
            scores = []
            for threshold in GRID:
                losses = []
                for part in ids:
                    batch = subset(validation, part)
                    beliefs = predict(key, {key: model}, batch, threshold=threshold)
                    losses.append(expected_regret(beliefs, batch))
                scores.append(torch.stack(losses).mean().item())
            best = int(np.argmin(scores))
            output["rows"].append({
                "seed": seed, "method": key,
                "threshold": GRID[best],
                "validation_scores": scores,
                "checkpoint_sha256": sha256(checkpoint),
            })
            del model
    return output


@torch.inference_mode()
def example_trace(thresholds):
    seed, method = 37, "P"
    selected = next(row for row in thresholds["rows"]
                    if row["seed"] == seed and row["method"] == method)
    threshold = selected["threshold"]
    model, checkpoint = load_model(seed, method)
    data = generate(4000, 32027, "cuda", facts=64, family="test",
                    corruption=0.2, attack="targeted", budget=2048)
    regrets = []
    for ids in torch.arange(4000, device="cuda").split(BATCH):
        batch = subset(data, ids)
        p = predict(method, {method: model}, batch, threshold=threshold)
        choice = utilities(p, batch).topk(3, dim=-1).indices[:, :1]
        actual = utilities(batch["y"], batch)
        regrets.append(actual.max(-1).values - actual.gather(1, choice).squeeze(-1))
    regret = torch.cat(regrets)
    raw_path = ROOT / "results/runs/37/P_severity_0.2_targeted_64_2048_0.1.npz"
    with np.load(raw_path) as saved:
        expected = saved["regret"].copy()
    maximum_difference = float(np.max(np.abs(regret.cpu().numpy() - expected)))
    if maximum_difference > 1e-5:
        raise RuntimeError(f"Recovered decisions differ from original run: {maximum_difference}")
    # Exact upper order-statistic median; tied regrets retain ascending row order.
    index = int(regret.argsort(dim=0, stable=True)[len(regret) // 2].item())
    batch = subset(data, slice(index, index + 1))
    p = predict(method, {method: model}, batch, threshold=threshold)
    rec_logits, gate_logits = model(batch["x"], batch["adj"])
    q = gate_logits.sigmoid()
    gate = topk_gate(q, 6) * (q > threshold)
    changed = ((p[0] - batch["prior"][0]).abs() > 1e-4).nonzero().flatten().tolist()
    selected_facts = gate[0].bool().nonzero().flatten().tolist()
    predicted = utilities(p, batch)[0]
    actual = utilities(batch["y"], batch)[0]
    top = predicted.topk(3).indices.tolist()
    support = set(selected_facts)
    candidates = []
    for rank, candidate in enumerate(top, 1):
        required = batch["req"][0, candidate].bool().nonzero().flatten().tolist()
        support.update(required)
        candidates.append({
            "rank": rank, "candidate_id": candidate, "required_fact_ids": required,
            "known_value": float(batch["values"][0, candidate]),
            "known_cost": float(batch["costs"][0, candidate]),
            "predicted_utility": float(predicted[candidate]),
            "relaxed_prerequisite_probability": float(p[0, required].prod()),
        })
    observations = []
    for fact in sorted(support):
        sources = []
        for source in range(3):
            available = bool(batch["observed"][0, fact, source])
            sources.append({
                "source_id": source, "local": source == 0, "available": available,
                "report": float(batch["reports"][0, fact, source]) if available else None,
                "age": float(batch["x"][0, fact, 6 + source]) if available else None,
            })
        observations.append({
            "fact_id": fact, "parent_fact_id": int(batch["parent"][0, fact]),
            "sources": sources, "noisy_history": float(batch["x"][0, fact, 9]),
            "provisional_belief": float(batch["prior"][0, fact]),
            "reconstruction": float(rec_logits.sigmoid()[0, fact]),
            "final_belief": float(p[0, fact]), "gate_probability": float(q[0, fact]),
            "gate_selected": fact in selected_facts, "belief_changed": fact in changed,
        })
    actual_candidates = [{
        "candidate_id": candidate, "true_utility": float(actual[candidate]),
        "true_feasible": bool(batch["y"][0, batch["req"][0, candidate].bool()].bool().all()),
    } for candidate in top]
    payloads = serialize_reports(batch)
    return {
        "purpose": "Input-evidence and decision inspection; not a causal attribution or a full planner explanation.",
        "selection": {
            "training_seed": seed, "method": method, "test_seed": 32027,
            "condition": "severity", "corruption": 0.2, "attack": "targeted",
            "facts": 64, "budget_bytes_per_remote_source": 2048, "repair_fraction": 0.1,
            "snapshot_index": index, "topology_group": int(batch["group"][0]),
            "rule": "Stable ascending regret order, row at index N//2 (upper sample median).",
            "selection_uses_evaluation_truth": True,
            "selection_note": "Chosen only for illustration after aggregate metrics were fixed.",
            "threshold": threshold, "threshold_selection_split": "validation",
            "checkpoint_sha256": sha256(checkpoint),
            "original_regret_reproduction_max_absolute_error": maximum_difference,
        },
        "observable_inputs_and_model_decision": {
            "delivered_remote_bytes": {str(k): len(v) for k, v in payloads.items()},
            "selected_fact_ids": selected_facts, "changed_fact_ids": changed,
            "supporting_observations": observations, "top_three_candidates": candidates,
            "logical_rule": "Each non-root fact implies its listed parent.",
        },
        "evaluator_only_hidden_truth": {
            "not_available_to_model": True,
            "selected_candidate_regret": float(regret[index]),
            "oracle_candidate_id": int(actual.argmax()),
            "oracle_utility": float(actual.max()),
            "top_three_actual_outcomes": actual_candidates,
            "supporting_true_facts": {str(fact): int(batch["y"][0, fact]) for fact in sorted(support)},
            "genuine_changed_fact_ids": batch["transitions"][0].nonzero().flatten().tolist(),
            "delivered_changed_report_fraction": float(batch["changed_fraction"][0]),
            "delivered_intervention_fraction": float(batch["intervention_fraction"][0]),
        },
    }


def main():
    manifest = ROOT / "results/study_manifest.json"
    if not manifest.is_file():
        raise RuntimeError("Complete the main study before reconstructing metadata.")
    settings = json.loads(manifest.read_text())["settings"]
    expected = {"val": 2000, "test": 4000, "batch": BATCH, "seeds": SEEDS, "pilot": False}
    mismatches = {key: settings.get(key) for key, value in expected.items()
                  if settings.get(key) != value}
    if mismatches:
        raise RuntimeError(f"Frozen replay settings do not match this study: {mismatches}")
    if not torch.cuda.is_available() or "5000 Ada" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Use scripts/run_gpu.py to select the attached Ada GPU.")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(2027)
    validation = generate(2000, 22027, "cuda", family="validation")
    thresholds = recover_thresholds(validation)
    trace = example_trace(thresholds)
    for name, obj in (("selected_thresholds", thresholds), ("example_trace", trace)):
        path = ROOT / "results" / f"{name}.json"
        path.write_text(json.dumps(obj, indent=2) + "\n")
        print(f"Wrote {path.relative_to(ROOT)} ({path.stat().st_size} bytes)")
    print(json.dumps({"threshold": trace["selection"]["threshold"],
                      "snapshot_index": trace["selection"]["snapshot_index"],
                      "reproduction_max_error": trace["selection"]["original_regret_reproduction_max_absolute_error"]}))


if __name__ == "__main__":
    main()
