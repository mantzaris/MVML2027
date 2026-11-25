#!/usr/bin/env python3
"""Replay frozen V2 checkpoints after the complete fresh-test study.

Recomputes seed-37 JH and ID on the four prespecified core conditions. It
compares stored candidate probabilities, regret, and plan Brier per snapshot,
then writes a compact audit and an illustrative, clearly separated trace.
Run only after results/v2/manifest.json exists; no training or tuning occurs.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from run_v2 import ROOT, RegeneratorV2, infer, generate, subset, utilities, candidate_utilities
from regen_v2 import independent_probability
from regen import serialize_reports

SEED = 37
TEST_SEED = 42027
COUNT = 4000
BATCH = 256
TOLERANCE = 1e-5
CONDITIONS = [
    ("severity", 0.0, "targeted"),
    ("severity", 0.2, "targeted"),
    ("attack", 0.2, "random"),
    ("attack", 0.2, "coherent"),
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tag(condition):
    name, severity, attack = condition
    return f"{name}_{severity}_{attack}_64_2048_0.1"


def fingerprint(batch):
    return hashlib.sha256(batch["x"].cpu().numpy().tobytes()
                          + batch["y"].cpu().numpy().tobytes()
                          + batch["req"].cpu().numpy().tobytes()).hexdigest()


def load_model(method, manifest):
    path = ROOT / "results/checkpoints/v2" / f"{SEED}_{method}.pt"
    record = next(r for r in manifest["checkpoints"]
                  if int(r["seed"]) == SEED and r["method"] == method)
    digest = sha(path)
    if digest != record["checkpoint_sha256"]:
        raise AssertionError(f"Checkpoint changed: {path}")
    model = RegeneratorV2(method).cuda()
    model.load_state_dict(torch.load(path, map_location="cuda", weights_only=True))
    return model.eval(), path, digest


@torch.inference_mode()
def recompute(model, data):
    chunks = []
    for ids in torch.arange(COUNT, device="cuda").split(BATCH):
        b = subset(data, ids)
        _, prob = infer(model, b)
        actual = utilities(b["y"], b)
        top = candidate_utilities(prob, b).topk(3, -1).indices
        truth = (torch.bmm(b["req"], (1-b["y"]).unsqueeze(-1)).squeeze(-1) < .5)
        regret = actual.max(-1).values - actual.gather(1, top[:, :1]).squeeze(-1)
        brier = (prob-truth.float()).square().mean(-1)
        chunks.append({"planprob": prob.cpu().numpy(), "regret": regret.cpu().numpy(),
                       "plan_brier": brier.cpu().numpy()})
    return {key: np.concatenate([part[key] for part in chunks]) for key in chunks[0]}


@torch.inference_mode()
def example_trace(model, data, replayed, checkpoint_sha):
    # A deterministic illustration chosen only after all efficacy metrics exist.
    index = int(np.argsort(replayed["regret"], kind="stable")[COUNT//2])
    b = subset(data, slice(index, index+1))
    # Preserve the original batch shape to avoid a rare batch-one kernel/tie
    # difference between the displayed choice and the stored evaluator result.
    start = (index//BATCH)*BATCH
    block = subset(data, slice(start, min(start+BATCH, COUNT)))
    local = index-start
    block_p, block_joint = infer(model, block)
    p, joint = block_p[local:local+1], block_joint[local:local+1]
    factors = model(block["x"], block["adj"], block["parent"]).float().sigmoid()[local:local+1]
    product = independent_probability(p, b["req"])
    ranked = candidate_utilities(joint, b).topk(3, -1).indices[0].tolist()
    actual = utilities(b["y"], b)[0]
    displayed_regret = float(actual.max()-actual[ranked[0]])
    if abs(displayed_regret-float(replayed["regret"][index])) > TOLERANCE:
        raise AssertionError("Displayed trace decision disagrees with replayed regret")
    roots = b["parent"][0] == torch.arange(64, device="cuda")
    candidates = []
    support = set()
    for candidate in ranked:
        required = b["req"][0, candidate].bool().nonzero().flatten().tolist()
        required_roots = sorted({int(b["parent"][0, f]) for f in required})
        required_children = [f for f in required if not bool(roots[f])]
        support.update(required)
        support.update(required_roots)
        candidates.append({
            "candidate_id": candidate, "required_fact_ids": required,
            "distinct_required_root_ids": required_roots,
            "root_factors": {str(r): float(factors[0, r]) for r in required_roots},
            "conditional_child_factors": {str(f): float(factors[0, f]) for f in required_children},
            "joint_success_probability": float(joint[0, candidate]),
            "product_of_marginals": float(product[0, candidate]),
            "predicted_utility": float(candidate_utilities(joint, b)[0, candidate]),
            "known_value": float(b["values"][0, candidate]),
            "known_cost": float(b["costs"][0, candidate]),
        })
    observations = []
    for fact in sorted(support):
        reports = []
        for source in range(3):
            if b["observed"][0, fact, source] > .5:
                reports.append({"source": source,
                                "reported_value": float(b["reports"][0, fact, source]),
                                "synthetic_age": float(b["x"][0, fact, 6+source])})
        observations.append({"fact_id": fact, "parent_id": int(b["parent"][0, fact]),
                             "reports": reports, "prior_belief": float(b["prior"][0, fact]),
                             "regenerated_marginal": float(p[0, fact]),
                             "conditional_or_root_factor": float(factors[0, fact])})
    payloads = serialize_reports(b)
    return {
        "purpose": "Inspect the dependency-scoring mechanism, not a causal explanation or a complete operational plan.",
        "selection": {
            "training_seed": SEED, "method": "JH", "test_seed": TEST_SEED,
            "condition": "severity", "corruption": .2, "attack": "targeted",
            "facts": 64, "budget_per_remote_observer": 2048,
            "snapshot_index": index, "topology_group": int(b["group"][0]),
            "rule": "Stable ascending regret order, element N//2 (upper sample median).",
            "selection_uses_evaluation_truth": True,
            "selection_note": "Selection is post hoc for illustration only; no model choice or tuning.",
            "checkpoint_sha256": checkpoint_sha,
            "inference_batch_start": start, "inference_batch_size": len(block["y"]),
            "displayed_choice_regret_checked": True,
        },
        "observable_inputs_and_model_decision": {
            "top_three_candidates": candidates,
            "supporting_observations": observations,
            "remote_payload_bytes": {str(source): len(payload) for source, payload in payloads.items()},
            "scoring_semantics": "Each distinct required root contributes once, followed by each required child's conditional factor.",
            "assumption": "Joint score is exact under the conditional star-forest posterior, not a claim of exact true posterior inference.",
        },
        "evaluator_only_hidden_truth": {
            "not_available_to_model": True,
            "selected_candidate_regret": float(replayed["regret"][index]),
            "oracle_candidate_id": int(actual.argmax()), "oracle_utility": float(actual.max()),
            "top_three_actual_outcomes": [
                {"candidate_id": j, "true_utility": float(actual[j]),
                 "truly_feasible": bool(b["y"][0, b["req"][0, j].bool()].bool().all())}
                for j in ranked],
            "supporting_true_facts": {str(f): int(b["y"][0, f]) for f in sorted(support)},
            "genuine_changed_fact_ids": b["transitions"][0].nonzero().flatten().tolist(),
        },
    }


def main():
    path = ROOT / "results/v2/manifest.json"
    if not path.is_file():
        raise RuntimeError("Wait for the complete V2 study before replaying checkpoints.")
    manifest = json.loads(path.read_text())
    expected = {"test": COUNT, "test_seed": TEST_SEED, "batch": BATCH,
                "seeds": [11, 23, 37, 51, 71], "pilot": False}
    for key, value in expected.items():
        if manifest["settings"].get(key) != value:
            raise AssertionError(f"Unexpected frozen setting {key}: {manifest['settings'].get(key)}")
    if len(manifest["checkpoints"]) != 25:
        raise AssertionError("Completed manifest must record all 25 new trained checkpoints.")
    for relative, digest in manifest["source_hashes"].items():
        if sha(ROOT/relative) != digest:
            raise AssertionError(f"Frozen scientific source changed: {relative}")
    if not torch.cuda.is_available() or "5000 Ada" not in torch.cuda.get_device_name():
        raise RuntimeError("Use the attached Ada GPU through scripts/run_gpu.py.")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(2027)
    models = {method: load_model(method, manifest) for method in ["JH", "ID"]}
    rows = []
    trace = None
    for condition in CONDITIONS:
        name, severity, attack = condition
        data = generate(COUNT, TEST_SEED, "cuda", family="test", facts=64,
                        corruption=severity, attack=attack, budget=2048, missing=.2)
        data_sha = fingerprint(data)
        if data_sha != manifest["test_hashes"][tag(condition)]:
            raise AssertionError(f"Regenerated inputs changed: {tag(condition)}")
        for method, (model, checkpoint_path, checkpoint_sha) in models.items():
            raw = ROOT / "results/runs/v2" / str(SEED) / f"{method}_{tag(condition)}.npz"
            replayed = recompute(model, data)
            errors = {}
            with np.load(raw, allow_pickle=False) as saved:
                for key, current in replayed.items():
                    if current.shape != saved[key].shape:
                        raise AssertionError(f"Shape mismatch: {raw.name}/{key}")
                    errors[key] = float(np.max(np.abs(current-saved[key])))
            if not all(np.isfinite(error) and error <= TOLERANCE for error in errors.values()):
                raise AssertionError(f"Replay mismatch: {method}/{tag(condition)} {errors}")
            rows.append({"method": method, "condition": name, "corruption": severity,
                         "attack": attack, "snapshots": COUNT, "max_absolute_errors": errors,
                         "checkpoint_path": str(checkpoint_path.relative_to(ROOT)),
                         "checkpoint_sha256": checkpoint_sha, "raw_path": str(raw.relative_to(ROOT)),
                         "raw_sha256": sha(raw), "input_data_sha256": data_sha})
            if method == "JH" and condition == ("severity", .2, "targeted"):
                trace = example_trace(model, data, replayed, checkpoint_sha)
        del data
    output = {
        "passed": True, "training_seed": SEED, "fresh_test_seed": TEST_SEED,
        "hardware": torch.cuda.get_device_name(), "torch": torch.__version__,
        "cuda": torch.version.cuda, "precision": "FP32 inference; TF32 allowed; deterministic algorithms",
        "absolute_tolerance": TOLERANCE, "model_condition_replays": len(rows),
        "model_snapshot_evaluations": len(rows)*COUNT,
        "scope": "Replay verification, not a new efficacy estimate or model selection exercise.",
        "source_manifest_sha256": sha(path), "replay_script_sha256": sha(__file__),
        "maximum_absolute_errors": {key: max(row["max_absolute_errors"][key] for row in rows)
                                    for key in rows[0]["max_absolute_errors"]},
        "rows": rows,
    }
    (ROOT/"results/v2/replay.json").write_text(json.dumps(output, indent=2)+"\n")
    (ROOT/"results/v2/example_trace.json").write_text(json.dumps(trace, indent=2)+"\n")
    print(json.dumps({"replays": len(rows), "maximum_absolute_errors": output["maximum_absolute_errors"],
                      "example_snapshot": trace["selection"]["snapshot_index"]}))


if __name__ == "__main__":
    main()
