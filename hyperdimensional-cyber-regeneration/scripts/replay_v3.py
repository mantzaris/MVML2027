#!/usr/bin/env python3
"""Replay frozen V3 checkpoints after the complete fresh-test study.

Recomputes seed-37 SCR, DAE and LOGIT on the four prespecified core conditions. It
compares stored candidate probabilities, regret, and plan Brier per snapshot,
then writes a compact replay audit.
Run only after results/v3/manifest.json exists; no training or tuning occurs.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from run_v3 import ROOT, make_model, infer, generate, subset, utilities, candidate_utilities
from regen_v2 import independent_probability
from regen import serialize_reports

SEED = 37
TEST_SEED = 62027
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
    path = ROOT / "results/checkpoints/v3" / f"{SEED}_{method}.pt"
    record = next(r for r in manifest["checkpoints"]
                  if int(r["seed"]) == SEED and r["method"] == method)
    digest = sha(path)
    if digest != record["checkpoint_sha256"]:
        raise AssertionError(f"Checkpoint changed: {path}")
    model = RegeneratorV3(method).cuda()
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



def main():
    path = ROOT / "results/v3/manifest.json"
    if not path.is_file():
        raise RuntimeError("Wait for the complete V3 study before replaying checkpoints.")
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
    models = {method: load_model(method, manifest) for method in ["SCR", "DAE", "LOGIT"]}
    rows = []
    for condition in CONDITIONS:
        name, severity, attack = condition
        data = generate(COUNT, TEST_SEED, "cuda", family="test", facts=64,
                        corruption=severity, attack=attack, budget=2048, missing=.2)
        data_sha = fingerprint(data)
        if data_sha != manifest["test_hashes"][tag(condition)]:
            raise AssertionError(f"Regenerated inputs changed: {tag(condition)}")
        for method, (model, checkpoint_path, checkpoint_sha) in models.items():
            raw = ROOT / "results/runs/v3" / str(SEED) / f"{method}_{tag(condition)}.npz"
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
    (ROOT/"results/v3/replay.json").write_text(json.dumps(output, indent=2)+"\n")
    print(json.dumps({"replays": len(rows), "maximum_absolute_errors": output["maximum_absolute_errors"]}))


if __name__ == "__main__":
    main()
