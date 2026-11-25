#!/usr/bin/env python3
"""Bounded wall timing from decoded flow measurements to attack probabilities.

Includes CPU orchestration, H2D transfer of numeric measurements/observation
masks/hashed endpoint metadata/valid flags, CUDA training-fitted normalization,
CUDA endpoint-sharing graph construction, model inference, and D2H float32
probabilities. Excludes CSV scanning/decoding, raw packet/NetFlow construction,
network transport, waiting for flow or window completion, and cold model/key
loading. This fixed-window audit is not operational end-to-end detection time.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from real_data import (CONDITIONS, TEST, WINDOW, input_hash, load_capture,
                       require_cuda, sha, subset, windows)
from real_models import KEYS, infer, make_model
from real_metrics import GaussianNB

SEEDS = [11, 23, 37, 51, 71]
SEED = 37
SCENARIO = 10
METHODS = list(KEYS) + ["NB"]
PROBABILITY_TOLERANCE = 1e-5


def completed_manifest():
    """Check a completed frozen study before any audit model inference."""
    path = ROOT / "results/real/manifest.json"
    if not path.is_file():
        raise RuntimeError("Wait for the complete real-data GPU study before this audit")
    manifest = json.loads(path.read_text())
    for key, expected in {"pilot": False, "seeds": SEEDS, "batch": 256}.items():
        if manifest["settings"].get(key) != expected:
            raise AssertionError(f"Unexpected completed-study setting {key}")
    if manifest.get("methods") != METHODS or manifest.get("wall_seconds", 0) <= 0:
        raise AssertionError("Expected all nine methods and a completed wall-time record")
    expected_checkpoints = {(seed, method) for seed in SEEDS for method in METHODS}
    actual = {(int(record["seed"]), record["method"]) for record in manifest.get("checkpoints", [])}
    if actual != expected_checkpoints or len(manifest["checkpoints"]) != 45:
        raise AssertionError("Expected all 45 seed/method checkpoint records")
    if len(manifest.get("thresholds", [])) != 45 or len(manifest.get("raw_hashes", {})) != 675:
        raise AssertionError("Expected all 45 thresholds and 675 raw case hashes")
    expected_cases = {f"{condition}_{scenario}" for condition in CONDITIONS for scenario in TEST}
    if set(manifest.get("case_hashes", {})) != expected_cases:
        raise AssertionError("Expected all fifteen held-out capture/condition cases")
    for relative, digest in manifest["source_hashes"].items():
        if sha(ROOT / relative) != digest:
            raise AssertionError(f"Frozen scientific source changed: {relative}")
    data_path = ROOT / "results/real/ctu13_data_manifest.json"
    if sha(data_path) != manifest["data_manifest_sha256"]:
        raise AssertionError("Prepared-data manifest changed")
    data_manifest = json.loads(data_path.read_text())
    if not data_manifest.get("complete") or not data_manifest["temporal_split_check"]["passed"]:
        raise AssertionError("Complete, chronologically separated captured data required")
    for record in data_manifest["files"]:
        if record["scenario"] in TEST and sha(ROOT / record["prepared_path"]) != record["prepared_sha256"]:
            raise AssertionError(f"Prepared test capture changed: {record['scenario']}")
    return path, manifest, data_manifest


def configure_cuda():
    require_cuda()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(2027)


def frozen_preprocessing(manifest):
    """Restore recorded FP32 statistics; never fit on validation or test data."""
    return {key: torch.tensor(values, device="cuda", dtype=torch.float32)
            for key, values in manifest["preprocessing"].items()}


def load_model(method, manifest, seed=SEED):
    path = ROOT / "results/checkpoints/real" / f"{seed}_{method}.pt"
    record = next(record for record in manifest["checkpoints"]
                  if int(record["seed"]) == seed and record["method"] == method)
    digest = sha(path)
    if digest != record["checkpoint_sha256"]:
        raise AssertionError(f"Checkpoint changed: {path}")
    model = (GaussianNB() if method == "NB" else make_model(method)).cuda()
    model.load_state_dict(torch.load(path, map_location="cuda", weights_only=True))
    return model.eval(), path, digest


def predictions(model, batch):
    return model.probabilities(batch) if model.key == "NB" else infer(model, batch)


def prepare(host, preprocessing):
    """Measured label-free path from a decoded CPU window to CUDA inputs.

    Host inputs contain only features[1,32,14], observed[1,32,14], hashed src/dst
    and valid[1,32]. Normalization statistics and model weights are resident on
    CUDA before the timing starts. IP hashes affect edges, never numeric inputs.
    """
    features = host["features"].to("cuda")
    observed = host["observed"].to("cuda")
    src, dst = host["src"].to("cuda"), host["dst"].to("cuda")
    valid = host["valid"].to("cuda")
    observed = observed & valid[..., None]
    scaled = ((features - preprocessing["mean"]) / preprocessing["std"]).clamp(-5, 5)
    values = torch.where(observed, scaled, torch.zeros_like(scaled))
    adj = (((src[:, :, None] == src[:, None, :]) & (src[:, :, None] != 0)) |
           ((dst[:, :, None] == dst[:, None, :]) & (dst[:, :, None] != 0)) |
           ((src[:, :, None] == dst[:, None, :]) & (src[:, :, None] != 0)) |
           ((dst[:, :, None] == src[:, None, :]) & (dst[:, :, None] != 0)))
    adj = adj & valid[:, :, None] & valid[:, None, :]
    return dict(x=torch.cat((values, observed.float()), -1), adj=adj.float(), valid=valid)


def cpu_first_window(scenario):
    """Read decoded CPU arrays outside timing; no labels are placed in host."""
    path = ROOT / "data/external/ctu13/parsed" / f"scenario_{scenario:02d}.npz"
    host = {}
    with np.load(path, allow_pickle=False) as archive:
        for key in ["features", "observed", "src", "dst"]:
            array = archive[key][:WINDOW].copy()
            if array.dtype == np.uint64:
                array = array.view(np.int64)
            host[key] = torch.from_numpy(array)[None].contiguous()
    if host["features"].shape != (1, WINDOW, 14):
        raise AssertionError("Timing expects a complete first 32-flow window")
    host["features"] = host["features"].float()
    host["observed"] = host["observed"].bool()
    host["valid"] = torch.ones((1, WINDOW), dtype=torch.bool)
    return host


@torch.inference_mode()
def main():
    manifest_path, manifest, data_manifest = completed_manifest()
    configure_cuda()
    preprocessing = frozen_preprocessing(manifest)
    reference = windows(load_capture(SCENARIO), SCENARIO, preprocessing)
    if input_hash(reference) != manifest["case_hashes"][f"clean_{SCENARIO}"]:
        raise AssertionError("Reconstructed clean reference capture changed")
    host = cpu_first_window(SCENARIO)
    restored = prepare(host, preprocessing)
    direct_window = subset(reference, slice(0, 1))
    for key in ["x", "adj", "valid"]:
        torch.testing.assert_close(restored[key], direct_window[key], atol=0., rtol=0.)
    component_bytes = {key: value.numel() * value.element_size() for key, value in host.items()}
    rows, checkpoints, stored_errors = [], {}, {}
    for method in METHODS:
        model, checkpoint_path, checkpoint_digest = load_model(method, manifest)
        checkpoints[method] = {"path": str(checkpoint_path.relative_to(ROOT)), "sha256": checkpoint_digest}
        # Test prepared-input equivalence at the same batch-one inference shape.
        prepared_probability = predictions(model, restored)
        expected_probability = predictions(model, direct_window)
        torch.testing.assert_close(prepared_probability, expected_probability, atol=1e-6, rtol=0.)
        # Separately match the evaluator's original batch shape to stored outputs.
        original_batch = subset(reference, slice(0, manifest["settings"]["batch"]))
        original_probability = predictions(model, original_batch)[0].cpu().numpy()
        raw_path = ROOT / "results/runs/real" / str(SEED) / f"{method}_clean_{SCENARIO}.npz"
        if sha(raw_path) != manifest["raw_hashes"][str(raw_path.relative_to(ROOT))]:
            raise AssertionError(f"Saved raw probabilities changed: {raw_path}")
        with np.load(raw_path, allow_pickle=False) as raw:
            error = float(np.max(np.abs(original_probability - raw["prob"][:WINDOW])))
        if not np.isfinite(error) or error > PROBABILITY_TOLERANCE:
            raise AssertionError(f"Stored first-window replay differs for {method}: {error}")
        stored_errors[method] = error
        durations = []
        for repetition in range(55):
            torch.cuda.synchronize()
            start = time.perf_counter()
            b = prepare(host, preprocessing)
            returned_probability = predictions(model, b).cpu().numpy()
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000
            if repetition >= 5:
                durations.append(elapsed)
        rows.append(dict(method=method, median_ms=float(np.median(durations)),
                         p10_ms=float(np.quantile(durations, .1)), p95_ms=float(np.quantile(durations, .95)),
                         repeats=50, returned_probabilities=int(returned_probability.size),
                         output_bytes=int(returned_probability.nbytes)))
        del model
    result = dict(seed=SEED, scenario=SCENARIO, condition="clean", window=0, flows=WINDOW,
        rows=rows, hardware=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda,
        warmups=5, scope=__doc__.strip(), input_component_bytes=component_bytes,
        input_bytes=sum(component_bytes.values()), output_bytes=WINDOW*4,
        input_byte_scope="Actual decoded host tensors transferred per window; not a wire codec or compression measurement.",
        resident_normalization_bytes=sum(preprocessing[key].numel()*preprocessing[key].element_size() for key in ["mean", "std"]),
        model_and_statistics_resident=True, labels_in_measured_inputs=False,
        input_roundtrip="Prepared features, adjacency and validity exactly match the stored-case reconstruction.",
        prediction_roundtrip="All nine batch-one prediction paths match direct prepared-reference inputs at absolute tolerance 1e-6.",
        original_batch_first_window_max_absolute_errors=stored_errors,
        original_batch_probability_tolerance=PROBABILITY_TOLERANCE,
        checkpoint_hashes=checkpoints, manifest_sha256=sha(manifest_path),
        data_manifest_sha256=manifest["data_manifest_sha256"], case_input_sha256=input_hash(reference),
        source_hashes={str(path.relative_to(ROOT)): sha(path) for path in
            [Path(__file__), ROOT/"src/real_data.py", ROOT/"src/real_models.py", ROOT/"src/real_metrics.py"]})
    output = ROOT / "results/real/pipeline_timing.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(output=str(output.relative_to(ROOT)), methods=len(rows),
                          input_bytes=result["input_bytes"], output_bytes=result["output_bytes"],
                          maximum_stored_probability_error=max(stored_errors.values()))))


if __name__ == "__main__":
    main()
