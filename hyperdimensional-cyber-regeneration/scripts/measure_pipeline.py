#!/usr/bin/env python3
"""Bounded host-wall timing from prepared evidence to ranked candidate IDs.

This is a post-training timing audit, not a second evaluation or selection run.
It includes real codec packing/unpacking, transfers, CUDA feature reconstruction,
model inference and candidate ranking. Sensing, bandwidth admission, graph/action
catalogue construction, transport delay and cold model loading are excluded.
"""
import hashlib
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from regen import Model, generate, serialize_reports, utilities
from run_study import predict, subset

METHODS = ["B0", "B1", "B2", "B3", "B4", "B5", "P", "random_gate",
           "no_provenance", "no_semantic", "dense_gate"]
KEYS = ["B1", "B2", "P", "no_provenance", "no_semantic", "dense_gate"]


def prepare(host):
    # Serialize actual remote payloads. Local observer stays on the local node.
    payloads = serialize_reports(host)
    facts = host["x"].shape[1]
    values = [[.5] * 3 for _ in range(facts)]
    seen = [[0.] * 3 for _ in range(facts)]
    ages = [[1.] * 3 for _ in range(facts)]
    for fact in range(facts):
        values[fact][0] = float(host["reports"][0, fact, 0])
        seen[fact][0] = float(host["observed"][0, fact, 0])
        ages[fact][0] = float(host["x"][0, fact, 6])
    for source, payload in payloads.items():
        if not payload:
            continue
        magic, sid, count, reserved = struct.unpack_from("<4sIII", payload)
        if magic != b"HCR1" or sid != source or len(payload) != 16 + count * 12:
            raise ValueError("Invalid codec payload")
        for fact, sid, value, age in struct.iter_unpack("<HHff", payload[16:]):
            values[fact][sid] = value
            seen[fact][sid] = 1.
            ages[fact][sid] = age
    reports = torch.tensor([values], device="cuda")
    observed = torch.tensor([seen], device="cuda")
    age = torch.tensor([ages], device="cuda")
    # Metadata and previous observation are prepared externally; no test truth
    # or intervention labels are transferred or used in the measured path.
    data = {k: host[k].to("cuda") for k in
            ["adj", "req", "values", "costs", "parent"]}
    history = host["x"][:, :, 9].to("cuda")
    degree = host["x"][:, :, 11].to("cuda")
    criticality = host["x"][:, :, 12].to("cuda")
    weight = observed * (1.5 - .5 * age)
    prior = ((weight * reports).sum(-1) + .45 * history) / (weight.sum(-1) + .45)
    prior = prior.clamp(.02, .98)
    data["prior"] = prior
    data["x"] = torch.cat([reports, observed, age, history.unsqueeze(-1),
                           prior.unsqueeze(-1), degree.unsqueeze(-1),
                           criticality.unsqueeze(-1), (prior-history).abs().unsqueeze(-1)], -1)
    data["reports"], data["observed"] = reports, observed
    return data, sum(map(len, payloads.values()))


def main():
    if not torch.cuda.is_available() or "5000 Ada" not in torch.cuda.get_device_name():
        raise RuntimeError("Attached RTX 5000 Ada required")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.use_deterministic_algorithms(True)
    threshold_path = ROOT / "results/selected_thresholds.json"
    thresholds_data = json.loads(threshold_path.read_text())
    # Explainability script publishes a flat list of seed/method/threshold rows.
    entries = thresholds_data["rows"]
    thresholds = {r["method"]: r["threshold"] for r in entries if int(r["seed"]) == 37}
    models = {}
    for key in KEYS:
        model = Model("dense" if key in ["B1", "B2", "dense_gate"] else "hdc").cuda()
        model.load_state_dict(torch.load(ROOT / "results/checkpoints" / f"37_{key}.pt",
                                        map_location="cuda", weights_only=True))
        models[key] = model.eval()
    reference = subset(generate(4000, 32027, "cuda", family="test"), slice(0, 1))
    # Prepared input is resident on host at the timing start; truth omitted.
    host = {k: v.cpu() for k, v in reference.items() if k in
            ["x", "reports", "observed", "adj", "req", "values", "costs", "parent"]}
    results = []
    with torch.inference_mode():
        restored, payload_bytes = prepare(host)
        if not torch.allclose(restored["x"], reference["x"], atol=1e-6, rtol=0):
            raise AssertionError("Codec round-trip changed model features")
        for method in METHODS:
            key = {"B4": "P", "B5": "P", "random_gate": "P"}.get(method, method)
            threshold = thresholds.get(key, 0.)
            actual = predict(method, models, restored, threshold=threshold)
            expected = predict(method, models, reference, threshold=threshold)
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=0)
            times = []
            for repeat in range(55):
                torch.cuda.synchronize()
                start = time.perf_counter()
                data, _ = prepare(host)
                p = predict(method, models, data, threshold=threshold)
                top = utilities(p, data).topk(3, -1).indices.cpu().tolist()
                torch.cuda.synchronize()
                elapsed = (time.perf_counter() - start) * 1000
                if repeat >= 5:
                    times.append(elapsed)
            results.append(dict(method=method, median_ms=float(np.median(times)),
                                p10_ms=float(np.quantile(times, .1)),
                                p95_ms=float(np.quantile(times, .95)),
                                repeats=len(times), top3=top[0]))
    output = dict(hardware=torch.cuda.get_device_name(), seed=37, facts=64,
                  scenario="first fixed reference test snapshot; timing audit only",
                  payload_bytes=payload_bytes, scope=__doc__.strip(),
                  precision="FP32 inference; TF32 allowed", warmups=5,
                  round_trip_predictions="all 11 methods matched original CUDA features/predictions at atol 1e-6",
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  threshold_sha256=hashlib.sha256(threshold_path.read_bytes()).hexdigest(),
                  rows=results)
    path = ROOT / "results/pipeline_timing.json"
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"path": str(path), "methods": len(results), "payload_bytes": payload_bytes}))


if __name__ == "__main__":
    main()
