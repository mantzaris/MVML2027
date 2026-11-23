#!/usr/bin/env python3
"""Numerical validity and bounded GPU timing pilot; not research performance.

Run only through scripts/run_on_ada.py. The optional classical spline-EMD
reference uses a 4096-sample CPU fixture as allowed by the research protocol.
All research feature/simulation workloads are CUDA. Reports are atomically
written inside this project, preventing partial artifacts after power loss.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
import torch
from uac.dsp import extract_features, emd_linear, hht_features, morlet_power
from uac.synthetic import generate_batch


def atomic_json(path, content):
    path = path.resolve()
    if PROJECT not in path.parents:
        raise ValueError("Validation artifacts must remain inside this submission directory")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as handle:
        json.dump(content, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def classic_first_imf_reference(x):
    """Small independent CubicSpline reference fixture, not preprocessing."""
    import numpy as np
    from scipy.interpolate import CubicSpline
    from scipy.signal import find_peaks
    h = x.copy()
    indices = np.arange(len(x))
    for _ in range(10):
        maxima = find_peaks(h)[0]
        minima = find_peaks(-h)[0]
        if len(maxima) < 2 or len(minima) < 2:
            break
        upper_indices = np.concatenate(([0], maxima, [len(x) - 1]))
        lower_indices = np.concatenate(([0], minima, [len(x) - 1]))
        upper = CubicSpline(upper_indices, h[upper_indices], bc_type="natural")(indices)
        lower = CubicSpline(lower_indices, h[lower_indices], bc_type="natural")(indices)
        mean = (upper + lower) / 2
        candidate = h - mean
        sd = np.mean(mean ** 2) / max(np.mean(h ** 2), 1e-20)
        normalized_mean = np.mean(np.abs(mean)) / max(np.sqrt(np.mean(h ** 2)), 1e-20)
        extrema_count = len(find_peaks(candidate)[0]) + len(find_peaks(-candidate)[0])
        zero_count = np.sum(candidate[1:] * candidate[:-1] < 0)
        h = candidate
        if sd <= .05 and normalized_mean <= .05 and abs(extrema_count - zero_count) <= 1:
            break
    return h


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT / "manifests" / "dsp_validation.json")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--benchmark-lengths", type=int, nargs="+", default=[48000, 160000])
    args = parser.parse_args()
    config = json.loads((PROJECT / "configs" / "gpu.json").read_text())
    if os.environ.get("CUDA_VISIBLE_DEVICES") != config["gpu_uuid"]:
        raise RuntimeError("Use the exact-UUID GPU launcher")
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name(0) != config["gpu_name"]:
        raise RuntimeError("The required Ada GPU is not exclusively visible")
    report = {"status": "running", "gpu": torch.cuda.get_device_name(0),
              "gpu_uuid": config["gpu_uuid"], "torch_version": torch.__version__,
              "hht_method": "bounded piecewise-linear CUDA EMD approximation; 6 IMFs, <=10 sifts",
              "thresholds": {"reconstruction_relative_rms_max": 1e-5,
                             "pure_tone_if_relative_error_max": .05,
                             "wavelet_peak_relative_error_max": .15,
                             "synthetic_snr_db_absolute_error_max": 1e-3},
              "benchmarks": [], "limitations": ["Numerical fixtures do not validate real-world detection",
                  "Spline-EMD agreement is diagnostic, not equivalence", "Timing is a pilot; no full workload throughput claim"]}
    atomic_json(args.output, report)
    with torch.inference_mode():
        t = torch.arange(16384, device="cuda", dtype=torch.float32) / 16000
        tone = torch.sin(2 * torch.pi * 440 * t)[None]
        feature, diag = hht_features(tone, return_diagnostics=True)
        tone_if = float(diag["mean_instantaneous_frequency_hz"][0, 0])
        reconstruction = float(diag["reconstruction_relative_rms"].max())
        power, _, frequencies, coverage = morlet_power(tone)
        wavelet_peak = float(frequencies[power[0].argmax()])
        report["known_tone"] = {"true_hz": 440, "imf0_mean_if_hz": tone_if,
                                 "if_relative_error": abs(tone_if - 440) / 440,
                                 "wavelet_peak_hz": wavelet_peak,
                                 "wavelet_relative_error": abs(wavelet_peak - 440) / 440,
                                 "reconstruction_relative_rms": reconstruction,
                                 "sift_counts": diag["sift_counts"].tolist(),
                                 "unconverged_or_extrema_failed": diag["unconverged_or_extrema_failed"].tolist()}
        fixture_time = torch.arange(4096, device="cuda", dtype=torch.float32) / 16000
        fixture = torch.sin(2 * torch.pi * 128 * fixture_time) + .3 * torch.sin(2 * torch.pi * 1024 * fixture_time)
        imfs, _, _ = emd_linear(fixture[None])
        reference = classic_first_imf_reference(fixture.cpu().numpy())
        reference_gpu = torch.as_tensor(reference, device="cuda", dtype=torch.float32)
        edge = slice(256, -256)
        disagreement = ((imfs[0, 0, edge] - reference_gpu[edge]).square().mean()
                        / reference_gpu[edge].square().mean().clamp_min(1e-20)).sqrt()
        report["classical_spline_reference"] = {"fixture_samples": 4096, "reference_device": "CPU small correctness fixture only",
                                                 "imf0_relative_rms_difference_interior": float(disagreement),
                                                 "interpretation": "Approximation discrepancy; not a pass/fail equivalence test"}
        synthetic = generate_batch(3, 16384, seed=713, source_counts=[0, 1, 2])
        mask = synthetic["event_masks"]
        active = synthetic["source_active"]
        measured_snr = 10 * torch.log10((synthetic["stems"].square() * mask).sum(-1).clamp_min(1e-20)
                                       / (synthetic["background"][:, None].square() * mask).sum(-1).clamp_min(1e-20))
        snr_error = float((measured_snr[active] - synthetic["snr_db"][active]).abs().max())
        report["synthetic_checks"] = {"source_counts": synthetic["source_count"].tolist(),
                                       "exact_mixture_reconstruction": bool(torch.equal(synthetic["audio"], synthetic["background"] + synthetic["stems"].sum(1))),
                                       "maximum_snr_db_error": snr_error,
                                       "maximum_stem_value_outside_label_mask": float(synthetic["stems"][~mask].abs().max())}
        for n in args.benchmark_lengths:
            batch = generate_batch(args.batch_size, n, seed=1000 + n)
            for group in ("fourier", "stft", "wavelet", "hht"):
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                output = extract_features(batch["audio"], groups=(group,))
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                record = {"batch_size": args.batch_size, "samples": n, "representation": group,
                          "feature_dimension": output["features"].shape[1], "seconds": elapsed,
                          "milliseconds_per_clip": elapsed * 1000 / args.batch_size,
                          "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024 ** 2,
                          "finite": bool(torch.isfinite(output["features"]).all())}
                if group == "hht":
                    d = output["diagnostics"]["hht"]
                    record.update({"mean_sifts": float(d["sift_counts"].float().mean()),
                                   "unconverged_or_extrema_failed_fraction": float(d["unconverged_or_extrema_failed"].float().mean()),
                                   "maximum_reconstruction_relative_rms": float(d["reconstruction_relative_rms"].max())})
                report["benchmarks"].append(record)
                print(json.dumps(record), flush=True)
                atomic_json(args.output, report)
        checks = {"tone_if": abs(tone_if - 440) / 440 <= .05,
                  "wavelet_frequency": abs(wavelet_peak - 440) / 440 <= .15,
                  "reconstruction": reconstruction <= 1e-5,
                  "synthetic_snr": snr_error <= 1e-3,
                  "synthetic_exact_mixture": report["synthetic_checks"]["exact_mixture_reconstruction"],
                  "synthetic_support": report["synthetic_checks"]["maximum_stem_value_outside_label_mask"] == 0,
                  "all_features_finite": all(row["finite"] for row in report["benchmarks"])}
        report["checks"] = checks
        hht_failures = [row["unconverged_or_extrema_failed_fraction"] for row in report["benchmarks"] if row["representation"] == "hht"]
        hht_limited = any(value > .5 for value in hht_failures)
        report["hht_readiness"] = "exploratory_approximation_only" if hht_limited else "requires_domain_validity_study"
        report["status"] = ("basic_numerics_passed_hht_limited" if hht_limited else "basic_numerics_passed") if all(checks.values()) else "failed"
        if hht_limited:
            report["limitations"].append("Most noisy-scene modes reached bounded sifting without strict IMF convergence; HHT is exploratory and not validated classical decomposition")
        atomic_json(args.output, report)
        print(json.dumps({"status": report["status"], "checks": checks, "output": str(args.output)}), flush=True)
        return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
