#!/usr/bin/env python3
"""Controlled validity checks for central conditional probe and full-record null.

Uses the actual trained flow checkpoint and known synthetic stems. No real data
or held-out detector outcomes are read. A simple measurement fixture tests score
algebra; its numbers are not detection performance. Original-shard seeded replay
is checked, not arbitrary minibatch-invariant generation.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time
import unittest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "tests"))
import torch
from uac.models import interval_mask, matched_sham
from uac.synthetic import generate_batch
from uac.verification import (load_flow, generate_intervention_features, intervention_components,
                              verification_score, band_mismatch)
from uac.runtime import RunLock, atomic_json, sha256_file, source_fingerprint, utc_now
from extract_real_features import _safe_gpu


class RecordingFlow:
    def __init__(self, model):
        self.model = model
        self.replacements = []
        self.after_replacement_rng = []

    def replace(self, audio, mask, **kwargs):
        value = self.model.replace(audio, mask, **kwargs)
        self.replacements.append(value.clone())
        self.after_replacement_rng.append(kwargs["generator"].get_state())
        return value


class MeasurementFixtureBank:
    def view_logits(self, features):
        return features[:, [0, 32, 96]].clamp(-8, 8)


def summary(values):
    if not values:
        return {"n": 0, "median": None, "p10": None, "p90": None}
    x = torch.tensor(values, device="cuda", dtype=torch.float32)
    return {"n": len(values), "median": float(x.median()),
            "p10": float(torch.quantile(x, .1)), "p90": float(torch.quantile(x, .9))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--output", default="manifests/verification_validation.json")
    args = parser.parse_args()
    _safe_gpu()
    os.environ["UAC_RUN_GPU_TESTS"] = "1"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    output = {"status": "running", "recorded_utc": utc_now(),
              "scope": "controlled numerical and intervention-validity checks only; not real detection performance",
              "flow_checkpoint_sha256": sha256_file("checkpoints/background_flow/final.pt"),
              "source_fingerprint": source_fingerprint(["src/uac/verification.py", "src/uac/models.py", "src/uac/synthetic.py", "scripts/validate_verification.py"]),
              "settings": vars(args), "median_convention": "lower central order statistic, including lower MAD",
              "source_seed_namespace": 81000000, "intervention_seed_namespace": 82000000,
              "probes": {}, "limitations": ["Synthetic harmonic/AM/chirp-like signals, not validated vessels",
                   "Central-probe target support and SNR are known only for these controlled scenes",
                   "Replacing an interval may suppress overlapping unrelated sources",
                   "Views share one waveform and generated draw; they are not independent stochastic observations",
                   "Valid means engineering checks passed, not physically correct source removal",
                   "Replaying one case regenerates its original seeded shard and selects that case; changing batch geometry is not RNG-invariant"]}
    with RunLock("runs/training.lock"):
        suite = unittest.defaultTestLoader.discover(str(PROJECT / "tests"), pattern="test_verification.py")
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        output["unit_checks"] = {"run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors), "passed": result.wasSuccessful()}
        if not result.wasSuccessful():
            output["status"] = "failed_unit_check"
            atomic_json(args.output, output)
            return 1
        flow = load_flow()
        bank = MeasurementFixtureBank()
        all_support_passed = True
        for probe in ("central", "full"):
            label = "central_context_conditioned_probe" if probe == "central" else "full_record_unconditional_null"
            records, target_before, target_after, sham_target, overlap_before, overlap_after = [], [], [], [], [], []
            max_outside, max_energy, max_band_cache_difference = 0., 0., 0.
            max_score_difference, replay_equal, max_replay_difference = 0., True, 0.
            started = time.perf_counter()
            for start in range(0, args.scenes, args.batch_size):
                count = min(args.batch_size, args.scenes - start)
                counts = [(start + i) % 3 for i in range(count)]
                batch = generate_batch(count, num_samples=48000, seed=81000000 + start, family="train", source_counts=counts)
                audio = batch["audio"]
                mask = torch.ones_like(audio) if probe == "full" else interval_mask(count, 48000, 12000, 36000, taper_samples=160)
                recorder = RecordingFlow(flow)
                seed = 82000000 + start
                cache = generate_intervention_features(recorder, audio, seed=seed, k=args.k, steps=args.steps, probe=probe)
                components = intervention_components(bank, cache, {})
                batch_scores = verification_score(components, ["fourier", "stft", "wavelet"])
                single_scores = torch.cat([verification_score(intervention_components(bank, {key: value[i:i + 1] for key, value in cache.items()}, {}),
                                                         ["fourier", "stft", "wavelet"]) for i in range(count)])
                max_score_difference = max(max_score_difference, float((batch_scores - single_scores).abs().max()))
                if start == 0:
                    replay = generate_intervention_features(flow, audio, seed=seed, k=args.k, steps=args.steps, probe=probe)
                    replay_equal = replay_equal and all(torch.equal(cache[key], replay[key]) for key in cache)
                    for key in cache:
                        if cache[key].dtype != torch.bool:
                            max_replay_difference = max(max_replay_difference, float((cache[key] - replay[key]).abs().max()))
                for draw, replacement in enumerate(recorder.replacements):
                    rng = torch.Generator(device="cuda")
                    rng.set_state(recorder.after_replacement_rng[draw])
                    noise = torch.randn(audio.shape, device="cuda", generator=rng)
                    sham, diagnostic = matched_sham(audio, replacement, mask, noise)
                    exterior = mask == 0
                    if bool(exterior.any()):
                        exterior_rms = (((replacement - audio).square() * exterior).sum(-1) / exterior.sum(-1).clamp_min(1)).sqrt()
                        normalized_outside = exterior_rms / audio.square().mean(-1).sqrt().clamp_min(1e-8)
                        max_outside = max(max_outside, float(normalized_outside.max()))
                        all_support_passed = all_support_passed and bool(torch.equal(sham[exterior], audio[exterior]))
                    max_energy = max(max_energy, float(diagnostic["relative_energy_mismatch"].max()))
                    check_band = band_mismatch(replacement - audio, sham - audio)
                    max_band_cache_difference = max(max_band_cache_difference, float((check_band - cache["band_error"][:, draw]).abs().max()))
                    for i in range(count):
                        individual = {"scene_id": f"verification-fixture-{start + i:04d}", "source_count": counts[i],
                                      "draw": draw, "candidate_valid_all_draws": bool(cache["valid"][i]),
                                      "energy_mismatch": float(cache["energy_mismatch"][i, draw]),
                                      "band_error": float(cache["band_error"][i, draw]),
                                      "outside_error": float(cache["outside_error"][i, draw]),
                                      "boundary_penalty": float(cache["boundary_penalty"][i, draw]),
                                      "rms_ratio": float(cache["rms_ratio"][i, draw]),
                                      "target_probe_energy_fraction": None}
                        for source in range(counts[i]):
                            stem = batch["stems"][i, source]
                            supported_energy = (stem.square() * mask[i]).sum()
                            total_energy = stem.square().sum().clamp_min(1e-20)
                            energy_fraction = float(supported_energy / total_energy)
                            if source == 0:
                                individual["target_probe_energy_fraction"] = energy_fraction
                            if energy_fraction < .1 or float(supported_energy) < 1e-10:
                                continue
                            project = lambda value: float((((value - batch["background"][i]) * stem * mask[i]).sum() / supported_energy).abs())
                            before, after, sham_value = project(audio[i]), project(replacement[i]), project(sham[i])
                            if source == 0:
                                target_before.append(before); target_after.append(after); sham_target.append(sham_value)
                            else:
                                overlap_before.append(before); overlap_after.append(after)
                        records.append(individual)
            torch.cuda.synchronize()
            scene_valid = [row["candidate_valid_all_draws"] for row in records if row["draw"] == 0]
            output["probes"][label] = {"probe": probe, "probe_seconds": [.75, 2.25] if probe == "central" else [0., 3.],
                "n_scenes": args.scenes, "k": args.k, "candidate_validity_rate": sum(scene_valid) / len(scene_valid),
                "maximum_normalized_exterior_rms": max_outside, "maximum_sham_rms_mismatch": max_energy,
                "recomputed_band_metric_max_difference": max_band_cache_difference,
                "batch_versus_each_case_score_max_difference": max_score_difference,
                "original_shard_seeded_replay_bitwise_equal": replay_equal, "replay_max_difference": max_replay_difference,
                "elapsed_seconds": time.perf_counter() - started,
                "projection_eligibility": "source contributes at least10% of its energy inside probe and supported energy>1e-10; each draw counted",
                "absolute_known_stem_projection_before": summary(target_before),
                "absolute_known_stem_projection_after": summary(target_after),
                "absolute_known_stem_projection_sham": summary(sham_target),
                "overlapping_other_source_projection_before": summary(overlap_before),
                "overlapping_other_source_projection_after": summary(overlap_after),
                "projection_limitation": "Matched-filter sensitivity after subtracting known simulated background; not a learned detector, separation metric or causal estimate",
                "records": records}
            atomic_json(args.output, output)
            print(json.dumps({key: value for key, value in output["probes"][label].items() if key != "records"}), flush=True)
        checks = {"unit_checks": output["unit_checks"]["passed"], "sham_exterior_exact": all_support_passed,
                  "replacement_exterior_tolerance": all(row["maximum_normalized_exterior_rms"] < 1e-5 for row in output["probes"].values()),
                  "sham_energy_match": all(row["maximum_sham_rms_mismatch"] < .05 for row in output["probes"].values()),
                  "band_metric_replay": all(row["recomputed_band_metric_max_difference"] == 0 for row in output["probes"].values()),
                  "case_score_exact": all(row["batch_versus_each_case_score_max_difference"] == 0 for row in output["probes"].values()),
                  "seeded_original_shard_replay_exact": all(row["original_shard_seeded_replay_bitwise_equal"] for row in output["probes"].values())}
        output["checks"] = checks
        output["status"] = "numerical_checks_passed" if all(checks.values()) else "numerical_check_failed"
        output["note"] = "Validity rates and target/sham projections are observations, not required favorable outcomes. Rejected candidates retain the base score."
        atomic_json(args.output, output)
        print(json.dumps({"status": output["status"], "checks": checks, "output": args.output}), flush=True)
        return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
