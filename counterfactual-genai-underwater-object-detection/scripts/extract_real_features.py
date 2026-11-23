#!/usr/bin/env python3
"""Resumable CUDA feature extraction for the frozen real-audio manifest.

Preserves native WAVs; stores anti-aliased 16 kHz waveforms plus all 184 features
in durable shards. Primary models select the first 152 (Fourier/STFT/wavelet);
bounded-HHT features remain exploratory. Every shard has independent checksums,
source/config/code fingerprints and original grouping metadata. Power loss can
lose at most an unfinished shard; restarting validates and skips completed ones.

Examples (through run_on_ada.py):
  .venv/bin/python scripts/extract_real_features.py --split train development
  .venv/bin/python scripts/extract_real_features.py --split train --max-shards 1
  .venv/bin/python scripts/extract_real_features.py --split test --allow-test-features

Test extraction additionally requires runs/test_freeze.json with data_fingerprint
matching the SHA256 of data/manifests/planned_real.jsonl. Each verified split
manifest and original audio hash is recorded separately; completed splits can
be processed while other splits are still downloading. This script fits no model or
normalization and computes no detector outcome.
"""
from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
import numpy as np
import torch
import torchaudio.functional as AF
from uac.data import load_rows, decode_wav
from uac.dsp import ALL_GROUPS, PRIMARY_GROUPS, GROUP_DIMS, feature_names, extract_features
from uac.runtime import (RunLock, StopRequested, EventLedger, atomic_bytes, atomic_json,
                         fingerprint, project_path, sha256_file, source_fingerprint, utc_now)


def _configuration(args):
    return {"schema_version": 1, "sample_rate": args.sample_rate,
            "groups": list(ALL_GROUPS), "feature_dimensions": GROUP_DIMS,
            "primary_groups": list(PRIMARY_GROUPS), "primary_indices": list(range(152)),
            "shard_size": args.shard_size, "batch_size": args.batch_size,
            "resample": {"method": "sinc_interp_kaiser", "lowpass_filter_width": 64,
                         "rolloff": .9475937167399596, "beta": 14.769656459379492,
                         "device": "cuda", "preserve_amplitude": True},
            "hht_status": "bounded piecewise-linear approximation; exploratory only",
            "waveform_policy": "native channel from manifest; no padding, concatenation, gain normalization or clipping"}


def _safe_gpu():
    config = json.loads((PROJECT / "configs/gpu.json").read_text())
    if os.environ.get("CUDA_VISIBLE_DEVICES") != config["gpu_uuid"]:
        raise RuntimeError("Run through scripts/run_on_ada.py with the exact Ada UUID")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one CUDA device must be visible; no CPU fallback")
    props = torch.cuda.get_device_properties(0)
    if props.name != config["gpu_name"] or props.total_memory < config["minimum_memory_mib"] * 1024 ** 2:
        raise RuntimeError("Visible CUDA device is not the required 32 GB Ada")
    return config


def _check_test_gate(splits, args, data_fingerprint):
    if "test" not in splits:
        return
    # Computing features is itself locked to prevent accidental early test inspection.
    if not args.allow_test_features:
        raise RuntimeError("Test features are locked: explicitly set --allow-test-features after protocol freeze")
    freeze_path = PROJECT / "runs/test_freeze.json"
    if not freeze_path.is_file():
        raise RuntimeError("Test features are locked: runs/test_freeze.json does not exist")
    freeze = json.loads(freeze_path.read_text())
    if freeze.get("data_fingerprint") != data_fingerprint:
        raise RuntimeError("Test freeze data_fingerprint does not match the frozen planned cohort manifest")


def _receipt_if_valid(payload_path, receipt_path, expected, ledger):
    """Return a checked receipt, or preserve corrupted artifacts before replay."""
    if not receipt_path.exists():
        return None
    try:
        receipt = json.loads(receipt_path.read_text())
    except (OSError, ValueError):
        receipt = None
    if receipt is not None:
        for key, value in expected.items():
            if receipt.get(key) != value:
                raise RuntimeError(f"Existing shard is incompatible ({key}): {receipt_path}. Use a new output directory; do not mix runs.")
        if (payload_path.is_file() and payload_path.stat().st_size == receipt.get("payload_bytes")
                and sha256_file(payload_path) == receipt.get("payload_sha256")):
            return receipt
    suffix = f".corrupt-{time.time_ns()}"
    quarantined = []
    for path in (payload_path, receipt_path):
        if path.exists():
            target = path.with_name(path.name + suffix)
            os.replace(path, target)
            quarantined.append(str(target.relative_to(PROJECT)))
    ledger.append("feature_shard_corruption_replay", artifacts=quarantined)
    return None


def _decode_to_cuda(rows, analysis_rate, resample):
    decoded, rates, lengths = [], set(), set()
    for row in rows:
        # Original byte hashes are verified for each newly computed shard. Raw
        # decoding and stacking are I/O, while filtering/resampling stays on GPU.
        if not row.get("header_verified") or not row.get("sha256"):
            raise ValueError("Only fully verified real-manifest rows may be extracted")
        path = project_path(row["path"])
        if sha256_file(path) != row["sha256"]:
            raise ValueError(f"Raw audio checksum mismatch: {row['id']}")
        waveform, rate = decode_wav(PROJECT, row)
        decoded.append(waveform)
        rates.add(rate)
        lengths.add(len(waveform))
    if len(rates) != 1 or len(lengths) != 1:
        raise ValueError("Batch needs equal native rates and lengths; do not hide variable duration with padding")
    native_rate = next(iter(rates))
    audio = torch.from_numpy(np.stack(decoded)).to(device="cuda", dtype=torch.float32)
    if native_rate != analysis_rate:
        audio = AF.resample(audio, orig_freq=native_rate, new_freq=analysis_rate,
                            lowpass_filter_width=resample["lowpass_filter_width"],
                            rolloff=resample["rolloff"], resampling_method=resample["method"],
                            beta=resample["beta"])
    expected_length = (next(iter(lengths)) * analysis_rate + native_rate - 1) // native_rate
    if audio.device.type != "cuda" or audio.shape[-1] != expected_length or not bool(torch.isfinite(audio).all()):
        raise RuntimeError("Invalid CUDA analysis waveform")
    return audio


def _extract_shard(rows, args, config, stop):
    audio_parts, feature_parts, diagnostics = [], [], []
    group_slices = None
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.inference_mode():
        for begin in range(0, len(rows), args.batch_size):
            batch_rows = rows[begin:begin + args.batch_size]
            audio = _decode_to_cuda(batch_rows, args.sample_rate, config["resample"])
            result = extract_features(audio, args.sample_rate, groups=ALL_GROUPS)
            group_slices = result["group_slices"]
            audio_parts.append(audio.cpu())
            feature_parts.append(result["features"].cpu())
            d = result["diagnostics"]["hht"]
            diagnostics.append({key: value.cpu() for key, value in d.items()})
            if stop.requested:
                return None
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    combined_diagnostics = {key: torch.cat([part[key] for part in diagnostics]) for key in diagnostics[0]}
    payload = {"schema_version": 1, "ids": [row["id"] for row in rows],
               "rows": rows, "audio": torch.cat(audio_parts), "features": torch.cat(feature_parts),
               "labels": torch.tensor([row["label"] for row in rows], dtype=torch.float32),
               "groups": [row["group"] for row in rows], "sample_rate": args.sample_rate,
               "feature_names": feature_names(ALL_GROUPS), "group_slices": group_slices,
               "primary_indices": list(range(152)), "hht_diagnostics": combined_diagnostics}
    measured = {"seconds_including_io_and_transfers": elapsed,
                "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 1024 ** 2,
                "hht_flagged_mode_fraction": float(combined_diagnostics["unconverged_or_extrema_failed"].float().mean()),
                "audio_shape": list(payload["audio"].shape), "feature_shape": list(payload["features"].shape)}
    return payload, measured


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", nargs="+", choices=("train", "development", "calibration", "test"), default=["train", "development"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--shard-size", type=int, default=128)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--output-dir", type=Path, default=Path("data/features/real"))
    parser.add_argument("--max-shards", type=int, default=None, help="Pilot cap on newly computed shards in this invocation")
    parser.add_argument("--allow-test-features", action="store_true")
    args = parser.parse_args()
    if min(args.batch_size, args.shard_size) < 1 or args.sample_rate != 16000 or (args.max_shards is not None and args.max_shards < 1):
        parser.error("Positive batch/shard limits and the frozen 16000 Hz derivative are required")
    if len(set(args.split)) != len(args.split):
        parser.error("Each split may appear only once")
    output_root = project_path(args.output_dir)
    manifest = PROJECT / "data/manifests/planned_real.jsonl"
    if not manifest.is_file():
        raise RuntimeError("Frozen planned cohort manifest not available; run scripts/prepare_data.py first")
    data_hash = sha256_file(manifest)
    _check_test_gate(args.split, args, data_hash)
    gpu = _safe_gpu()
    config = _configuration(args)
    code_hash = source_fingerprint(["scripts/extract_real_features.py", "src/uac/dsp.py", "src/uac/data.py", "src/uac/runtime.py"])
    config_hash = fingerprint(config)
    newly_computed = 0
    with RunLock("runs/training.lock"), StopRequested() as stop:
        ledger = EventLedger()
        ledger.append("real_feature_extraction_started", splits=args.split, config_hash=config_hash,
                      data_fingerprint=data_hash, code_fingerprint=code_hash)
        for split in args.split:
            split_manifest = PROJECT / f"data/manifests/real_{split}.jsonl"
            if not split_manifest.is_file():
                split_manifest = PROJECT / "data/manifests/real.jsonl"
            if not split_manifest.is_file():
                raise RuntimeError(f"Verified {split} manifest is not complete yet")
            split_manifest_hash = sha256_file(split_manifest)
            rows = load_rows(PROJECT, split=split)
            rows_hash = fingerprint(rows)
            if not rows:
                raise ValueError(f"No rows in required partition {split}")
            directory = output_root / split
            directory.mkdir(parents=True, exist_ok=True)
            index_path = directory / "index.json"
            if index_path.is_file():
                previous = json.loads(index_path.read_text())
                if (previous.get("status") == "complete" and previous.get("config_hash") == config_hash
                        and previous.get("code_fingerprint") == code_hash and previous.get("data_fingerprint") == data_hash
                        and previous.get("rows_fingerprint") == rows_hash and previous.get("completed_rows") == len(rows)):
                    valid = all(project_path(r["payload"]).is_file()
                                and project_path(r["payload"]).stat().st_size == r["payload_bytes"]
                                and sha256_file(r["payload"]) == r["payload_sha256"] for r in previous["shards"])
                    if valid:
                        print(json.dumps({"status": "already_complete", "split": split, "rows": len(rows)}), flush=True)
                        continue
            index = {"schema_version": 1, "status": "running", "split": split,
                     "config": config, "config_hash": config_hash, "code_fingerprint": code_hash,
                     "data_fingerprint": data_hash, "split_manifest_sha256": split_manifest_hash,
                     "rows_fingerprint": rows_hash, "gpu_uuid": gpu["gpu_uuid"],
                     "expected_rows": len(rows), "completed_rows": 0, "shards": [],
                     "feature_names": feature_names(ALL_GROUPS),
                     "group_slices": {"fourier": [0, 32], "stft": [32, 96], "wavelet": [96, 152], "hht": [152, 184]},
                     "created_or_resumed_utc": utc_now()}
            for begin in range(0, len(rows), args.shard_size):
                subset = rows[begin:begin + args.shard_size]
                stem = f"shard-{begin:06d}-{begin + len(subset):06d}"
                payload_path = directory / f"{stem}.pt"
                receipt_path = directory / f"{stem}.json"
                expected = {"schema_version": 1, "split": split, "begin": begin, "count": len(subset),
                            "ids": [row["id"] for row in subset], "config_hash": config_hash,
                            "code_fingerprint": code_hash, "data_fingerprint": data_hash,
                            "split_manifest_sha256": split_manifest_hash, "rows_fingerprint": rows_hash,
                            "payload": str(payload_path.relative_to(PROJECT))}
                receipt = _receipt_if_valid(payload_path, receipt_path, expected, ledger)
                if receipt is None:
                    if stop.requested or (args.max_shards is not None and newly_computed >= args.max_shards):
                        index["status"] = "paused" if stop.requested else "pilot_limit"
                        atomic_json(index_path, index)
                        ledger.append("real_feature_extraction_paused", split=split, next_row=begin,
                                      reason="signal" if stop.requested else "pilot_limit")
                        return 75 if stop.requested else 0
                    extracted = _extract_shard(subset, args, config, stop)
                    if extracted is None:
                        index["status"] = "paused"
                        atomic_json(index_path, index)
                        ledger.append("real_feature_extraction_paused", split=split, next_row=begin, reason="signal_unfinished_shard_replay")
                        return 75
                    payload, measurements = extracted
                    payload.update(config_hash=config_hash, code_fingerprint=code_hash, data_fingerprint=data_hash)
                    buffer = io.BytesIO()
                    torch.save(payload, buffer)
                    atomic_bytes(payload_path, buffer.getvalue())
                    receipt = {**expected, "payload_sha256": sha256_file(payload_path),
                               "payload_bytes": payload_path.stat().st_size,
                               "completed_utc": utc_now(), "measurements": measurements}
                    atomic_json(receipt_path, receipt)
                    newly_computed += 1
                    ledger.append("real_feature_shard_committed", split=split, begin=begin, count=len(subset),
                                  payload=expected["payload"], payload_sha256=receipt["payload_sha256"], **measurements)
                    print(json.dumps({"split": split, "completed_through": begin + len(subset),
                                      "expected_rows": len(rows), **measurements}), flush=True)
                index["shards"].append(receipt)
                index["completed_rows"] += len(subset)
                atomic_json(index_path, index)
            index["status"] = "complete"
            index["completed_utc"] = utc_now()
            atomic_json(index_path, index)
            ledger.append("real_feature_split_complete", split=split, rows=len(rows),
                          index=str(index_path.relative_to(PROJECT)), data_fingerprint=data_hash)
            print(json.dumps({"status": "complete", "split": split, "rows": len(rows), "index": str(index_path)}), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as exc:
        print(json.dumps({"status": "error", "error": str(exc), "cpu_fallback": False}), file=sys.stderr)
        raise SystemExit(2)
