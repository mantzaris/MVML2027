#!/usr/bin/env python3
"""Durable, CUDA-only frozen Wav2Vec2 embeddings from verified real shards.

Input waveforms remain physically scaled in the real cache. For this pretrained
speech foundation comparator only, apply its official zero-mean/unit-variance
normalization on CUDA, with variance epsilon 1e-7. Equal-length inputs need no
padding/attention mask. Frozen last-layer mean and population SD yield 1536
features. This is an audio transfer baseline, not an underwater foundation model.
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
import torch
from transformers import AutoModel
from uac.runtime import (RunLock, StopRequested, EventLedger, atomic_bytes, atomic_json,
                         fingerprint, project_path, sha256_file, source_fingerprint, utc_now)
from extract_real_features import _safe_gpu, _receipt_if_valid, _check_test_gate

MODEL_NAME = "facebook/wav2vec2-base-960h"


def load_index(split, root="data/features/real"):
    path = project_path(root) / split / "index.json"
    index = json.loads(path.read_text())
    if index.get("status") != "complete" or index.get("completed_rows") != index.get("expected_rows"):
        raise RuntimeError(f"Feature split not complete: {path}")
    total = 0
    for record in index["shards"]:
        payload = project_path(record["payload"])
        if not payload.is_file() or payload.stat().st_size != record["payload_bytes"] or sha256_file(payload) != record["payload_sha256"]:
            raise RuntimeError(f"Feature shard integrity check failed: {payload}; resume its extractor first")
        total += record["count"]
    if total != index["expected_rows"]:
        raise RuntimeError("Feature index count does not match its shards")
    return index, sha256_file(path)


def load_shard(record):
    # Only trusted self-created payloads with an already verified digest are read.
    payload = torch.load(project_path(record["payload"]), map_location="cpu", weights_only=False)
    if payload["ids"] != record["ids"] or len(payload["features"]) != record["count"]:
        raise RuntimeError("Feature shard identities or dimensions differ from receipt")
    return payload


def normalize_for_foundation(audio):
    if audio.device.type != "cuda":
        raise RuntimeError("Foundation normalization must remain on CUDA")
    return (audio - audio.mean(-1, keepdim=True)) / (audio.var(-1, keepdim=True, correction=0) + 1e-7).sqrt()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", nargs="+", choices=("train", "development", "calibration", "test"), default=["train", "development"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--input-dir", default="data/features/real")
    parser.add_argument("--output-dir", default="data/features/foundation")
    parser.add_argument("--max-shards", type=int)
    parser.add_argument("--allow-test-features", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or (args.max_shards is not None and args.max_shards < 1):
        parser.error("Batch and shard limits must be positive")
    gpu = _safe_gpu()
    lock = json.loads((PROJECT / "manifests/model_lock.json").read_text())[MODEL_NAME]
    if not lock.get("download_complete"):
        raise RuntimeError("Pinned foundation model download is incomplete")
    model_dir = project_path(lock["path"])
    preprocessor = json.loads((model_dir / "preprocessor_config.json").read_text())
    if preprocessor.get("sampling_rate") != 16000 or not preprocessor.get("do_normalize") or preprocessor.get("return_attention_mask"):
        raise RuntimeError("Pinned feature-extractor settings differ from the implemented official normalization")
    artifact_paths = list(model_dir.glob("*.safetensors")) + [model_dir / "config.json", model_dir / "preprocessor_config.json"]
    if not any(path.suffix == ".safetensors" for path in artifact_paths):
        raise RuntimeError("Pinned safetensors weights are absent")
    artifact_hashes = {str(path.relative_to(PROJECT)): sha256_file(path) for path in artifact_paths}
    config = {"schema_version": 1, "model": MODEL_NAME, "revision": lock["revision"],
              "weight_artifact_hashes": artifact_hashes, "dtype": "float32", "batch_size": args.batch_size,
              "sample_rate": 16000, "normalize": "CUDA population variance; epsilon 1e-7",
              "pooling": "last_hidden_state temporal mean plus population standard deviation",
              "input_dimension": 1536, "frozen": True}
    config_hash = fingerprint(config)
    code_hash = source_fingerprint(["scripts/extract_foundation.py", "scripts/extract_real_features.py", "src/uac/runtime.py"])
    output_root = project_path(args.output_dir)
    completed_new = 0
    with RunLock("runs/training.lock"), StopRequested() as stop:
        ledger = EventLedger()
        # Do not acquire GPU weights or inspect held-out payloads before the gate.
        prepared = []
        for split in args.split:
            index_path = project_path(args.input_dir) / split / "index.json"
            metadata = json.loads(index_path.read_text())
            _check_test_gate([split], args, metadata["data_fingerprint"])
            prepared.append((split, *load_index(split, args.input_dir)))
        model = AutoModel.from_pretrained(str(model_dir), local_files_only=True).to(device="cuda", dtype=torch.float32).eval()
        model.requires_grad_(False)
        if model.config.hidden_size != 768 or any(p.device.type != "cuda" for p in model.parameters()):
            raise RuntimeError("Foundation model dimensions or CUDA residency mismatch")
        ledger.append("foundation_extraction_started", model=MODEL_NAME, revision=lock["revision"], config_hash=config_hash)
        for split, source_index, source_index_hash in prepared:
            directory = output_root / split
            directory.mkdir(parents=True, exist_ok=True)
            index_path = directory / "index.json"
            if index_path.is_file():
                previous = json.loads(index_path.read_text())
                if (previous.get("status") == "complete" and previous.get("config_hash") == config_hash
                        and previous.get("code_fingerprint") == code_hash
                        and previous.get("source_index_sha256") == source_index_hash):
                    valid = all(project_path(r["payload"]).is_file()
                                and project_path(r["payload"]).stat().st_size == r["payload_bytes"]
                                and sha256_file(r["payload"]) == r["payload_sha256"] for r in previous["shards"])
                    if valid:
                        print(json.dumps({"status": "already_complete", "split": split, "rows": previous["completed_rows"]}), flush=True)
                        continue
            index = {"schema_version": 1, "status": "running", "split": split, "config": config,
                     "config_hash": config_hash, "code_fingerprint": code_hash,
                     "data_fingerprint": source_index["data_fingerprint"], "source_index_sha256": source_index_hash,
                     "gpu_uuid": gpu["gpu_uuid"], "expected_rows": source_index["expected_rows"],
                     "completed_rows": 0, "feature_dimension": 1536, "shards": [], "created_or_resumed_utc": utc_now()}
            for source_record in source_index["shards"]:
                begin, count = source_record["begin"], source_record["count"]
                stem = f"shard-{begin:06d}-{begin + count:06d}"
                payload_path, receipt_path = directory / f"{stem}.pt", directory / f"{stem}.json"
                expected = {"schema_version": 1, "split": split, "begin": begin, "count": count,
                            "ids": source_record["ids"], "config_hash": config_hash,
                            "code_fingerprint": code_hash, "data_fingerprint": source_index["data_fingerprint"],
                            "source_payload_sha256": source_record["payload_sha256"],
                            "payload": str(payload_path.relative_to(PROJECT))}
                receipt = _receipt_if_valid(payload_path, receipt_path, expected, ledger)
                if receipt is None:
                    if stop.requested or (args.max_shards is not None and completed_new >= args.max_shards):
                        index["status"] = "paused" if stop.requested else "pilot_limit"
                        atomic_json(index_path, index)
                        return 75 if stop.requested else 0
                    source = load_shard(source_record)
                    if source["sample_rate"] != 16000:
                        raise RuntimeError("Foundation requires the verified 16 kHz derivative")
                    torch.cuda.reset_peak_memory_stats()
                    started = time.perf_counter()
                    embeddings = []
                    with torch.inference_mode():
                        for start in range(0, count, args.batch_size):
                            audio = source["audio"][start:start + args.batch_size].to("cuda")
                            normalized = normalize_for_foundation(audio)
                            hidden = model(input_values=normalized).last_hidden_state
                            embedding = torch.cat((hidden.mean(1), hidden.std(1, correction=0)), -1)
                            if embedding.shape[1] != 1536 or not bool(torch.isfinite(embedding).all()):
                                raise RuntimeError("Invalid foundation embedding")
                            embeddings.append(embedding.cpu())
                            if stop.requested:
                                index["status"] = "paused"
                                atomic_json(index_path, index)
                                return 75
                    torch.cuda.synchronize()
                    payload = {"schema_version": 1, "ids": source["ids"], "rows": source["rows"],
                               "groups": source["groups"], "labels": source["labels"], "features": torch.cat(embeddings),
                               "model": MODEL_NAME, "revision": lock["revision"], "config_hash": config_hash,
                               "data_fingerprint": source_index["data_fingerprint"], "code_fingerprint": code_hash}
                    buffer = io.BytesIO()
                    torch.save(payload, buffer)
                    atomic_bytes(payload_path, buffer.getvalue())
                    receipt = {**expected, "payload_sha256": sha256_file(payload_path),
                               "payload_bytes": payload_path.stat().st_size, "completed_utc": utc_now(),
                               "seconds_including_transfers": time.perf_counter() - started,
                               "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 1024 ** 2}
                    atomic_json(receipt_path, receipt)
                    completed_new += 1
                    ledger.append("foundation_shard_committed", split=split, begin=begin, count=count,
                                  payload=expected["payload"], payload_sha256=receipt["payload_sha256"])
                    print(json.dumps({"split": split, "completed_through": begin + count,
                                      "expected_rows": index["expected_rows"], "seconds": receipt["seconds_including_transfers"]}), flush=True)
                index["shards"].append(receipt)
                index["completed_rows"] += count
                atomic_json(index_path, index)
            index["status"] = "complete"
            index["completed_utc"] = utc_now()
            atomic_json(index_path, index)
            ledger.append("foundation_split_complete", split=split, rows=index["completed_rows"], index=str(index_path.relative_to(PROJECT)))
            print(json.dumps({"status": "complete", "split": split, "rows": index["completed_rows"]}), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as exc:
        print(json.dumps({"status": "error", "error": str(exc), "cpu_fallback": False}), file=sys.stderr)
        raise SystemExit(2)
