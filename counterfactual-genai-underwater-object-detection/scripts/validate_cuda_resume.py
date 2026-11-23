#!/usr/bin/env python3
"""Cross-process, tiny CUDA checkpoint replay and corrupt-generation recovery."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from uac.runtime import CheckpointStore, atomic_json, completed_stage, fingerprint, sha256_file

CONFIG = {"test": "cuda_resume_v1", "seed": 173, "batch_size": 4}
EXPECTED_UUID = "GPU-10d1f16f-9e79-08bb-b2ba-3353c04422cf"


def serializable(value):
    import torch
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(item) for item in value]
    return value


def worker(phase, directory):
    import numpy as np
    import torch
    if os.environ.get("CUDA_VISIBLE_DEVICES") != EXPECTED_UUID:
        raise RuntimeError("Validation must run through the exact Ada UUID launcher")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one CUDA device required")
    if torch.cuda.get_device_name(0) != "NVIDIA RTX 5000 Ada Generation":
        raise RuntimeError("Unexpected GPU")
    torch.use_deterministic_algorithms(True)
    seed = 173 if phase == "baseline" else 987
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.ReLU(),
                                torch.nn.Dropout(0.2), torch.nn.Linear(16, 1)).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
    store = CheckpointStore(directory / ("fallback" if phase == "fallback" else "checkpoints"),
                            config=CONFIG, data_fingerprint="synthetic-indexed-fixture-v1",
                            code_fingerprint=sha256_file(__file__))

    def step(index):
        # A stable sample cursor controls the deterministic data; augmentation
        # and dropout exercise restored stochastic state on all four RNGs.
        signatures = {"python": random.random(), "numpy": float(np.random.rand()),
                      "torch_cpu": torch.rand(3).tolist()}
        x = torch.arange(32, device="cuda", dtype=torch.float32).reshape(4, 8) / 32
        x = x + index * 0.1 + 0.01 * torch.randn_like(x)
        y = torch.sin(x.sum(dim=1, keepdim=True))
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        torch.cuda.synchronize()
        return {"loss": float(loss.item()), "rng_signatures": signatures,
                "model": serializable(model.state_dict()),
                "optimizer": serializable(optimizer.state_dict()),
                "scheduler": serializable(scheduler.state_dict()),
                "next_batch_cursor": index + 1}

    if phase == "baseline":
        for index in (0, 1):
            result = step(index)
            store.save_training(model, optimizer, scheduler=scheduler,
                                epoch=0, batch_cursor=index + 1, global_step=index + 1,
                                extra={"epoch_order": [0, 1, 2], "next_sample_id": index + 1})
        atomic_json(directory / "expected_step2.json", result)
        atomic_json(directory / "expected_step3.json", step(2))
    else:
        cursor = store.load_training(model, optimizer, scheduler=scheduler)
        expected_step = 1 if phase == "fallback" else 2
        assert cursor["global_step"] == expected_step, cursor
        assert cursor["epoch"] == 0
        assert cursor["batch_cursor"] == expected_step
        assert cursor["extra"]["next_sample_id"] == expected_step
        result = step(cursor["batch_cursor"])
        atomic_json(directory / f"{phase}_result.json", result)
        atomic_json(directory / f"{phase}_recovery.json", cursor["recovery"])
    return {"torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["baseline", "resumed", "fallback"])
    parser.add_argument("--directory", type=Path)
    args = parser.parse_args()
    if args.phase:
        worker(args.phase, args.directory)
        return 0
    directory = ROOT / ".local" / "validation" / f"cuda-resume-{uuid.uuid4().hex[:12]}"
    directory.mkdir(parents=True)
    env = os.environ.copy()
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    for phase in ("baseline", "resumed"):
        subprocess.run([sys.executable, __file__, "--phase", phase,
                        "--directory", str(directory)], cwd=ROOT, env=env, check=True)
    expected3 = json.loads((directory / "expected_step3.json").read_text())
    resumed = json.loads((directory / "resumed_result.json").read_text())
    if expected3 != resumed:
        raise AssertionError("Cross-process next optimizer step did not match exactly")
    clean_index = json.loads((directory / "checkpoints" / "latest.json").read_text())
    clean_latest = directory / "checkpoints" / clean_index["entries"][0]["payload"]
    clean_sha = sha256_file(clean_latest)
    shutil.copytree(directory / "checkpoints", directory / "fallback")
    corrupt_copy = directory / "fallback" / clean_index["entries"][0]["payload"]
    corrupt_copy.write_bytes(b"intentional interrupted payload fixture")
    subprocess.run([sys.executable, __file__, "--phase", "fallback",
                    "--directory", str(directory)], cwd=ROOT, env=env, check=True)
    expected2 = json.loads((directory / "expected_step2.json").read_text())
    fallback = json.loads((directory / "fallback_result.json").read_text())
    if expected2 != fallback:
        raise AssertionError("Previous-generation recovery did not reproduce next step")
    if sha256_file(clean_latest) != clean_sha:
        raise AssertionError("Original clean checkpoint changed")
    recovery = json.loads((directory / "fallback_recovery.json").read_text())
    assert len(recovery["rejected"]) == 1
    result = {"status": "passed", "gpu_uuid": EXPECTED_UUID,
              "cross_process_next_step_exact_match": True,
              "optimizer_scheduler_and_rng_match": True, "data_cursor_restored": True,
              "corrupt_latest_fallback_exact_match": True,
              "original_clean_checkpoint_preserved": True,
              "fixture_directory": str(directory.relative_to(ROOT)),
              "limitation": "Small deterministic CUDA fixture, not the final research models."}
    result_path = ROOT / "manifests" / "cuda_resume_validation.json"
    atomic_json(result_path, result)
    completed_stage(ROOT / "manifests" / "cuda_resume_validation_receipt.json",
                    stage="cuda_resume_validation", config_hash=fingerprint(CONFIG),
                    artifacts=[result_path, directory / "resumed_result.json",
                               directory / "fallback_result.json", clean_latest],
                    data_fingerprint="synthetic-indexed-fixture-v1")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
