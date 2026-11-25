#!/usr/bin/env python3
"""Verify selected CUDA hardware, BF16 training operations, and HDC algebra."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys

PROJECT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional small local JSON report.")
    args = parser.parse_args()
    manifest = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "required_backend": "cuda", "cpu_fallback": False, "ready": False,
        "scope": "Hardware/environment smoke check, not a research experiment",
    }
    code = 2
    try:
        import torch
        manifest["torch"] = torch.__version__
        manifest["torch_cuda_runtime"] = torch.version.cuda
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Run scripts/run_gpu.py in a host shell with GPU access.")
        if torch.cuda.device_count() != 1:
            raise RuntimeError("Exactly one GPU must be exposed. Use scripts/run_gpu.py to select the Ada GPU.")
        device = torch.device("cuda:0")
        props = torch.cuda.get_device_properties(device)
        expected = os.environ.get("REGEN_EXPECTED_GPU_NAME", "NVIDIA RTX 5000 Ada Generation")
        if props.name != expected:
            raise RuntimeError(f"Expected {expected}, found {props.name}; refusing a different GPU.")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("The selected GPU/runtime does not support the required BF16 path.")
        torch.manual_seed(11)
        with torch.cuda.device(device):
            tensor = torch.ones((64, 64), device=device)
            result = tensor @ tensor
            if not torch.equal(result, torch.full_like(result, 64.0)):
                raise RuntimeError("CUDA matrix multiplication produced an unexpected result")
            vectors = torch.randint(0, 2, (2, 4096), device=device).float().mul_(2).sub_(1)
            if not torch.equal((vectors[0] * vectors[1]) * vectors[1], vectors[0]):
                raise RuntimeError("Bipolar HDC binding/unbinding identity failed")
            model = torch.nn.Sequential(
                torch.nn.Linear(64, 128), torch.nn.GELU(), torch.nn.Linear(128, 1)
            ).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            features = torch.randn(256, 64, device=device)
            labels = torch.randn(256, 1, device=device)
            previous = model[0].weight.detach().clone()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                predictions = model(features)
            loss = torch.nn.functional.mse_loss(predictions.float(), labels)
            if predictions.dtype != torch.bfloat16 or not torch.isfinite(loss):
                raise RuntimeError("BF16 forward/FP32 loss verification failed")
            loss.backward()
            if any(p.grad is None or p.grad.device.type != "cuda" or not torch.isfinite(p.grad).all()
                   for p in model.parameters()):
                raise RuntimeError("CUDA backward pass produced missing or invalid gradients")
            optimizer.step()
            torch.cuda.synchronize(device)
            if torch.equal(previous, model[0].weight):
                raise RuntimeError("Optimizer did not update model parameters")
        manifest.update({
            "ready": True, "device": str(device), "gpu": props.name,
            "physical_gpu_index": os.environ.get("REGEN_GPU_PHYSICAL_INDEX"),
            "driver": os.environ.get("REGEN_DRIVER_VERSION"),
            "vram_gib": round(props.total_memory / 2**30, 2),
            "compute_capability": f"{props.major}.{props.minor}",
            "bf16_supported": True,
            "checks": {"fp32_matmul": True, "hdc_bipolar_binding": True,
                       "bf16_forward_fp32_loss": True, "cuda_backward": True,
                       "optimizer_update": True},
        })
        code = 0
    except ImportError:
        manifest["reason"] = "PyTorch is not installed in this interpreter. Use scripts/run_gpu.py."
    except Exception as exc:
        manifest["reason"] = f"{type(exc).__name__}: {exc}"
    encoded = json.dumps(manifest, indent=2) + "\n"
    print(encoded, end="")
    if args.output:
        output = args.output if args.output.is_absolute() else PROJECT / args.output
        output = output.resolve()
        if not output.is_relative_to(PROJECT):
            print("Refusing to write a report outside this project.", file=sys.stderr)
            return 2
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
    return code

if __name__ == "__main__":
    sys.exit(main())
