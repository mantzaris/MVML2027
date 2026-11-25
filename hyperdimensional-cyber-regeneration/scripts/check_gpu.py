#!/usr/bin/env python3
"""Check CUDA readiness without allowing a CPU research fallback."""
import json
import platform
import sys

def main():
    manifest = {"python": platform.python_version(), "required_backend": "cuda",
                "cpu_fallback": False, "ready": False}
    try:
        import torch
    except ImportError:
        manifest["reason"] = "PyTorch is not installed. Install a compatible CUDA build on the research GPU host."
        print(json.dumps(manifest, indent=2))
        return 2
    manifest["torch"] = torch.__version__
    manifest["torch_cuda_runtime"] = torch.version.cuda
    if not torch.cuda.is_available():
        manifest["reason"] = "CUDA is unavailable. GPU research runs must stop; CPU substitution is disabled."
        print(json.dumps(manifest, indent=2))
        return 2
    try:
        device = torch.device("cuda:0")
        props = torch.cuda.get_device_properties(device)
        tensor = torch.ones((64, 64), device=device)
        result = tensor @ tensor
        torch.cuda.synchronize(device)
        if result.device.type != "cuda" or not torch.allclose(result, torch.full_like(result, 64.0)):
            raise RuntimeError("CUDA tensor operation verification failed")
        manifest.update({
            "ready": True,
            "device": str(device),
            "gpu": props.name,
            "vram_gib": round(props.total_memory / 2**30, 2),
            "compute_capability": f"{props.major}.{props.minor}",
            "bf16_supported": torch.cuda.is_bf16_supported(),
            "check": "CUDA allocation, matrix multiplication, synchronization and value verification passed",
        })
    except Exception as exc:
        manifest["reason"] = f"CUDA runtime check failed: {type(exc).__name__}: {exc}"
        print(json.dumps(manifest, indent=2))
        return 2
    print(json.dumps(manifest, indent=2))
    return 0

if __name__ == "__main__":
    sys.exit(main())
