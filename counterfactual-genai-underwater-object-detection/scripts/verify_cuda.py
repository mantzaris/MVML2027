#!/usr/bin/env python3
"""Run tiny CUDA-only tensor checks after run_on_ada.py has pinned the device.

No dependency installation or CPU fallback is performed. This verifies CUDA
execution only; it is not a detector benchmark or full pipeline validation.
"""

import argparse
import json
import os
from pathlib import Path
import sys


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "gpu.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        if os.environ.get("CUDA_VISIBLE_DEVICES") != config["gpu_uuid"]:
            raise RuntimeError("Run through run_on_ada.py to pin the exact required UUID")
        import torch

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Exactly one usable CUDA device must be visible")
        props = torch.cuda.get_device_properties(0)
        if props.name != config["gpu_name"]:
            raise RuntimeError("PyTorch device name does not match the required Ada GPU")
        if props.total_memory < int(config["minimum_memory_mib"]) * 1024**2:
            raise RuntimeError("PyTorch reports insufficient GPU memory")
        torch.cuda.reset_peak_memory_stats(0)
        with torch.inference_mode():
            matrix = torch.ones((128, 128), device="cuda:0", dtype=torch.float32)
            product = matrix @ matrix
            signal = torch.arange(4096, device="cuda:0", dtype=torch.float32)
            signal = torch.sin(signal * (2 * torch.pi * 37 / 4096))
            spectrum = torch.fft.rfft(signal)
            window = torch.hann_window(256, device="cuda:0")
            spectrogram = torch.stft(signal, n_fft=256, hop_length=64,
                                     window=window, return_complex=True)
            values = (product, spectrum, spectrogram)
            if any(value.device.type != "cuda" for value in values):
                raise RuntimeError("A tensor operation left CUDA")
            if any(not torch.isfinite(value).all().item() for value in values):
                raise RuntimeError("A CUDA operation returned nonfinite values")
            if not torch.allclose(product, torch.full_like(product, 128)):
                raise RuntimeError("CUDA matrix multiplication failed its value check")
            if int(spectrum.abs().argmax().item()) != 37:
                raise RuntimeError("CUDA FFT failed the known-frequency check")
            torch.cuda.synchronize(0)
        report = {
            "status": "passed", "gpu_uuid": config["gpu_uuid"],
            "gpu_name": props.name, "total_memory_mib": props.total_memory // 1024**2,
            "visible_cuda_devices": torch.cuda.device_count(), "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "checks": ["float32_matmul", "known_frequency_rfft", "complex_stft", "finite_cuda_outputs"],
            "peak_allocated_mib": torch.cuda.max_memory_allocated(0) / 1024**2,
            "limitation": "Smoke check only; no training, HHT validation, or performance benchmark.",
        }
        print(json.dumps(report, indent=2))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "error", "error": str(exc), "cpu_fallback": False}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
