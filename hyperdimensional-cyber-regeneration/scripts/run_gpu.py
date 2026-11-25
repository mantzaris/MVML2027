#!/usr/bin/env python3
"""Run project Python on exactly one matching NVIDIA GPU."""
import argparse
import csv
from dataclasses import dataclass
import io
import os
from pathlib import Path
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_GPU = "NVIDIA RTX 5000 Ada Generation"

@dataclass(frozen=True)
class GPU:
    index: str
    uuid: str
    name: str
    driver: str

def query_gpus():
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,name,driver_version",
             "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=15)
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(
            "NVIDIA devices are unavailable in this shell. Use a host shell with GPU "
            "access; the restricted execution sandbox may hide /dev/nvidia*.") from exc
    devices = []
    for row in csv.reader(io.StringIO(proc.stdout), skipinitialspace=True):
        if row:
            if len(row) != 4:
                raise RuntimeError("Unexpected NVIDIA inventory format")
            devices.append(GPU(*(field.strip() for field in row)))
    return devices

def select_gpu(devices, expected_name=DEFAULT_GPU, requested_uuid=None):
    matches = [d for d in devices if d.name.casefold() == expected_name.casefold()]
    if requested_uuid:
        matches = [d for d in matches if d.uuid == requested_uuid]
    if not matches:
        raise RuntimeError(f"No matching GPU: {expected_name}. CPU fallback is disabled.")
    if len(matches) != 1:
        raise RuntimeError("Multiple matching GPUs. Select one with --gpu-uuid.")
    return matches[0]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-name", default=DEFAULT_GPU)
    parser.add_argument("--gpu-uuid", default=os.environ.get("REGEN_GPU_UUID"))
    parser.add_argument("python_args", nargs=argparse.REMAINDER,
                        help="Arguments after -- are passed to the project's Python.")
    args = parser.parse_args()
    python = PROJECT / ".venv" / "bin" / "python"
    try:
        if not python.is_file():
            raise RuntimeError("Project .venv is missing. Follow docs/GPU_SETUP.md.")
        gpu = select_gpu(query_gpus(), args.gpu_name, args.gpu_uuid)
        env = os.environ.copy()
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = gpu.uuid
        env["REGEN_EXPECTED_GPU_NAME"] = gpu.name
        env["REGEN_DRIVER_VERSION"] = gpu.driver
        env["REGEN_GPU_PHYSICAL_INDEX"] = gpu.index
        for name, folder in [
            ("XDG_CACHE_HOME", "cache"), ("TORCH_HOME", "torch"),
            ("MPLCONFIGDIR", "matplotlib"), ("TMPDIR", "tmp"),
            ("CUDA_CACHE_PATH", "cuda-cache"), ("TRITON_CACHE_DIR", "triton"),
            ("PIP_CACHE_DIR", "pip")]:
            directory = PROJECT / "work" / folder
            directory.mkdir(parents=True, exist_ok=True)
            env[name] = str(directory)
        env.pop("PYTHONPATH", None)
        env.pop("PYTHONHOME", None)
        env["PYTHONNOUSERSITE"] = "1"
        command = args.python_args
        if command and command[0] == "--":
            command = command[1:]
        if not command:
            command = ["scripts/check_gpu.py"]
        print(f"Using {gpu.name}: physical GPU {gpu.index} -> cuda:0", file=sys.stderr, flush=True)
        os.chdir(PROJECT)
        os.execve(str(python), [str(python), *command], env)
    except (RuntimeError, OSError) as exc:
        print(f"GPU launch failed: {exc}", file=sys.stderr)
        return 2

if __name__ == "__main__":
    sys.exit(main())
