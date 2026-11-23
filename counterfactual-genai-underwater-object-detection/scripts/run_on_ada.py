#!/usr/bin/env python3
"""Verify the intended Ada GPU and launch argv with only that UUID visible.

Examples:
  python3 scripts/run_on_ada.py --dry-run
  python3 scripts/run_on_ada.py -- /path/to/python scripts/verify_cuda.py

This launcher does not import torch, install software, create remote connections,
or choose a substitute GPU. Child research programs must require CUDA explicitly.
"""

import argparse
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "gpu.json"


def validate_gpu_inventory(inventory, config):
    """Return the exact verified device; reject malformed or mismatched records."""
    records = []
    for row in csv.reader(io.StringIO(inventory)):
        if not row or all(not value.strip() for value in row):
            continue
        if len(row) != 3:
            raise ValueError("Unexpected nvidia-smi inventory format")
        uuid, name, memory = (value.strip() for value in row)
        records.append({"uuid": uuid, "name": name, "memory_mib": int(memory)})
    matches = [record for record in records if record["uuid"] == config["gpu_uuid"]]
    if len(matches) != 1:
        raise ValueError("Required GPU UUID is absent or appears more than once")
    device = matches[0]
    if device["name"] != config["gpu_name"]:
        raise ValueError("GPU name does not match the configured Ada device")
    if device["memory_mib"] < int(config["minimum_memory_mib"]):
        raise ValueError("Configured GPU has insufficient total VRAM")
    return device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not args.dry_run and not command:
        parser.error("Supply a command after --, or use --dry-run")
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid,name,memory.total", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=15,
        )
        device = validate_gpu_inventory(result.stdout, config)
        env = os.environ.copy()
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = device["uuid"]
        # This variable records intent; child applications still must assert CUDA.
        env["MVML_REQUIRE_CUDA"] = "1"
        print(json.dumps({"status": "gpu_identity_verified", "device": device,
                          "cuda_visible_devices": device["uuid"],
                          "dry_run": args.dry_run}), flush=True)
        if args.dry_run:
            return 0
        return subprocess.run(command, env=env, check=False).returncode
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
