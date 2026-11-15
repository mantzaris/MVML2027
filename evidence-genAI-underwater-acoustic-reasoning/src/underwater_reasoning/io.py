"""Content hashes and durable, non-overwriting research records."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".writing-")
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def safe_relative(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Audio paths must be relative and contained in the data root")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("Audio path escapes the data root")
    return resolved


class ResultStore:
    """One immutable atomic JSON shard per full content key; single writer only.

    Failed/incomplete shards never masquerade as successes. A retry can replace a
    failed shard; successful shards are reused and cannot be silently overwritten.
    """
    def __init__(self, root: Path):
        self.root = root

    def read(self, key: dict) -> dict | None:
        path = self.root / (canonical_hash(key) + ".json")
        if not path.exists():
            return None
        record = json.loads(path.read_text())
        if record["key"] != key:
            raise ValueError("Cache key collision or corrupt shard")
        return record if record["status"] == "success" else None

    def write(self, key: dict, payload: dict, status: str = "success") -> None:
        if status not in {"success", "failed"}:
            raise ValueError("Invalid shard status")
        previous = self.read(key)
        if previous is not None:
            if previous["payload"] != payload or status != "success":
                raise ValueError("Refusing to overwrite a completed result")
            return
        atomic_json(self.root / (canonical_hash(key) + ".json"),
                    {"key": key, "status": status, "payload": payload})
