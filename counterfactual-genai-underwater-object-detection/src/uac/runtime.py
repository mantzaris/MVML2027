"""Durable project-local run state. No torch import until a checkpoint needs it."""
from __future__ import annotations

import contextlib
import datetime as dt
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import signal
import socket
import time
import uuid
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_VERSION = 1


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def project_path(path: str | Path) -> Path:
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    if not path.is_relative_to(PROJECT_ROOT):
        raise ValueError(f"Project work must stay under {PROJECT_ROOT}: {path}")
    return path


def _fsync_dir(directory: Path) -> None:
    fd = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    path = project_path(path)
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_bytes(path: str | Path, value: bytes) -> None:
    """Commit bytes with a same-directory rename; old value survives a failed write."""
    path = project_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_dir(path.parent)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def atomic_json(path: str | Path, value: Any) -> None:
    atomic_bytes(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n")


def source_fingerprint(paths: list[str | Path]) -> str:
    records = {}
    for item in paths:
        path = project_path(item)
        records[str(path.relative_to(PROJECT_ROOT))] = sha256_file(path)
    return fingerprint(records)


class RunLock:
    """Kernel lock released automatically after exit, SIGKILL, or a power cycle."""
    def __init__(self, path: str | Path = "runs/active.lock"):
        self.path = project_path(path)
        self.handle = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+")
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.handle.close()
            self.handle = None
            raise RuntimeError(f"A research run already holds {self.path}") from None
        self.handle.seek(0)
        self.handle.truncate()
        json.dump({"pid": os.getpid(), "host": socket.gethostname(), "started_utc": utc_now()}, self.handle)
        self.handle.write("\n")
        self.handle.flush()
        os.fsync(self.handle.fileno())
        _fsync_dir(self.path.parent)
        return self

    def __exit__(self, *exc):
        if self.handle is not None:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()
            self.handle = None


class EventLedger:
    """Fsynced JSONL, hash chained. Only an incomplete final record is repairable."""
    def __init__(self, path: str | Path = "runs/events.jsonl"):
        self.path = project_path(path)

    def _scan(self, handle, repair_tail: bool):
        handle.seek(0)
        raw = handle.read()
        records = []
        offset = 0
        previous = None
        for index, line in enumerate(raw.splitlines(keepends=True)):
            start = offset
            offset += len(line)
            is_last = offset == len(raw)
            try:
                if not line.endswith(b"\n"):
                    raise ValueError("Uncommitted final line")
                row = json.loads(line)
                actual_hash = row.pop("record_sha256")
                if row.get("sequence") != index:
                    raise ValueError("Sequence mismatch")
                if row.get("previous_sha256") != previous:
                    raise ValueError("Hash-chain mismatch")
                if fingerprint(row) != actual_hash:
                    raise ValueError("Record checksum mismatch")
                row["record_sha256"] = actual_hash
            except (ValueError, TypeError, KeyError) as exc:
                if not is_last or not repair_tail:
                    raise ValueError(f"Corrupt event ledger at byte {start}: {exc}") from exc
                # Complete newline-terminated bad records signal corruption, not an
                # interrupted append. Never erase them automatically.
                if line.endswith(b"\n"):
                    raise ValueError(f"Corrupt complete ledger record at byte {start}") from exc
                quarantine = self.path.with_name(f"{self.path.name}.partial-{time.time_ns()}")
                atomic_bytes(quarantine, raw[start:])
                handle.seek(start)
                handle.truncate()
                handle.flush()
                os.fsync(handle.fileno())
                _fsync_dir(self.path.parent)
                break
            records.append(row)
            previous = row["record_sha256"]
        return records

    def read(self, repair_tail: bool = False) -> list[dict]:
        if not self.path.exists():
            return []
        mode = "r+b" if repair_tail else "rb"
        with self.path.open(mode) as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX if repair_tail else fcntl.LOCK_SH)
            return self._scan(handle, repair_tail)

    def append(self, event: str, **fields) -> dict:
        reserved = {"event", "sequence", "timestamp_utc", "previous_sha256", "record_sha256"}
        if reserved.intersection(fields):
            raise ValueError("Ledger fields contain reserved keys")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            rows = self._scan(handle, repair_tail=True)
            row = {
                "sequence": len(rows), "timestamp_utc": utc_now(), "event": event,
                "previous_sha256": rows[-1]["record_sha256"] if rows else None,
                **fields,
            }
            row["record_sha256"] = fingerprint(row)
            payload = canonical_json(row) + b"\n"
            handle.seek(0, os.SEEK_END)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_dir(self.path.parent)
        return row


def capture_rng(torch_module=None) -> dict:
    result = {"python": random.getstate()}
    try:
        np = importlib.import_module("numpy")
        result["numpy"] = np.random.get_state()
    except ModuleNotFoundError:
        pass
    if torch_module is not None:
        result["torch_cpu"] = torch_module.get_rng_state()
        if torch_module.cuda.is_available():
            result["torch_cuda"] = torch_module.cuda.get_rng_state_all()
    return result


def restore_rng(state: dict, torch_module=None) -> None:
    random.setstate(state["python"])
    if "numpy" in state:
        importlib.import_module("numpy").random.set_state(state["numpy"])
    if torch_module is not None and "torch_cpu" in state:
        torch_module.set_rng_state(state["torch_cpu"])
    if torch_module is not None and "torch_cuda" in state:
        if not torch_module.cuda.is_available():
            raise RuntimeError("Cannot restore saved CUDA RNG without CUDA; no CPU fallback")
        if len(state["torch_cuda"]) != torch_module.cuda.device_count():
            raise RuntimeError("CUDA device count differs from checkpoint")
        torch_module.cuda.set_rng_state_all(state["torch_cuda"])


class CheckpointStore:
    """Immutable generations with checksums and an atomic latest/previous index.

    Callers must hold RunLock while saving. Only self-created, trusted checkpoints
    may be loaded: torch checkpoints use pickle and are not an untrusted format.
    """
    def __init__(self, directory: str | Path, *, config: Any, data_fingerprint: str,
                 code_fingerprint: str | None = None, torch_module=None):
        self.directory = project_path(directory)
        self.expected = {
            "checkpoint_version": CHECKPOINT_VERSION,
            "config_sha256": fingerprint(config),
            "data_fingerprint": data_fingerprint,
            "code_fingerprint": code_fingerprint,
        }
        self.config = config
        self._torch = torch_module
        self.last_recovery = None

    @property
    def torch(self):
        if self._torch is None:
            self._torch = importlib.import_module("torch")
        return self._torch

    def _metadata_candidates(self) -> list[dict]:
        if not self.directory.exists():
            return []
        result = []
        # Sidecars are committed only after the payload has reached durable storage.
        # This also recovers when latest.json is lost or corrupted.
        for path in self.directory.glob("generation-*.json"):
            try:
                meta = json.loads(path.read_text())
                checksum = meta.pop("metadata_sha256")
                if fingerprint(meta) != checksum:
                    continue
                meta["metadata_sha256"] = checksum
                if all(meta.get(k) == v for k, v in self.expected.items()):
                    result.append(meta)
            except (OSError, ValueError, KeyError, TypeError):
                continue
        return sorted(result, key=lambda m: m["created_ns"], reverse=True)

    def save_state(self, state: dict, *, global_step: int, cursor: dict | None = None) -> dict:
        self.directory.mkdir(parents=True, exist_ok=True)
        generation = f"generation-{time.time_ns()}-{uuid.uuid4().hex[:8]}"
        target = self.directory / f"{generation}.pt"
        temporary = target.with_name(f".{target.name}.tmp")
        payload = {
            **self.expected, "config": self.config, "global_step": int(global_step),
            "cursor": cursor or {}, "state": state,
        }
        try:
            with temporary.open("xb") as handle:
                self.torch.save(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
            _fsync_dir(self.directory)
        finally:
            with contextlib.suppress(FileNotFoundError):
                temporary.unlink()
        meta = {
            **self.expected, "generation": generation, "created_ns": time.time_ns(),
            "created_utc": utc_now(), "payload": target.name,
            "payload_sha256": sha256_file(target), "payload_bytes": target.stat().st_size,
            "global_step": int(global_step), "cursor": cursor or {},
        }
        meta["metadata_sha256"] = fingerprint(meta)
        atomic_json(self.directory / f"{generation}.json", meta)
        candidates = self._metadata_candidates()
        # Never discard old generations before committing the index. Keep two
        # physically valid generations (and any differently configured run).
        good = []
        for candidate in candidates:
            p = self.directory / candidate["payload"]
            if p.is_file() and sha256_file(p) == candidate["payload_sha256"]:
                good.append(candidate)
            if len(good) == 2:
                break
        atomic_json(self.directory / "latest.json", {
            "checkpoint_version": CHECKPOINT_VERSION,
            "latest": good[0]["generation"] if good else None,
            "previous": good[1]["generation"] if len(good) > 1 else None,
            "entries": good,
        })
        keep = {m["generation"] for m in good}
        for candidate in candidates:
            if candidate["generation"] not in keep:
                # Both unlink steps are safe only after newer generations + index
                # have committed. A torn cleanup leaves an ignorable orphan.
                for suffix in (".json", ".pt"):
                    with contextlib.suppress(FileNotFoundError):
                        (self.directory / f"{candidate['generation']}{suffix}").unlink()
        _fsync_dir(self.directory)
        return meta

    def load_state(self, *, map_location="cpu") -> dict | None:
        failures = []
        candidates = self._metadata_candidates()
        if not candidates:
            if self.directory.exists() and any(self.directory.glob("generation-*.json")):
                raise RuntimeError("No checkpoint matches version/config/data/code; use a new run directory")
            self.last_recovery = {"loaded": None, "rejected": []}
            return None
        for meta in candidates:
            path = self.directory / meta["payload"]
            try:
                if not path.is_file() or path.stat().st_size != meta["payload_bytes"]:
                    raise ValueError("Missing or truncated payload")
                if sha256_file(path) != meta["payload_sha256"]:
                    raise ValueError("Payload checksum mismatch")
                payload = self.torch.load(path, map_location=map_location, weights_only=False)
                if not all(payload.get(k) == v for k, v in self.expected.items()):
                    raise ValueError("Payload compatibility mismatch")
                if payload["global_step"] != meta["global_step"] or payload["cursor"] != meta["cursor"]:
                    raise ValueError("Payload cursor mismatch")
                self.last_recovery = {"loaded": meta, "rejected": failures}
                return payload
            except (OSError, ValueError, KeyError, RuntimeError, EOFError) as exc:
                failures.append({"generation": meta["generation"], "reason": str(exc)})
        self.last_recovery = {"loaded": None, "rejected": failures}
        raise RuntimeError(f"All compatible checkpoint generations failed integrity checks: {failures}")

    def save_training(self, model, optimizer=None, *, scheduler=None, scaler=None,
                      epoch: int, batch_cursor: int, global_step: int,
                      extra: dict | None = None) -> dict:
        if self.torch.cuda.is_available():
            self.torch.cuda.synchronize()
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "rng": capture_rng(self.torch),
            "extra": extra or {},
        }
        return self.save_state(state, global_step=global_step,
                               cursor={"epoch": int(epoch), "batch_cursor": int(batch_cursor)})

    def load_training(self, model, optimizer=None, *, scheduler=None, scaler=None,
                      restore_rng_state: bool = True) -> dict | None:
        payload = self.load_state(map_location="cpu")
        if payload is None:
            return None
        state = payload["state"]
        model.load_state_dict(state["model"])
        for key, instance in (("optimizer", optimizer), ("scheduler", scheduler), ("scaler", scaler)):
            if instance is not None:
                if state[key] is None:
                    raise RuntimeError(f"Checkpoint lacks required {key} state")
                instance.load_state_dict(state[key])
        if restore_rng_state:
            restore_rng(state["rng"], self.torch)
        return {"epoch": payload["cursor"]["epoch"],
                "batch_cursor": payload["cursor"]["batch_cursor"],
                "global_step": payload["global_step"], "extra": state["extra"],
                "recovery": self.last_recovery}


def completed_stage(path: str | Path, *, stage: str, config_hash: str,
                    artifacts: list[str | Path], data_fingerprint: str | None = None,
                    details: dict | None = None) -> dict:
    entries = {}
    for item in artifacts:
        file_path = project_path(item)
        entries[str(file_path.relative_to(PROJECT_ROOT))] = {
            "sha256": sha256_file(file_path), "bytes": file_path.stat().st_size,
        }
    record = {"version": 1, "stage": stage, "status": "complete",
              "completed_utc": utc_now(), "config_hash": config_hash,
              "data_fingerprint": data_fingerprint, "artifacts": entries,
              "details": details or {}}
    atomic_json(path, record)
    return record


def stage_is_complete(path: str | Path, *, config_hash: str,
                      data_fingerprint: str | None = None) -> bool:
    path = project_path(path)
    if not path.exists():
        return False
    try:
        record = json.loads(path.read_text())
        if (record["version"] != 1 or record["status"] != "complete"
                or record["config_hash"] != config_hash
                or record.get("data_fingerprint") != data_fingerprint):
            return False
        for name, entry in record["artifacts"].items():
            item = project_path(name)
            if not item.is_file() or item.stat().st_size != entry["bytes"]:
                return False
            if sha256_file(item) != entry["sha256"]:
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


class StopRequested:
    """Set a flag on SIGINT/SIGTERM; caller saves safely at an optimizer boundary."""
    def __init__(self):
        self.requested = False
        self.signal_number = None
        self.previous = {}

    def _handler(self, signum, frame):
        self.requested = True
        self.signal_number = signum

    def __enter__(self):
        for signum in (signal.SIGINT, signal.SIGTERM):
            self.previous[signum] = signal.signal(signum, self._handler)
        return self

    def __exit__(self, *exc):
        for signum, handler in self.previous.items():
            signal.signal(signum, handler)
