"""Meaningful interruption/recovery tests; all temporary files stay in project."""
import importlib.util
import json
import os
from pathlib import Path
import pickle
import random
import signal
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from uac.runtime import (
    CheckpointStore, EventLedger, RunLock, StopRequested, atomic_json,
    capture_rng, completed_stage, fingerprint, restore_rng, stage_is_complete,
)


class FakeTorch:
    """Serializer adapter: checkpoints tested without loading a compute runtime."""
    @staticmethod
    def save(value, handle):
        pickle.dump(value, handle)

    @staticmethod
    def load(path, **kwargs):
        with Path(path).open("rb") as handle:
            return pickle.load(handle)


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        local = ROOT / ".local" / "runtime_tests"
        local.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=local)
        self.directory = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def store(self, config=None):
        return CheckpointStore(self.directory / "checkpoints",
                               config=config or {"seed": 17, "stage": "test"},
                               data_fingerprint="data-sha", code_fingerprint="code-sha",
                               torch_module=FakeTorch())

    def test_atomic_json_preserves_previous_when_rename_fails(self):
        path = self.directory / "state.json"
        atomic_json(path, {"step": 3})
        with mock.patch("uac.runtime.os.replace", side_effect=OSError("interrupted")):
            with self.assertRaises(OSError):
                atomic_json(path, {"step": 4})
        self.assertEqual(json.loads(path.read_text()), {"step": 3})
        self.assertFalse(list(path.parent.glob(".*.tmp")))

    def test_new_checkpoint_before_index_commit_is_recoverable(self):
        store = self.store()
        store.save_state({"value": 1}, global_step=1)
        import uac.runtime as runtime
        real_atomic = runtime.atomic_json
        def fail_index(path, value):
            if Path(path).name == "latest.json":
                raise OSError("power cut before latest pointer")
            return real_atomic(path, value)
        with mock.patch("uac.runtime.atomic_json", side_effect=fail_index):
            with self.assertRaises(OSError):
                store.save_state({"value": 2}, global_step=2)
        recovered = self.store().load_state()
        self.assertEqual(recovered["global_step"], 2)
        self.assertEqual(recovered["state"]["value"], 2)

    def test_corrupt_latest_falls_back_to_previous(self):
        store = self.store()
        first = store.save_state({"value": 1}, global_step=1)
        last = store.save_state({"value": 2}, global_step=2)
        (store.directory / last["payload"]).write_bytes(b"torn checkpoint")
        recovered = store.load_state()
        self.assertEqual(recovered["global_step"], 1)
        self.assertEqual(len(store.last_recovery["rejected"]), 1)

    def test_uncommitted_temporary_checkpoint_is_ignored(self):
        store = self.store()
        store.save_state({"value": 1}, global_step=1)
        (store.directory / ".generation-incomplete.pt.tmp").write_bytes(b"half written")
        self.assertEqual(store.load_state()["global_step"], 1)

    def test_only_last_two_generations_kept(self):
        store = self.store()
        for step in range(4):
            store.save_state({"value": step}, global_step=step)
        self.assertEqual(len(list(store.directory.glob("generation-*.pt"))), 2)
        self.assertEqual(len(list(store.directory.glob("generation-*.json"))), 2)
        index = json.loads((store.directory / "latest.json").read_text())
        self.assertEqual([entry["global_step"] for entry in index["entries"]], [3, 2])

    def test_config_mismatch_does_not_silently_resume(self):
        self.store().save_state({"value": 1}, global_step=1)
        with self.assertRaisesRegex(RuntimeError, "No checkpoint matches"):
            self.store(config={"seed": 99}).load_state()

    def test_corrupt_index_recovers_from_generation_metadata(self):
        store = self.store()
        store.save_state({"value": 1}, global_step=1)
        (store.directory / "latest.json").write_text('{"latest":')
        self.assertEqual(store.load_state()["global_step"], 1)

    def test_partial_ledger_tail_quarantined_then_resumed(self):
        ledger = EventLedger(self.directory / "events.jsonl")
        ledger.append("start", stage="train")
        with ledger.path.open("ab") as handle:
            handle.write(b'{"sequence":1,"interrupted":')
        ledger.append("resumed", stage="train")
        records = ledger.read()
        self.assertEqual([r["event"] for r in records], ["start", "resumed"])
        self.assertEqual([r["sequence"] for r in records], [0, 1])
        quarantines = list(self.directory.glob("events.jsonl.partial-*"))
        self.assertEqual(len(quarantines), 1)
        self.assertEqual(quarantines[0].read_bytes(), b'{"sequence":1,"interrupted":')

    def test_complete_or_interior_ledger_corruption_not_erased(self):
        ledger = EventLedger(self.directory / "events.jsonl")
        ledger.append("start")
        with ledger.path.open("ab") as handle:
            handle.write(b'{"bad":true}\n')
        original = ledger.path.read_bytes()
        with self.assertRaisesRegex(ValueError, "Corrupt complete"):
            ledger.append("later")
        self.assertEqual(ledger.path.read_bytes(), original)

    def test_stage_completion_rejects_changed_artifact(self):
        artifact = self.directory / "predictions.json"
        atomic_json(artifact, {"prediction": 0.2})
        marker = self.directory / "complete.json"
        completed_stage(marker, stage="eval", config_hash="cfg",
                        artifacts=[artifact], data_fingerprint="data")
        self.assertTrue(stage_is_complete(marker, config_hash="cfg", data_fingerprint="data"))
        self.assertFalse(stage_is_complete(marker, config_hash="other", data_fingerprint="data"))
        atomic_json(artifact, {"prediction": 0.3})
        self.assertFalse(stage_is_complete(marker, config_hash="cfg", data_fingerprint="data"))

    def test_lock_prevents_second_writer_and_releases(self):
        path = self.directory / "run.lock"
        with RunLock(path):
            with self.assertRaises(RuntimeError):
                with RunLock(path):
                    pass
        with RunLock(path):
            pass

    def test_python_rng_state_reproduces_sequence(self):
        random.seed(17)
        state = capture_rng()
        expected = [random.random() for _ in range(5)]
        restore_rng(state)
        self.assertEqual(expected, [random.random() for _ in range(5)])

    def test_signal_sets_flag_without_writing_checkpoint_in_handler(self):
        previous = signal.getsignal(signal.SIGTERM)
        with StopRequested() as stop:
            os.kill(os.getpid(), signal.SIGTERM)
            self.assertTrue(stop.requested)
            self.assertEqual(stop.signal_number, signal.SIGTERM)
        self.assertEqual(signal.getsignal(signal.SIGTERM), previous)

    def test_paths_outside_project_rejected(self):
        with self.assertRaises(ValueError):
            atomic_json("/tmp/mvml-should-not-be-created.json", {"bad": True})


if __name__ == "__main__":
    unittest.main()
