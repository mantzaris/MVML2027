"""Meaningful metadata-only restart guards; no tensor computation or GPU use."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_research as runner
import stage_provenance as provenance
from synthetic_common import DEFAULTS

class StageRestartTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=ROOT / 'tmp', prefix='stage-guard-fixture-')
        self.root = Path(self.directory.name)
        (self.root / 'scripts').mkdir()
        (self.root / 'scripts/run_research.py').write_text('supervisor fixture')
        (self.root / 'output.json').write_text('{"complete":true}\n')
        self.commands = [['python', 'task.py']]
        self.inputs = {'task.py':'source-v1', 'config.json':'configuration-v1'}
        self.patches = [patch.object(runner, 'ROOT', self.root),
            patch.object(runner, 'ledger', MagicMock()),
            patch.object(runner, 'stage_inputs', side_effect=lambda *_: dict(self.inputs)),
            patch.object(runner, 'native_adoption_evidence', return_value={'basis':'fixture native proof'}),
            patch.object(runner, 'sha256_file', side_effect=lambda p: hashlib.sha256((self.root / p).read_bytes()).hexdigest())]
        for p in self.patches: p.start()
    def tearDown(self):
        for p in reversed(self.patches): p.stop()
        self.directory.cleanup()
    def test_unchanged_output_cannot_mask_changed_source_or_config(self):
        runner.stage('fixture', self.commands, ['output.json'], adopt=True)
        runner.stage('fixture', self.commands, ['output.json'], adopt=True)
        for name in ('config.json', 'task.py'):
            before = self.inputs[name]
            self.inputs[name] = 'changed'
            with self.assertRaisesRegex(RuntimeError, 'inputs changed'):
                runner.stage('fixture', self.commands, ['output.json'], adopt=True)
            self.inputs[name] = before
    def test_legacy_migration_preserves_exact_original_bytes(self):
        folder = self.root / 'runs/stages'; folder.mkdir(parents=True)
        old = {'stage':'fixture', 'commands':self.commands, 'command_hash':runner.fingerprint(self.commands),
               'outputs':{'output.json':runner.sha256_file('output.json')}, 'completed_utc':'historical-time'}
        original = json.dumps(old, separators=(',', ':')).encode() + b'\n'
        (folder / 'fixture.json').write_bytes(original)
        runner.stage('fixture', self.commands, ['output.json'], adopt=True)
        migrated = json.loads((folder / 'fixture.json').read_text())
        self.assertEqual(migrated['schema_version'], 2)
        self.assertEqual(migrated['completed_utc'], 'historical-time')
        self.assertEqual((self.root / migrated['migration']['original_receipt']).read_bytes(), original)
        self.assertEqual(migrated['migration']['original_receipt_sha256'], hashlib.sha256(original).hexdigest())
        self.assertEqual(migrated['migration']['evidence']['basis'], 'fixture native proof')
    def test_changed_completed_output_is_refused(self):
        runner.stage('fixture', self.commands, ['output.json'], adopt=True)
        (self.root / 'output.json').write_text('tampered')
        with self.assertRaisesRegex(RuntimeError, 'output changed'):
            runner.stage('fixture', self.commands, ['output.json'], adopt=True)

class NativeCompletionTests(unittest.TestCase):
    def test_best_epoch_without_full_training_history_is_not_adoptable(self):
        with patch.object(provenance, 'read', return_value=copy.deepcopy(DEFAULTS)), \
             patch.object(provenance, 'verify_native_receipt', return_value={'details':{'history':[{'epoch':1}]}}):
            with self.assertRaisesRegex(RuntimeError, 'epoch history incomplete'):
                provenance.native_adoption_evidence('synthetic_training')

if __name__ == '__main__':
    unittest.main()
