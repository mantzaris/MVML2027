"""Fail-closed test freeze integrity; every test stage calls before inference."""
import json
from pathlib import Path
from .runtime import sha256_file


def require_frozen_test():
    path=Path('runs/test_freeze.json')
    if not path.exists():raise RuntimeError('Final test remains locked; no protocol freeze')
    freeze=json.loads(path.read_text())
    if freeze.get('status')!='frozen' or not freeze.get('artifacts'):raise RuntimeError('Invalid or empty test freeze')
    if sha256_file('data/manifests/planned_real.jsonl')!=freeze['data_fingerprint']:raise RuntimeError('Frozen cohort changed')
    for relative,record in freeze['artifacts'].items():
        if sha256_file(relative)!=record['sha256']:raise RuntimeError('Frozen artifact changed: '+relative)
    return freeze
