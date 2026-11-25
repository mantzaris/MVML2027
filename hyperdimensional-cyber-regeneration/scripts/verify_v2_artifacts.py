#!/usr/bin/env python3
"""Verify the second study's compact artifacts and preserved first-study code."""
import csv
import hashlib
import json
import math
from pathlib import Path

from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1]
MAIN=['v2_fig01_generalization','v2_fig02_joint_calibration','v2_fig03_matched_controls','v2_fig04_evidence_compute']
EXTRA=['v2_figS1_training','v2_figS2_seed_effects']


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def check_inputs(metadata):
    for relative,digest in metadata['inputs'].items():assert sha(ROOT/relative)==digest,relative


def main():
    data=ROOT/'results/v2';manifest=json.loads((data/'manifest.json').read_text())
    assert manifest['settings']['test_seed']==42027 and not manifest['settings']['pilot']
    assert manifest['settings']['seeds']==[11,23,37,51,71]
    for relative,digest in manifest['source_hashes'].items():assert sha(ROOT/relative)==digest,relative
    old=json.loads((ROOT/'results/study_manifest.json').read_text())
    for relative,digest in old['source_hashes'].items():assert sha(ROOT/relative)==digest,relative
    rows=list(csv.DictReader((data/'summary.csv').open()))
    assert len(rows)==1530
    keys=['seed','method','condition','corruption','attack','facts','budget','repair_fraction','missing']
    assert len({tuple(r[k] for k in keys) for r in rows})==1530
    for r in rows:
        for key in ['regret','recall3','brier','plan_brier','logical_violation','latency_ms']:
            assert math.isfinite(float(r[key])),(key,r)
        if r['method'] in ['JH','JD','JH_ERM','JH_product']:assert float(r['logical_violation'])==0
        assert float(r['bytes'])<=2*int(r['budget'])
    comparison=json.loads((data/'comparisons.json').read_text())
    assert sha(data/'manifest.json')==comparison['manifest_sha256']
    assert sha(data/'summary.csv')==comparison['summary_sha256']
    assert sha(ROOT/'scripts/analyze_v2.py')==comparison['analysis_source_sha256']
    for directory,stems in [(ROOT/'figures/v2',MAIN),(ROOT/'supplemental/v2',EXTRA)]:
        for stem in stems:
            p=json.loads((directory/f'{stem}.provenance.json').read_text());check_inputs(p)
            assert sha(ROOT/p['script'])==p['script_sha256']
            for ext,digest in p['output_sha256'].items():assert sha(directory/f'{stem}.{ext}')==digest
            assert len(PdfReader(directory/f'{stem}.pdf').pages)==1
    directory=ROOT/'supplemental/v2';pdf=directory/'Supplement_v2.pdf'
    pages=PdfReader(pdf).pages;assert len(pages)==3
    assert all(len(p.extract_text())>350 for p in pages)
    p=json.loads((directory/'Supplement_v2.provenance.json').read_text());check_inputs(p)
    assert sha(pdf)==p['pdf_sha256'];assert sha(ROOT/'scripts/make_v2_supplement.py')==p['script_sha256']
    replay=json.loads((data/'replay.json').read_text())
    assert replay['passed'] and replay['model_snapshot_evaluations']==32000
    assert all(value<=1e-5 for value in replay['maximum_absolute_errors'].values())
    assert sha(ROOT/'scripts/replay_v2.py')==replay['replay_script_sha256']
    assert sha(data/'manifest.json')==replay['source_manifest_sha256']
    timing=json.loads((data/'pipeline_timing.json').read_text())
    assert len(timing['rows'])==8 and all(r['repeats']==50 for r in timing['rows'])
    assert sha(data/'manifest.json')==timing['manifest_sha256']
    for relative,digest in timing['source_hashes'].items():assert sha(ROOT/relative)==digest
    print(json.dumps(dict(verified=True,rows=len(rows),main_figures=4,extra_figures=2,supplement_pages=3,
                         source_hashes_unchanged=True,first_study_source_preserved=True)))


if __name__=='__main__':main()
