#!/usr/bin/env python3
"""Check third-study artifacts against recorded measurements and preserved studies."""
import csv
import hashlib
import json
import math
from pathlib import Path
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1]
MAIN=['v3_fig01_decision_quality','v3_fig02_probability_semantics','v3_fig03_model_cost','v3_fig04_transfer']
EXTRA=['v3_figS1_training','v3_figS2_seed_effects']
DIRECT={'SCR','CAND_CTX','CAND_LOCAL'}
FACT=['brier','false_repair','false_certainty','coverage','preservation','logical_violation','changed_fraction']


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def inputs(metadata):
    for relative,digest in metadata['inputs'].items():assert sha(ROOT/relative)==digest,relative


def main():
    data=ROOT/'results/v3';manifest=json.loads((data/'manifest.json').read_text())
    assert manifest['settings']['test_seed']==62027 and not manifest['settings']['pilot']
    assert manifest['settings']['seeds']==[11,23,37,51,71]
    for relative,digest in manifest['source_hashes'].items():assert sha(ROOT/relative)==digest,relative
    for path in [ROOT/'results/study_manifest.json',ROOT/'results/v2/manifest.json']:
        previous=json.loads(path.read_text())
        for relative,digest in previous['source_hashes'].items():assert sha(ROOT/relative)==digest,relative
    rows=list(csv.DictReader((data/'summary.csv').open()))
    assert len(rows)==1980 and len({r['method'] for r in rows})==22
    keys=['seed','method','condition','corruption','attack','facts','budget','repair_fraction','missing']
    assert len({tuple(r[k] for k in keys) for r in rows})==1980
    for row in rows:
        for key in ['regret','recall3','plan_brier','latency_ms','latency_p95_ms','peak_vram_mb']:
            assert math.isfinite(float(row[key])) and float(row[key])>=0,(key,row)
        for key in FACT:
            assert math.isnan(float(row[key])) if row['method'] in DIRECT else math.isfinite(float(row[key])),(key,row)
        assert 0<=float(row['bytes'])<=2*int(row['budget'])
        if row['method']=='SCR':assert 0<=float(row['gate_entropy'])<=math.log(4)+1e-6
    comparison=json.loads((data/'comparisons.json').read_text())
    assert sha(data/'manifest.json')==comparison['manifest_sha256']
    assert sha(data/'summary.csv')==comparison['summary_sha256']
    assert sha(ROOT/'scripts/analyze_v3.py')==comparison['analysis_source_sha256']
    assert comparison['primary']['method']=='SCR' and comparison['primary']['comparator']=='ID'
    for directory,stems in [(ROOT/'figures/v3',MAIN),(ROOT/'supplemental/v3',EXTRA)]:
        for stem in stems:
            record=json.loads((directory/f'{stem}.provenance.json').read_text());inputs(record)
            assert sha(ROOT/record['script'])==record['script_sha256']
            for ext,digest in record['output_sha256'].items():assert sha(directory/f'{stem}.{ext}')==digest
            assert len(PdfReader(directory/f'{stem}.pdf').pages)==1
    directory=ROOT/'supplemental/v3';pdf=directory/'Supplement_v3.pdf'
    pages=PdfReader(pdf).pages;assert len(pages)==3
    assert all(len(page.extract_text())>350 for page in pages)
    record=json.loads((directory/'Supplement_v3.provenance.json').read_text());inputs(record)
    assert sha(pdf)==record['pdf_sha256'];assert sha(ROOT/'scripts/make_v3_supplement.py')==record['script_sha256']
    replay=json.loads((data/'replay.json').read_text())
    assert replay['passed'] and replay['model_snapshot_evaluations']==48000
    assert all(error<=1e-5 for error in replay['maximum_absolute_errors'].values())
    assert sha(ROOT/'scripts/replay_v3.py')==replay['replay_script_sha256']
    assert sha(data/'manifest.json')==replay['source_manifest_sha256']
    timing=json.loads((data/'pipeline_timing.json').read_text())
    assert len(timing['rows'])==7 and all(row['repeats']==50 for row in timing['rows'])
    assert sha(data/'manifest.json')==timing['manifest_sha256']
    for relative,digest in timing['source_hashes'].items():assert sha(ROOT/relative)==digest
    print(json.dumps(dict(verified=True,rows=len(rows),main_figures=4,extra_figures=2,supplement_pages=3,
        source_hashes_unchanged=True,previous_study_sources_preserved=True,direct_fact_metrics_unavailable=True)))


if __name__=='__main__':main()
