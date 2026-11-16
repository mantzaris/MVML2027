"""Verify method-defining files and models against the prospective test freeze."""
import json
from pathlib import Path
from underwater_reasoning.io import atomic_json,file_hash,canonical_hash

ROOT=Path(__file__).resolve().parents[1]
freeze=json.loads((ROOT/'protocol/test_unlock_log.json').read_text())
if not freeze.get('unlocked'):raise SystemExit('Test not yet frozen')
editorial={'scripts/build_manuscript.py','scripts/build_artifact_manifest.py','scripts/check_readiness.py','scripts/collect_worker_finish.py','scripts/summarize_execution.py','scripts/verify_preservation.py','scripts/sync_artifacts.py','scripts/make_figures.py','scripts/make_real_examples.py'}
changed=[];editorial_changes=[];checked=0
for relative,digest in freeze['software_and_protocol_hashes'].items():
    path=ROOT/relative;different=not path.is_file() or file_hash(path)!=digest
    if relative in editorial:
        if different:editorial_changes.append(relative)
        continue
    checked+=1
    if different:changed.append(relative)
for relative,digest in freeze['controller_checkpoint_hashes'].items():
    checked+=1
    if file_hash(ROOT/relative)!=digest:changed.append(relative)
for relative,key in [('data/derived/classifier_seed17.npz','anchor_classifier_cache_sha256'),('data/private/experiment_manifest.json','split_manifest_sha256'),('data/private/waveform_hashes.json','waveform_audit_sha256')]:
    checked+=1
    if file_hash(ROOT/relative)!=freeze[key]:changed.append(relative)
manifest=json.loads((ROOT/'data/private/experiment_manifest.json').read_text());references=[];reviews=[]
for part,ids in manifest['assessments'].items():
    for uid in ids:
        references.append([uid,file_hash(ROOT/f'data/derived/references/{uid}.json')])
        for role in (0,1):reviews.append([part,uid,role,file_hash(ROOT/f'results/raw/review/main/{part}/{uid}_{role}.json')])
if canonical_hash(references)!=freeze['reference_digest']:changed.append('main reference digest')
if canonical_hash(reviews)!=freeze['ai_review_digest']:changed.append('main AI review digest')
record={'method_files_and_models_match':not changed,'checked_files_and_models':checked,'changed_method_artifacts':changed,
    'editorial_files_changed_after_freeze':editorial_changes,'main_references_and_reviews_match':not any('digest' in p for p in changed),
    'frozen_at_utc':freeze['frozen_at_utc'],'scope':'Scientific source, prompts, configuration, protocol, controller weights, anchor cache, split and main references/reviews. Editorial rendering changes are listed separately.'}
atomic_json(ROOT/'results/manifests/freeze_integrity.json',record);print(json.dumps(record,indent=2))
if changed:raise SystemExit('Method-defining artifacts changed after freeze; investigate and disclose')
