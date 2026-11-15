"""Verify completed development choices and freeze their hashes before test."""
import argparse,json
from datetime import datetime,timezone
from pathlib import Path
from underwater_reasoning.io import atomic_json,canonical_hash,file_hash
from underwater_reasoning.report_schema import terminal_schema

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--unlock',action='store_true');args=parser.parse_args()
    destination=ROOT/'protocol/test_unlock_log.json'
    previous=json.loads(destination.read_text())
    if previous.get('unlocked'):raise ValueError('Test already unlocked; preserve its original freeze')
    if list((ROOT/'results/raw/terminal/main/test').glob('*.json')) or list((ROOT/'results/raw/planner/test').glob('*.json')):
        raise ValueError('Test outcomes already exist before this freeze')
    pilot=json.loads((ROOT/'results/manifests/pilot_gate.json').read_text())
    if not pilot['passed']:raise ValueError('Development gate did not pass')
    repeat=json.loads((ROOT/'results/manifests/grammar_repeat.json').read_text())
    if repeat['schema_valid']!=repeat['reports'] or repeat['format_valid']!=repeat['reports']:raise ValueError('Structured repeat smoke test contains invalid reports')
    settings=json.loads((ROOT/'configs/locked_controller.json').read_text())
    design=json.loads((ROOT/'results/manifests/pretest_design_audit.json').read_text())
    if design['test_terminal_outcomes_examined'] or design['locked_comparator']!=settings['locked_comparator']:raise ValueError('Pre-test precision audit does not match the locked design')
    manifest=json.loads((ROOT/'data/private/experiment_manifest.json').read_text())
    schema_hash=canonical_hash(terminal_schema());cache_counts={};keys=[]
    for partition in ('policy_fit','development','calibration'):
        count=0
        for uid in manifest['assessments'][partition]:
            for mask in list(range(16))+[-1]:
                path=ROOT/f'results/raw/terminal/main/{partition}/{uid}_{mask}.json'
                record=json.loads(path.read_text());key=record['key']
                if key.get('format_schema_sha256')!=schema_hash or key.get('format_enforcer')!='xgrammar==0.2.9':
                    raise ValueError('Mixed decoding conditions in comparison cache')
                keys.append([partition,uid,mask,canonical_hash(key),file_hash(path)]);count+=1
        cache_counts[partition]=count
    references=[];reviews=[]
    for part,ids in manifest['assessments'].items():
        for uid in ids:
            reference_path=ROOT/f'data/derived/references/{uid}.json';reference=json.loads(reference_path.read_text())
            if reference['reference']['version']!='independent-proxy-1.2':raise ValueError('Outdated reference estimator')
            references.append([uid,file_hash(reference_path)])
            for role in (0,1):
                path=ROOT/f'results/raw/review/main/{part}/{uid}_{role}.json';review=json.loads(path.read_text())
                if review['key']['reference_sha256']!=canonical_hash(reference['reference']):raise ValueError('AI reviewer used a different reference revision')
                reviews.append([part,uid,role,file_hash(path)])
    paths=[]
    for folder in ('src','scripts','prompts','configs','protocol'):
        paths.extend(p for p in (ROOT/folder).rglob('*') if p.is_file() and p.suffix in ('.py','.sh','.txt','.json','.md') and '__pycache__' not in p.parts and p!=destination)
    hashes={str(p.relative_to(ROOT)):file_hash(p) for p in sorted(paths)}
    record={'unlocked':args.unlock,'status':'frozen_before_test' if args.unlock else 'freeze_verified_not_unlocked',
        'frozen_at_utc':datetime.now(timezone.utc).isoformat(),'protocol_frozen':True,'externally_preregistered':False,
        'test_outcomes_examined_at_freeze':False,'locked_comparator':settings['locked_comparator'],
        'confirmatory_contrasts':['P vs '+settings['locked_comparator'],'P vs B7_class_only','P vs A1_one_shot'],
        'split_manifest_sha256':file_hash(ROOT/'data/private/experiment_manifest.json'),
        'waveform_audit_sha256':file_hash(ROOT/'data/private/waveform_hashes.json'),
        'reference_digest':canonical_hash(references),'ai_review_digest':canonical_hash(reviews),
        'pretest_terminal_cache_digest':canonical_hash(keys),'pretest_terminal_counts':cache_counts,
        'pilot_gate_sha256':file_hash(ROOT/'results/manifests/pilot_gate.json'),
        'pretest_design_audit_sha256':file_hash(ROOT/'results/manifests/pretest_design_audit.json'),
        'structured_repeat_audit_sha256':file_hash(ROOT/'results/manifests/grammar_repeat.json'),
        'software_and_protocol_hashes':hashes,'format_schema_sha256':schema_hash,
        'human_reviewers':0,'endpoint':'automated proxy supported coverage',
        'previous_preparation_record':previous}
    if args.unlock:atomic_json(destination,record)
    print(json.dumps({k:v for k,v in record.items() if k not in ('software_and_protocol_hashes','previous_preparation_record')},indent=2))

if __name__=='__main__':main()
