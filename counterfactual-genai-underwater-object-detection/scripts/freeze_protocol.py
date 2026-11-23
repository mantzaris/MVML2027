#!/usr/bin/env python3
"""Commit the test boundary only after complete, verified development evidence."""
import json
from pathlib import Path
from uac.runtime import atomic_json,sha256_file,source_fingerprint,fingerprint,EventLedger,RunLock,utc_now
from uac.protocol import require_frozen_test

ROOT=Path(__file__).resolve().parents[1]

def read(name):return json.loads((ROOT/name).read_text())
def require(condition,message):
    if not condition:raise RuntimeError(message)

TEST_ARTIFACT_PATTERNS=(
    'data/features/real/test', 'data/features/foundation/test',
    'data/features/synthetic/test', 'data/features/synthetic/heldout_test',
    'data/interventions/test', 'data/interventions_full/test',
    'results/raw/policies/test', 'results/raw/real_predictions_test_seed*.pt',
    'results/raw/supplemental/test', 'results/raw/supplemental/heldout_test',
    'results/raw/robustness', 'predictions/synthetic/seed_*/test',
    'predictions/semisynthetic', 'reports/synthetic_results.json',
    'results/tables/real_results.json', 'results/tables/final_real_results.json',
    'results/tables/robustness.json', 'results/tables/semisynthetic_results.json',
    'results/tables/supplemental_views.json',
)

def prefreeze_test_artifacts(root=None):
    # Directory checks also reject interrupted shards with no completed index.
    root=ROOT if root is None else Path(root)
    return sorted({str(path.relative_to(root)) for pattern in TEST_ARTIFACT_PATTERNS
                   for path in root.glob(pattern) if path.exists()})

def verify_policy_binding(config,expected_code_hash,expected_interventions_hash):
    require(config.get('code_hash')==expected_code_hash,'Policy source fingerprint differs from current run_policies implementation')
    require(config.get('interventions_hash')==expected_interventions_hash,'Policy intervention index differs from current central evidence')

def freeze():
    destination=ROOT/'runs/test_freeze.json'
    if destination.exists():
        existing=require_frozen_test();print(json.dumps({'status':'already_frozen','sha256':sha256_file(destination)}));return existing
    seeds=read('configs/research.json')['seeds'];paths=set()
    forbidden=prefreeze_test_artifacts()
    require(not forbidden,'Test-derived artifact exists before first freeze; audit rather than asserting untouched test: '+repr(forbidden))
    def pin(name):
        path=ROOT/name
        require(path.is_file(),'Missing prerequisite: '+name)
        paths.add(name)
    # load_rows uses the global manifest once acquisition publishes it. Pin
    # that actual row source and require identical split ordering/content.
    pin('data/manifests/real.jsonl')
    global_rows=[json.loads(line) for line in (ROOT/'data/manifests/real.jsonl').read_text().splitlines() if line.strip()]
    require(all(row['split'] in ('train','development','calibration','test') for row in global_rows),'Unexpected split in global real manifest')
    for split in ('train','development','calibration','test'):
        pin('data/manifests/real_'+split+'.jsonl')
        split_rows=[json.loads(line) for line in (ROOT/f'data/manifests/real_{split}.jsonl').read_text().splitlines() if line.strip()]
        require([row for row in global_rows if row['split']==split]==split_rows,'Global and split manifest content/order differ: '+split)
        if split!='test':
            for kind in ('real','foundation'):
                name=f'data/features/{kind}/{split}/index.json';index=read(name)
                require(index['status']=='complete','Incomplete cache: '+name);pin(name)
                for item in index['shards']:
                    require(sha256_file(item['payload'])==item['payload_sha256'],'Corrupt feature payload')
    detectors=read('manifests/detector_index.json')
    methods=['fourier','stft','wavelet','fourier_stft','fourier_wavelet','stft_wavelet','main','hht_exploratory','cnn','foundation','augmentation']
    for method in methods:
        for seed in seeds:
            rows=[r for r in detectors['records'] if r['method']==method and r['seed']==seed]
            require(len(rows)==1,f'Missing/duplicate detector {method}/{seed}')
            row=rows[0];require(sha256_file(row['checkpoint'])==row['checkpoint_sha256'],'Detector digest mismatch');pin(row['checkpoint'])
    supplemental=read('manifests/supplemental_model_index.json')
    require(supplemental['status']=='complete' and len(supplemental['records'])==12,'Supplemental training incomplete')
    for row in supplemental['records']:
        require(sha256_file(row['checkpoint'])==row['checkpoint_sha256'],'Supplemental checkpoint changed');pin(row['checkpoint'])
    calibration=read('manifests/calibration.json');selection=read('manifests/final_development_selection.json')
    require(set(calibration['seeds'])==set(map(str,seeds)),'Incomplete calibration')
    require(bool(selection['locked_comparator']),'Comparator not selected')
    from uac.agent import SYSTEM_PROMPT
    expected_prompt=fingerprint(SYSTEM_PROMPT)
    state_gate=read('manifests/agent_constrained_validation.json')
    require(state_gate['constrained_decoding']=='finite_legal_json_trie_v1' and state_gate['schema_validity_by_construction'],'Missing constrained integration gate')
    require(state_gate['code_fingerprint']==source_fingerprint(['src/uac/agent.py','src/uac/agent_batch.py','scripts/validate_constrained_agent.py']),'Constrained validation source changed')
    for row in state_gate['records']:
        require(sha256_file(row['artifact']['path'])==row['artifact']['sha256'],'Constrained fixture artifact changed');pin(row['artifact']['path'])
    require(state_gate['status']=='passed' and state_gate['valid_fraction']>=.98,'Production-shaped agent fixture gate failed')
    require(all(call['prompt_sha256']==expected_prompt for row in state_gate['records'] for call in row['calls']),'Agent fixture gate uses an outdated prompt')
    benchmark=read('results/tables/runtime_benchmark.json')
    require(benchmark['status']=='complete','Live benchmark incomplete')
    require(len(benchmark['records'])==5 and all(len(r['canonical_policy_parity'])==3 and all(p['status']=='matched' for p in r['canonical_policy_parity']) for r in benchmark['records']),'Live benchmark differs from archived calibrated inference; audit before test freeze')
    actions=[]
    policy_code_hash=source_fingerprint(['scripts/run_policies.py','src/uac/agent.py','src/uac/agent_batch.py','src/uac/verification.py'])
    for split in ('development','calibration'):
        for prefix in ('data/interventions','data/interventions_full'):
            name=f'{prefix}/{split}/index.json';require(read(name)['status']=='complete','Missing interventions');pin(name)
        for seed in seeds:
            for mode in ('agent','agent_no_generation'):
                name=f'results/raw/policies/{split}/{mode}_seed{seed}/all.json';obj=read(name);pin(name)
                actions.extend(t for r in obj['records'] for t in r['trace'])
                native=[json.loads(line) for line in (ROOT/f'data/manifests/real_{split}.jsonl').read_text().splitlines()]
                require([r['id'] for r in obj['records']]==[r['id'] for r in native],'Incomplete or reordered policy')
                require(obj['config']['batch_size']==read('configs/research.json')['policy_batch_size'],'Policy batch does not match prospective selection')
                require(obj['config']['detectors_hash']==sha256_file('manifests/detector_index.json'),'Policy uses outdated detector index')
                verify_policy_binding(obj['config'],policy_code_hash,fingerprint(read(f'data/interventions/{split}/index.json')))
                policy_index=read(str(Path(name).parent/'index.json'))
                require(policy_index['status']=='complete' and policy_index['sha256']==sha256_file(name),'Policy summary integrity failed')
    require(len(actions)>0,'No measured acoustic agent actions')
    require(all(t['prompt_sha256']==expected_prompt for t in actions),'Measured acoustic actions use an outdated prompt')
    require(all(t.get('constrained_decoding')=='finite_legal_json_trie_v1' and t.get('schema_validity_by_construction') for t in actions),'Unconstrained policy traces cannot enter final freeze')
    fraction=sum(t['valid'] for t in actions)/len(actions)
    require(fraction>=.98,'Measured acoustic agent format validity below predeclared 98% gate')
    agent_gate={'status':'passed','scope':'actual development and calibration acoustic states','n_actions':len(actions),'valid_fraction':fraction,'criterion':'>=0.98 integration legality; enforced by constrained decoding, not unconstrained competence','schema_validity_by_construction':True,'test_accessed':False}
    atomic_json('manifests/measured_agent_gate.json',agent_gate)
    for name in ['manifests/detector_index.json','manifests/supplemental_model_index.json','manifests/supplemental_feature_cost.json','manifests/calibration.json','manifests/final_development_selection.json','manifests/model_lock.json','manifests/measured_agent_gate.json','manifests/verification_validation.json','manifests/agent_policy_state_validation.json','manifests/agent_constrained_validation.json','manifests/agent_unconstrained_development_failure.json','manifests/supervisor_signal_validation.json','manifests/agent_batch_scaling.json','manifests/dsp_validation.json','manifests/cuda_resume_validation.json','manifests/training_v2_boundary_validation.json','results/tables/runtime_benchmark.json','checkpoints/background_flow/final.pt','data/manifests/planned_real.jsonl','data/manifests/split_audit.json','data/manifests/dependence_audit.json','requirements-research.lock','configs/runtime_benchmark.json','configs/robustness.json','scripts/evaluate_robustness.py','scripts/benchmark_live.py']:
        pin(name)
    # Pin source, prompts, immutable protocols and pretrained weight/config files.
    for folder,pattern in [('src','*.py'),('scripts','*.py'),('configs','*.json'),('protocol','*.md')]:
        for path in sorted((ROOT/folder).rglob(pattern)):
            if path.name not in ('build_results_report.py','run_research.py','status.py','snapshot_code.py'):pin(str(path.relative_to(ROOT)))
    for path in sorted((ROOT/'checkpoints/pretrained').rglob('*')):
        if path.is_file() and '.cache' not in path.parts:pin(str(path.relative_to(ROOT)))
    temporal={}
    for seed in seeds:
        name=f'checkpoints/synthetic_temporal/seed_{seed}/best.pt';pin(name);temporal[name]=sha256_file(name)
    from evaluate_semisynthetic import code_fingerprint as semi_code
    from evaluate_robustness import code_fingerprint as robustness_code
    record={'schema_version':1,'status':'frozen','created_utc':utc_now(),'data_fingerprint':sha256_file('data/manifests/planned_real.jsonl'),'locked_comparator':selection['locked_comparator'],'primary_metric':'Descriptive AP difference of mean-seed calibrated probability; one connected day-vessel test component. Separate nonconfirmatory paired-day bootstrap sensitivity, 2000 draws','synthetic_config_sha256':sha256_file('configs/synthetic_temporal.json'),'synthetic_checkpoints':temporal,'synthetic_features_config_sha256':sha256_file('configs/synthetic_features.json'),'semisynthetic_config_sha256':sha256_file('configs/semisynthetic.json'),'semisynthetic_code_fingerprint':semi_code(),'semisynthetic_parent_manifest_sha256':sha256_file('data/manifests/real_test.jsonl'),'robustness_config_sha256':sha256_file('configs/robustness.json'),'robustness_code_fingerprint':robustness_code(),'robustness_parent_manifest_sha256':sha256_file('data/manifests/real_test.jsonl'),'artifacts':{name:{'sha256':sha256_file(name)} for name in sorted(paths)},'no_test_outcomes_seen':True}
    atomic_json(destination,record);EventLedger().append('final_protocol_frozen',sha256=sha256_file(destination),artifacts=len(paths),comparator=selection['locked_comparator'],agent_gate=agent_gate)
    print(json.dumps({'status':'frozen','artifacts':len(paths),'comparator':selection['locked_comparator']}),flush=True)
    return record

if __name__=='__main__':
    with RunLock('runs/training.lock'):freeze()
