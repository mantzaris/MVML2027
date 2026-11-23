"""Stage input signatures and explicit, evidence-backed legacy receipt adoption.

Only metadata and file digests are computed here. Historical core hashes come
from native artifact receipts; newly expanded dependency hashes are not claimed
to have been recorded before this migration.
"""
import ast
import json
import math
from pathlib import Path
from uac.runtime import PROJECT_ROOT as ROOT, fingerprint, sha256_file, source_fingerprint

SEEDS = (17, 29, 43)

def read(path):
    return json.loads((ROOT / path).read_text())

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

def source_dependencies(commands):
    """Follow resolvable project imports and literal source-fingerprint paths."""
    pending = {str(Path(arg).relative_to(ROOT)) if Path(arg).is_absolute() else arg
               for command in commands for arg in command
               if str(arg).endswith('.py') and (ROOT / arg).is_file()}
    seen = set()
    def resolve(module):
        relative = Path(*module.split('.'))
        for base in (ROOT / 'src', ROOT / 'scripts'):
            for item in (base / relative.with_suffix('.py'), base / relative / '__init__.py'):
                if item.is_file():
                    pending.add(str(item.relative_to(ROOT)))
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        for node in ast.walk(ast.parse((ROOT / name).read_text(), filename=name)):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    resolve(alias.name)
            elif isinstance(node, ast.ImportFrom) and node.module:
                resolve(node.module)
                for alias in node.names:
                    resolve(node.module + '.' + alias.name)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                value = node.value
                if value.startswith(('src/', 'scripts/')) and value.endswith('.py') and (ROOT / value).is_file():
                    pending.add(value)
    return seen

def policy_paths(split):
    return {f'results/raw/policies/{split}/{mode}_seed{s}/all.json'
            for mode in ('agent', 'agent_no_generation') for s in SEEDS}

def feature_paths(splits, kinds=('real',)):
    return {f'data/features/{kind}/{split}/index.json' for kind in kinds for split in splits}

def stage_inputs(name, commands):
    paths = source_dependencies(commands) | {'scripts/project_env.sh', 'requirements-research.lock'}
    paths.add('configs/gpu.json')
    if name == 'gpu_identity':
        pass
    elif name == 'background_flow':
        paths.add('configs/research.json')
    elif name in ('synthetic_training', 'synthetic_test'):
        paths.add('configs/synthetic_temporal.json')
        if name == 'synthetic_test':
            paths.update(f'checkpoints/synthetic_temporal/seed_{s}/best.pt' for s in SEEDS)
    elif name in ('synthetic_train_features', 'synthetic_test_features'):
        paths.add('configs/synthetic_features.json')
    elif name in ('supplemental_training', 'supplemental_test'):
        paths.update(('configs/supplemental.json', 'configs/synthetic_features.json'))
        paths |= feature_paths(('train', 'development') if name.endswith('training') else ('test', 'heldout_test'), ('synthetic',))
        if name.endswith('test'):
            paths.update(('manifests/supplemental_model_index.json', 'manifests/supplemental_feature_cost.json'))
    elif name.startswith('features_'):
        split = name[len('features_'):]
        paths.update(('data/manifests/planned_real.jsonl', f'data/manifests/real_{split}.jsonl'))
    elif name.startswith('foundation_'):
        paths |= feature_paths((name[len('foundation_'):],))
        paths.add('manifests/model_lock.json')
    elif name == 'real_training':
        paths.add('configs/research.json')
        paths |= feature_paths(('train', 'development'), ('real', 'foundation'))
        paths |= feature_paths(('train',), ('synthetic',))
        paths.add('configs/synthetic_features.json')
    elif name.startswith('interventions_'):
        _, split, probe = name.split('_')
        paths |= feature_paths((split,))
        paths.update(('configs/research.json', 'checkpoints/background_flow/final.pt'))
    elif name.startswith('policies_'):
        split = name.split('_')[1]
        paths |= feature_paths((split,))
        paths.update(('configs/research.json', 'manifests/model_lock.json', 'manifests/detector_index.json', f'data/interventions/{split}/index.json'))
    elif name in ('calibration', 'real_test'):
        splits = ('development', 'calibration') if name == 'calibration' else ('test',)
        paths.update(('configs/research.json', 'manifests/detector_index.json'))
        paths |= feature_paths(splits, ('real', 'foundation'))
        for split in splits:
            paths |= policy_paths(split)
            paths.update(f'{prefix}/{split}/index.json' for prefix in ('data/interventions', 'data/interventions_full'))
        if name == 'real_test':
            paths.update(('manifests/calibration.json', 'manifests/final_development_selection.json'))
    elif name == 'live_benchmark':
        paths.update(('configs/runtime_benchmark.json', 'configs/research.json', 'manifests/model_lock.json', 'manifests/detector_index.json', 'manifests/calibration.json', 'checkpoints/background_flow/final.pt'))
        paths |= feature_paths(('calibration',))
    elif name == 'raw_acquisition':
        paths.add('data/manifests/planned_real.jsonl')
    elif name == 'freeze':
        paths.update(str(p.relative_to(ROOT)) for pattern in ('configs/*.json', 'protocol/*.md', 'src/uac/*.py', 'scripts/*.py') for p in ROOT.glob(pattern))
        # Match freeze_protocol: reporting and supervision edits do not alter
        # the experimental test boundary. Their own stage inputs still bind them.
        paths.difference_update('scripts/' + filename for filename in
                                ('build_results_report.py', 'run_research.py', 'status.py', 'snapshot_code.py'))
        paths.update(('manifests/detector_index.json', 'manifests/supplemental_model_index.json', 'manifests/calibration.json', 'manifests/final_development_selection.json', 'manifests/model_lock.json', 'results/tables/runtime_benchmark.json', 'data/manifests/planned_real.jsonl'))
        paths.add('data/manifests/real.jsonl')
        paths.update(f'data/manifests/real_{s}.jsonl' for s in ('train', 'development', 'calibration', 'test'))
        paths |= feature_paths(('train', 'development', 'calibration'), ('real', 'foundation'))
        paths |= policy_paths('development') | policy_paths('calibration')
    elif name == 'robustness':
        paths.update(('configs/robustness.json', 'configs/research.json', 'manifests/model_lock.json', 'manifests/calibration.json', 'manifests/detector_index.json', 'checkpoints/background_flow/final.pt', 'data/manifests/real_test.jsonl'))
        paths |= feature_paths(('train', 'test'))
    elif name == 'semisynthetic_test':
        paths.update(('configs/semisynthetic.json', 'configs/synthetic_temporal.json', 'data/manifests/real_test.jsonl'))
        paths.update(f'checkpoints/synthetic_temporal/seed_{s}/best.pt' for s in SEEDS)
    elif name in ('final_assembly', 'manuscript'):
        paths.update(('results/tables/real_results.json', 'results/tables/robustness.json', 'results/tables/runtime_benchmark.json', 'results/tables/semisynthetic_results.json', 'reports/synthetic_results.json', 'results/tables/supplemental_views.json'))
        if name == 'manuscript':
            paths.update(('results/tables/final_real_results.json', 'manifests/dsp_validation.json', 'data/manifests/split_audit.json', 'configs/research.json'))
    else:
        raise RuntimeError('No explicit input dependency specification for stage ' + name)
    if (name.endswith('_test') or name in ('synthetic_test_features', 'robustness', 'semisynthetic_test', 'supplemental_test') or name.startswith(('interventions_test_', 'policies_test_'))):
        paths.add('runs/test_freeze.json')
    # A model lock identifies revisions, but bytes must be pinned too.
    if 'manifests/model_lock.json' in paths:
        for key, entry in read('manifests/model_lock.json').items():
            if name.startswith('foundation_') and not key.startswith('facebook/'):
                continue
            if name.startswith('policies_') and not key.startswith('Qwen/'):
                continue
            paths.update(str(p.relative_to(ROOT)) for p in (ROOT / entry['path']).rglob('*')
                         if p.is_file() and '.cache' not in p.parts)
    # Indexes name immutable downstream model weights; bind these actual files.
    for index_name in ('manifests/detector_index.json', 'manifests/supplemental_model_index.json'):
        if index_name in paths:
            paths.update(r['checkpoint'] for r in read(index_name)['records'])
    missing = sorted(p for p in paths if not (ROOT / p).is_file())
    require(not missing, 'Required stage inputs missing for ' + name + ': ' + repr(missing))
    result = {p: sha256_file(p) for p in sorted(paths)}
    if name == 'background_flow':
        config = read('configs/research.json')
        del result['configs/research.json']
        result['config_fields:configs/research.json:generator'] = fingerprint({k: config[k] for k in ('generator_steps', 'generator_batch_size', 'generator_base_channels')})
    return result

def verify_native_receipt(path, config_hash, data_hash=None):
    obj = read(path)
    require(obj.get('status') == 'complete' and obj.get('config_hash') == config_hash
            and obj.get('data_fingerprint') == data_hash, 'Native completion/configuration mismatch: ' + path)
    for artifact, entry in obj['artifacts'].items():
        require((ROOT / artifact).stat().st_size == entry['bytes'] and sha256_file(artifact) == entry['sha256'], 'Native artifact mismatch: ' + artifact)
    return obj

def verify_resume_evidence(directory, code_hash, config_hash, minimum_steps):
    candidates = []
    for path in (ROOT / directory).glob('generation-*.json'):
        meta = json.loads(path.read_text())
        checksum = meta.pop('metadata_sha256')
        if fingerprint(meta) != checksum or meta['code_fingerprint'] != code_hash or meta['config_sha256'] != config_hash or meta['global_step'] < minimum_steps:
            continue
        payload = path.parent / meta['payload']
        if payload.stat().st_size == meta['payload_bytes'] and sha256_file(payload) == meta['payload_sha256']:
            candidates.append(str(path.relative_to(ROOT)))
    require(candidates, 'No verified completed checkpoint with current historical core code: ' + directory)
    return candidates

def native_adoption_evidence(name):
    """Fail closed for unsupported adoption; never treat a best checkpoint as completion."""
    evidence = {'basis': 'native completion artifacts revalidated', 'historical_scope': 'native recorded core configuration/code/data only',
                'expanded_dependencies': 'first pinned at migration/adoption; no retrospective source-hash claim'}
    if name == 'gpu_identity':
        config, report = read('configs/gpu.json'), read('manifests/gpu_verification.json')
        require(report['status'] == 'passed' and report['selected_gpu_uuid'] == config['gpu_uuid'] and report['selected_gpu_name'] == config['gpu_name'], 'GPU identity receipt mismatch')
        evidence.update(historical_scope='observed device identity receipt; no historical code hash', hardware_guard='run_on_ada.py revalidates actual hardware before every GPU child')
    elif name == 'background_flow':
        config = read('configs/research.json')
        settings = {'seed':17, 'steps':config['generator_steps'], 'batch_size':config['generator_batch_size'], 'base':config['generator_base_channels'], 'samples':48000, 'method':'clamped_class_free_flow_matching', 'source':'synthetic_training_family', 'sampling_steps':8, 'full_mask_fraction':0.3}
        ch = fingerprint(settings)
        verify_native_receipt('runs/background_flow_complete.json', ch)
        evidence['native_receipts'] = ['runs/background_flow_complete.json']
        evidence['verified_checkpoint_metadata'] = verify_resume_evidence('checkpoints/background_flow/resume', source_fingerprint(['src/uac/models.py','src/uac/synthetic.py','scripts/train_flow.py']), ch, settings['steps'])
    elif name == 'synthetic_training':
        from synthetic_common import DEFAULTS, code_hash, scene_fingerprint
        config = read('configs/synthetic_temporal.json')
        require(config == DEFAULTS, 'Synthetic configuration differs from implementation')
        evidence['native_receipts'], evidence['verified_checkpoint_metadata'] = [], []
        for seed in SEEDS:
            path = f'runs/synthetic_temporal_seed_{seed}_complete.json'
            ch = fingerprint({**config, 'model_seed':seed})
            receipt = verify_native_receipt(path, ch, scene_fingerprint(config))
            require([r['epoch'] for r in receipt['details']['history']] == list(range(1, config['epochs'] + 1)), 'Synthetic training epoch history incomplete')
            evidence['native_receipts'].append(path)
            evidence['verified_checkpoint_metadata'] += verify_resume_evidence(f'checkpoints/synthetic_temporal/seed_{seed}/resume', code_hash(), ch, config['epochs'] * math.ceil(config['train_scenes'] / config['batch_size']))
    elif name == 'synthetic_train_features':
        config = read('configs/synthetic_features.json')
        code_hash = source_fingerprint(['scripts/generate_synthetic_features.py','src/uac/synthetic.py','src/uac/dsp.py'])
        evidence['native_indices'] = []
        for split in ('train', 'development'):
            path = f'data/features/synthetic/{split}/index.json'; index = read(path)
            require(index['status'] == 'complete' and index['config'] == config and index['config_hash'] == fingerprint(config) and index['code_fingerprint'] == code_hash and index['data_fingerprint'] == sha256_file('configs/synthetic_features.json'), 'Synthetic feature provenance mismatch')
            require(index['expected_rows'] == index['completed_rows'] == config['counts'][split] == sum(r['count'] for r in index['shards']), 'Synthetic features incomplete')
            for record in index['shards']:
                require((ROOT / record['payload']).stat().st_size == record['payload_bytes'] and sha256_file(record['payload']) == record['payload_sha256'], 'Feature shard changed')
            evidence['native_indices'].append(path)
    elif name == 'supplemental_training':
        config, index = read('configs/supplemental.json'), read('manifests/supplemental_model_index.json')
        require(index['status'] == 'complete' and index['config_sha256'] == sha256_file('configs/supplemental.json') and index['code_fingerprint'] == source_fingerprint(['scripts/supplemental_views.py','src/uac/models.py','src/uac/training.py']), 'Supplemental provenance mismatch')
        require(index['synthetic_features_config_sha256'] == sha256_file('configs/synthetic_features.json'), 'Supplemental data configuration changed')
        require({(r['method'], r['seed']) for r in index['records']} == {(m, s) for m in config['methods'] for s in config['seeds']} and len(index['records']) == len(config['methods']) * len(config['seeds']), 'Supplemental grid incomplete')
        for split in ('train', 'development'):
            require(index[split + '_index_sha256'] == sha256_file(f'data/features/synthetic/{split}/index.json'), 'Supplemental source index changed')
        require(index['feature_cost_sha256'] == sha256_file('manifests/supplemental_feature_cost.json'), 'Supplemental timing changed')
        for record in index['records']:
            require(sha256_file(record['checkpoint']) == record['checkpoint_sha256'], 'Supplemental checkpoint changed')
        evidence['native_indices'] = ['manifests/supplemental_model_index.json']
    else:
        raise RuntimeError('No native evidence validator for adoption/migration of ' + name)
    evidence['evidence_sha256'] = {p: sha256_file(p) for key in ('native_receipts', 'native_indices', 'verified_checkpoint_metadata') for p in evidence.get(key, [])}
    return evidence
