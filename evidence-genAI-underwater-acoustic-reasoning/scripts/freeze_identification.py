"""Freeze the follow-up before any prediction on its newly acquired cohort."""
import json
from datetime import datetime, timezone
from pathlib import Path
from underwater_reasoning.io import atomic_json, file_hash

ROOT = Path(__file__).resolve().parents[1]


def main():
    destination = ROOT/'protocol/identification_freeze.json'
    if destination.exists():
        raise ValueError('Identification freeze already exists')
    if list((ROOT/'data/derived').glob('identification_seed*.npz')) or (ROOT/'results/raw/identification/evaluation_predictions.npz').exists():
        raise ValueError('Fresh outcomes already exist; cannot claim a prospective freeze')
    training = json.loads((ROOT/'results/manifests/identification_training.json').read_text())
    pilot = json.loads((ROOT/'results/manifests/identification_omni_pilot.json').read_text())
    cohort = json.loads((ROOT/'results/manifests/identification_cohort.json').read_text())
    assert training['fresh_test_outcomes_examined'] is False
    assert pilot['format_gate_passed'] and cohort['all_crc_checked']
    assert cohort['decoded_duplicates']==0 and cohort['original_audio_overlap']==0
    paths = ['configs/identification_study.json', 'protocol/identification_followup.md', 'prompts/identify_source.txt',
             'src/underwater_reasoning/identification.py', 'src/underwater_reasoning/omni.py', 'src/underwater_reasoning/io.py',
             'src/underwater_reasoning/runtime.py', 'scripts/train_classifier.py', 'configs/execution.json',
             'data/private/experiment_manifest.json', 'data/private/identification_manifest.json',
             'data/private/identification_waveform_hashes.json', 'data/derived/identification_fitting_physical.npz',
             'results/manifests/identification_cohort.json', 'results/manifests/identification_training.json',
             'results/manifests/identification_omni_pilot.json', 'results/manifests/classifier_training.json',
             'checkpoints/identification_fusion.joblib', 'tests/test_identification.py', 'scripts/run_identification_followup.sh']
    paths += [str(p.relative_to(ROOT)) for p in (ROOT/'scripts').glob('*identification*.py')]
    paths += [f'data/derived/classifier_seed{s}.npz' for s in [17,29,43]]
    original = json.loads((ROOT/'results/manifests/classifier_training.json').read_text())
    paths += [r['checkpoint'] for r in original['seeds']]
    record = {'frozen_utc':datetime.now(timezone.utc).isoformat(), 'fresh_test_outcomes_examined':False,
              'previous_study_outcomes_known':True, 'external_preregistration':False,
              'primary_comparator':training['primary_comparator'], 'files':{p:file_hash(ROOT/p) for p in sorted(set(paths))}}
    atomic_json(destination, record)
    print('Identification freeze:', record['frozen_utc'], len(record['files']), 'artifacts')


if __name__ == '__main__':
    main()
