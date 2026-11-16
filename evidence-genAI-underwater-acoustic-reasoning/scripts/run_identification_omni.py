"""Object-category generation baselines; no human labels or query metadata in prompts."""
from __future__ import annotations
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.special import softmax
from underwater_reasoning.io import atomic_json, canonical_hash, file_hash
from underwater_reasoning.omni import Omni
from underwater_reasoning.runtime import budget_check, seed_all

ROOT = Path(__file__).resolve().parents[1]
LABELS = ['background', 'cargo', 'passenger', 'tanker', 'tug_tow']


def parse(raw):
    try:
        # One deterministic fence/preamble removal, no content repair or second generation.
        start, end = raw.find('{'), raw.rfind('}')
        value = json.loads(raw[start:end+1])
        if set(value) != {'label', 'reason'} or value['label'] not in LABELS+['uncertain'] or not isinstance(value['reason'], str):
            return None
        return value
    except (ValueError, TypeError):
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pilot', action='store_true')
    args = parser.parse_args()
    seed_all(1729)
    if args.pilot:
        original = json.loads((ROOT/'data/private/experiment_manifest.json').read_text())['records']
        rows = []
        for label in LABELS:
            rows += sorted([r for r in original if r['partition']=='development' and r['label']==label],
                           key=lambda r: canonical_hash(['identification_pilot', r['id']]))[:4]
        variants = ['raw']
        key = 'pilot'
    else:
        frozen = json.loads((ROOT/'protocol/identification_freeze.json').read_text())
        for path, expected in frozen['files'].items():
            if file_hash(ROOT/path) != expected:
                raise ValueError('Freeze mismatch: '+path)
        rows = json.loads((ROOT/'data/private/identification_manifest.json').read_text())['records']
        variants = ['raw', 'acoustic_prior', 'fused_evidence']
        key = 'test'
        bundle = np.load(ROOT/'results/raw/identification/evaluation_predictions.npz')
        indices = {uid:i for i,uid in enumerate(bundle['ids'])}
        physical = np.load(ROOT/'data/derived/identification_test_physical.npz')
        physical_indices = {uid:i for i,uid in enumerate(physical['ids'])}
    prompt = (ROOT/'prompts/identify_source.txt').read_text()
    tasks = []
    for row in rows:
        for variant in variants:
            path = ROOT/f'results/raw/identification/omni/{key}/{variant}/{row["id"]}.json'
            if path.exists():
                continue
            text = prompt
            if variant != 'raw':
                p = bundle['anchor' if variant=='acoustic_prior' else 'fusion'][indices[row['id']]]
                text += '\nClassifier category probabilities: '+json.dumps(dict(zip(LABELS, np.round(p, 5).tolist())))
            if variant == 'fused_evidence':
                f = physical['features'][physical_indices[row['id']]]
                evidence = {'relative_band_energy': dict(zip(['20-100 Hz','100-500 Hz','500-2000 Hz','2000-8000 Hz'], np.round(f[32:36], 5).tolist())),
                            'first_last_received_energy_change_dB': round(float(f[42]*60), 2),
                            'fraction_digital_full_scale': round(float(f[45]), 6),
                            'class_analogy_probabilities': dict(zip(LABELS, np.round(bundle['prototype_vote'][indices[row['id']]], 5).tolist()))}
                text += '\nMeasured acoustic evidence (relative recording values, not sound pressure or proof of category): '+json.dumps(evidence)
            tasks.append((row, variant, path, text))
    if tasks:
        model = Omni(ROOT)
        for offset in range(0, len(tasks), 4):
            budget_check(reserve_seconds=420)
            batch = tasks[offset:offset+4]
            items = []
            for row, _, _, text in batch:
                audio, fs = sf.read(ROOT/row['audio_path'], dtype='float32')
                items.append({'audio':audio, 'sample_rate':fs, 'text':text})
            outputs = model.generate(items, max_new_tokens=128, report_schema=False)
            for (row, variant, path, text), output in zip(batch, outputs):
                parsed = None if output['hit_output_limit'] else parse(output['raw'])
                atomic_json(path, {'id':row['id'], 'variant':variant, 'prompt_sha256':canonical_hash(text),
                                   'output':output, 'parsed':parsed})
            if (offset+len(batch)) % 40 == 0 or offset+len(batch)==len(tasks):
                print('identification Omni', key, offset+len(batch), '/', len(tasks), flush=True)
    records = [json.loads(p.read_text()) for p in (ROOT/f'results/raw/identification/omni/{key}').glob('*/*.json')]
    summary = {'recorded_utc':datetime.now(timezone.utc).isoformat(), 'mode':key, 'requested':len(rows)*len(variants),
               'saved':len(records), 'valid':sum(r['parsed'] is not None for r in records),
               'output_limit_hits':sum(r['output']['hit_output_limit'] for r in records),
               'max_new_tokens':128, 'model_revision': 'ae9e1690543ffd5c0221dc27f79834d0294cba00',
               'prompt_sha256':file_hash(ROOT/'prompts/identify_source.txt'),
               'amortized_generation_seconds':sum(r['output']['amortized_seconds'] for r in records),
               'input_tokens_including_audio':sum(r['output']['input_tokens_with_audio'] for r in records),
               'output_tokens':sum(r['output']['output_tokens'] for r in records)}
    if args.pilot:
        summary['format_gate_passed'] = summary['saved']==summary['requested'] and summary['valid']/summary['requested']>=.95
    atomic_json(ROOT/f'results/manifests/identification_omni_{key}.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
