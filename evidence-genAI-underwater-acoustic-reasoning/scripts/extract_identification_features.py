"""CPU feature extraction; test variants are prepared only after the freeze."""
from __future__ import annotations
import argparse
import concurrent.futures as cf
import json
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy import signal
from underwater_reasoning.identification import physical_features
from underwater_reasoning.io import canonical_hash, atomic_json, file_hash

ROOT = Path(__file__).resolve().parents[1]


def transform(x, fs, name, uid):
    rms = float(np.sqrt(np.mean(x*x)))
    rng = np.random.default_rng(int(canonical_hash([1729, uid, name])[:8], 16))
    if name.startswith('noise_'):
        snr = int(name.split('_')[1])
        noise = rng.normal(size=len(x))
        noise *= rms/max(float(np.sqrt(np.mean(noise*noise))), 1e-20)*10**(-snr/20)
        return (x+noise).astype('float32')
    if name == 'lowpass_2000':
        return signal.sosfiltfilt(signal.butter(4, 2000, fs=fs, output='sos'), x).astype('float32')
    if name == 'gain_025':
        return (x*.25).astype('float32')
    if name == 'clipping':
        return np.clip(x/max(rms, 1e-20)*.4, -1, 1).astype('float32')
    raise ValueError(name)


def extract(row):
    x, fs = sf.read(ROOT/row['audio_path'], dtype='float32')
    return physical_features(x, fs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('partition', choices=['fitting', 'test'])
    args = parser.parse_args()
    if args.partition == 'fitting':
        rows = [r for r in json.loads((ROOT/'data/private/experiment_manifest.json').read_text())['records']
                if r['partition'] in ['policy_fit', 'development', 'calibration']]
    else:
        frozen = json.loads((ROOT/'protocol/identification_freeze.json').read_text())
        for path, expected in frozen['files'].items():
            if file_hash(ROOT/path) != expected:
                raise ValueError('Freeze mismatch: '+path)
        original = json.loads((ROOT/'data/private/identification_manifest.json').read_text())['records']
        variants = json.loads((ROOT/'configs/identification_study.json').read_text())['stress_conditions']
        rows = [{**r, 'variant': 'clean', 'original_id': r['id']} for r in original]
        for row in original:
            x, fs = sf.read(ROOT/row['audio_path'], dtype='float32')
            for variant in variants:
                uid = row['id']+'_'+variant
                path = 'data/raw/identification_stress/'+uid+'.wav'
                out = ROOT/path
                out.parent.mkdir(parents=True, exist_ok=True)
                if not out.exists():
                    sf.write(out, transform(x, fs, variant, row['id']), fs, subtype='FLOAT')
                rows.append({**row, 'id': uid, 'variant': variant, 'original_id': row['id'], 'audio_path': path})
        atomic_json(ROOT/'data/private/identification_evaluation_manifest.json', {'records': rows})
    path = ROOT/f'data/derived/identification_{args.partition}_physical.npz'
    if path.exists():
        old = np.load(path)
        if list(old['ids']) != [r['id'] for r in rows]:
            raise ValueError('Feature cache does not match cohort')
        print('existing physical features', len(rows))
        return
    with cf.ProcessPoolExecutor(max_workers=4) as pool:
        values = []
        for value in pool.map(extract, rows, chunksize=16):
            values.append(value)
            if len(values) % 200 == 0:
                print('physical features', len(values), '/', len(rows), flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, ids=np.array([r['id'] for r in rows]), features=np.array(values))
    print('physical features complete', len(rows), flush=True)


if __name__ == '__main__':
    main()
