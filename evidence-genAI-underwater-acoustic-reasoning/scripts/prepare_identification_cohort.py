"""Construct and acquire fresh source groups without opening model outcomes."""
from __future__ import annotations
import argparse
import concurrent.futures as cf
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from underwater_reasoning.io import atomic_json, canonical_hash, file_hash
from prepare_experiment import load_records, sample
from download_experiment import acquire

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    original = json.loads((ROOT/'data/private/experiment_manifest.json').read_text())
    old = original['records']
    rows, _ = load_records()
    ids = {r['id'] for r in old}
    groups = {r['source_group'] for r in old}
    fitted = [r for r in old if r['partition'] != 'test']
    days = {r['date'] for r in fitted}
    times = np.sort([r['start_seconds'] for r in fitted])
    candidates = []
    for row in rows:
        if row['id'] in ids or row['source_group'] in groups or row['date'] in days:
            continue
        i = np.searchsorted(times, row['start_seconds'])
        nearest = times[max(0, i-1):i+1]
        if len(nearest) and np.min(abs(nearest-row['start_seconds'])) < 65:
            continue
        candidates.append({**row, 'partition': 'identification_test'})
    config = json.loads((ROOT/'configs/identification_study.json').read_text())
    selected, sampling = sample(candidates, 'identification_test', config['fresh_test_requested_per_class'])
    selected.sort(key=lambda r: r['id'])
    assert not ids.intersection(r['id'] for r in selected)
    assert not groups.intersection(r['source_group'] for r in selected)
    assert not days.intersection(r['date'] for r in selected)
    manifest = {'records': selected, 'labels': original['labels'], 'sampling': sampling,
                'original_manifest_sha256': file_hash(ROOT/'data/private/experiment_manifest.json')}
    manifest_path = ROOT/'data/private/identification_manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError('Refusing to replace an existing cohort')
    atomic_json(manifest_path, manifest)
    summary = {'created_utc': datetime.now(timezone.utc).isoformat(), 'candidate_clips': len(candidates),
               'test_clips': len(selected), 'source_groups': len({r['source_group'] for r in selected}),
               'utc_days': len({r['date'] for r in selected}), 'sampling': sampling,
               'manifest_sha256': file_hash(manifest_path), 'old_test_days_overlap': len({r['date'] for r in selected} & {r['date'] for r in old if r['partition']=='test'}),
               'official_split_counts': dict(Counter(r['official_split'] for r in selected)),
               'new_source_groups': True, 'day_disjoint_from_fitting_selection_calibration': True,
               'global_60_second_guard': True, 'model_outcomes_examined': False,
               'not_official_test_split': True, 'missing_parent_provenance': True}
    if args.download:
        index = {e['name']: e for e in json.loads((ROOT/'.cache/release_index.json').read_text())}
        with cf.ThreadPoolExecutor(max_workers=8) as pool:
            results = []
            for result in pool.map(acquire, [(r, index[r['archive_name']]) for r in selected]):
                results.append(result)
                if len(results) % 50 == 0:
                    print('acquired', len(results), '/', len(selected), flush=True)
        hashes = [r['waveform_sha256'] for r in results]
        old_hashes = {r['waveform_sha256'] for r in json.loads((ROOT/'data/private/waveform_hashes.json').read_text())}
        if len(set(hashes)) != len(hashes) or old_hashes.intersection(hashes):
            raise ValueError('Decoded audio duplicate in fresh cohort')
        atomic_json(ROOT/'data/private/identification_waveform_hashes.json', results)
        summary.update(all_crc_checked=True, decoded_duplicates=0, original_audio_overlap=0,
                       audio_bytes=sum(r['bytes'] for r in results),
                       waveform_hash_manifest_sha256=file_hash(ROOT/'data/private/identification_waveform_hashes.json'))
    atomic_json(ROOT/'results/manifests/identification_cohort.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
