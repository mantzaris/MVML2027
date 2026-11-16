"""Fresh untrained-category cohort and its pre-inference supplementary freeze."""
import concurrent.futures as cf
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from underwater_reasoning.io import atomic_json, file_hash
import prepare_experiment as original
from download_experiment import acquire

ROOT=Path(__file__).resolve().parents[1]


def main():
    if (ROOT/'protocol/untrained_categories_freeze.json').exists():
        raise ValueError('Supplementary cohort already frozen')
    labels=['pilotvessel','rescue','sailing','pleasurecraft','fishing','dredger']
    original.LABEL_MAP={s:s for s in labels};original.LABELS=labels
    rows,_=original.load_records()
    old=json.loads((ROOT/'data/private/experiment_manifest.json').read_text())['records']
    fresh=json.loads((ROOT/'data/private/identification_manifest.json').read_text())['records']
    seen={r['source_group'] for r in old+fresh}
    fitted=[r for r in old if r['partition']!='test'];days={r['date'] for r in fitted}
    times=np.sort([r['start_seconds'] for r in fitted]);candidates=[]
    for row in rows:
        if row['source_group'] in seen or row['date'] in days:continue
        ix=np.searchsorted(times,row['start_seconds']);nearest=times[max(0,ix-1):ix+1]
        if len(nearest) and np.min(abs(nearest-row['start_seconds']))<65:continue
        candidates.append({**row,'partition':'untrained_category'})
    selected,sampling=original.sample(candidates,'untrained_category',15)
    selected.sort(key=lambda r:r['id'])
    # No valid five-class target exists. The loader-only placeholder is never scored or fitted.
    selected=[{**r,'untrained_category':r['label'],'label_index':0} for r in selected]
    manifest={'records':selected,'labels':labels,'sampling':sampling,'label_index_is_unused_loader_placeholder':True}
    atomic_json(ROOT/'data/private/untrained_categories_manifest.json',manifest)
    index={r['name']:r for r in json.loads((ROOT/'.cache/release_index.json').read_text())}
    with cf.ThreadPoolExecutor(max_workers=8) as pool:
        hashes=list(pool.map(acquire,[(r,index[r['archive_name']]) for r in selected]))
    existing=set()
    for path in ['data/private/waveform_hashes.json','data/private/identification_waveform_hashes.json']:
        existing.update(r['waveform_sha256'] for r in json.loads((ROOT/path).read_text()))
    values=[r['waveform_sha256'] for r in hashes]
    assert len(values)==len(set(values)) and not existing.intersection(values)
    atomic_json(ROOT/'data/private/untrained_categories_waveform_hashes.json',hashes)
    summary={'recorded_utc':datetime.now(timezone.utc).isoformat(),'clips':len(selected),
             'source_groups':len({r['source_group'] for r in selected}),'utc_days':len({r['date'] for r in selected}),
             'sampling':sampling,'all_crc_checked':True,'decoded_duplicates':0,'overlap_with_previous_audio':0,
             'new_source_groups':True,'day_and_time_separated_from_fitting':True,'primary_results_already_known':True,
             'supplementary_outcomes_examined':False,'bytes':sum(r['bytes'] for r in hashes)}
    atomic_json(ROOT/'results/manifests/untrained_categories_cohort.json',summary)
    paths=['protocol/untrained_categories_followup.md','scripts/prepare_untrained_categories.py',
           'scripts/evaluate_untrained_categories.py','protocol/identification_freeze.json',
           'data/private/untrained_categories_manifest.json','data/private/untrained_categories_waveform_hashes.json',
           'results/manifests/untrained_categories_cohort.json']
    atomic_json(ROOT/'protocol/untrained_categories_freeze.json',{'frozen_utc':datetime.now(timezone.utc).isoformat(),
        'supplementary_outcomes_examined':False,'primary_results_already_known':True,
        'files':{p:file_hash(ROOT/p) for p in paths}})
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
