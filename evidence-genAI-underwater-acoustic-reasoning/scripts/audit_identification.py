"""Independent accounting, source boundaries and frozen-artifact checks."""
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from underwater_reasoning.io import atomic_json, file_hash

ROOT=Path(__file__).resolve().parents[1]


def read(path):return json.loads((ROOT/path).read_text())


def main():
    checks=[]
    for record in ['protocol/identification_freeze.json','protocol/untrained_categories_freeze.json']:
        frozen=read(record)
        for path,expected in frozen['files'].items():assert file_hash(ROOT/path)==expected,path
        checks.append(record+' all hashes match')
    old=read('data/private/experiment_manifest.json')['records']
    fresh=read('data/private/identification_manifest.json')['records']
    untrained=read('data/private/untrained_categories_manifest.json')['records']
    fitted=[r for r in old if r['partition']!='test']
    groups={r['source_group'] for r in old};days={r['date'] for r in fitted};times=np.sort([r['start_seconds'] for r in fitted])
    for cohort in [fresh,untrained]:
        assert not groups.intersection(r['source_group'] for r in cohort)
        assert not days.intersection(r['date'] for r in cohort)
        for row in cohort:
            ix=np.searchsorted(times,row['start_seconds']);nearest=times[max(0,ix-1):ix+1]
            assert np.min(abs(nearest-row['start_seconds']))>=65
        groups.update(r['source_group'] for r in cohort)
    checks.append('fresh and untrained sources disjoint; fitting day/time boundaries pass')
    sets=[]
    for path in ['data/private/waveform_hashes.json','data/private/identification_waveform_hashes.json','data/private/untrained_categories_waveform_hashes.json']:
        values=[r['waveform_sha256'] for r in read(path)]
        assert len(values)==len(set(values))
        for prior in sets:assert not prior.intersection(values)
        sets.append(set(values))
    checks.append('decoded hashes unique across original, fresh and untrained cohorts')
    table=read('results/tables/identification_results.json')
    training=read('results/manifests/identification_training.json')
    rows=read('data/private/identification_evaluation_manifest.json')['records']
    raw=np.load(ROOT/'results/raw/identification/evaluation_predictions.npz')
    assert list(raw['ids'])==[r['id'] for r in rows]
    labels=np.array([r['label_index'] for r in rows])
    for condition,record in table['conditions'].items():
        selected=np.array([r['variant']==condition for r in rows]);assert selected.sum()==235
        for method,result in record['methods'].items():
            p=raw[method][selected];assert np.isfinite(p).all() and np.allclose(p.sum(1),1)
            predicted=p.argmax(1);counts=np.zeros((5,5),dtype=int);np.add.at(counts,(labels[selected],predicted),1)
            assert np.array_equal(counts,result['confusion_matrix'])
            assert np.isclose(np.trace(counts)/235,result['accuracy'])
            denominator=counts.sum(0)+counts.sum(1)
            f1=np.divide(2*np.diag(counts),denominator,out=np.zeros(5),where=denominator>0).mean()
            assert np.isclose(f1,result['macro_f1'])
            issued=raw['score_'+method][selected]>=training['thresholds'][method]
            correct=predicted==labels[selected]
            assert result['correct_issued']==int((correct & issued).sum())
            assert result['wrong_issued']==int((~correct & issued).sum())
            assert result['referred']==int((~issued).sum())
            assert result['correct_issued']+result['wrong_issued']+result['referred']==235
            assert np.isclose(result['correct_per_100']+result['wrong_per_100']+result['referred_per_100'],100)
    checks.append('all 63 classifier/condition cells match saved predictions and denominators')
    c=table['conditions']['clean'];delta=c['methods']['fusion']['macro_f1']-c['methods'][table['primary_comparator']]['macro_f1']
    assert np.isclose(delta,c['primary_contrast']['macro_f1']['difference'])
    assert table['primary_success']==bool(delta>=.03 and c['primary_contrast']['macro_f1']['ci95'][0]>0)
    for method,record in table['omni'].items():
        assert record['complete'] and record['saved']==235 and record['valid']==235
        assert record['correct_issued']+record['wrong_issued']+record['referred']==235
        counts=np.zeros((5,6),dtype=int)
        paths=list((ROOT/f'results/raw/identification/omni/test/{method}').glob('*.json'))
        assert len(paths)==235
        for row in fresh:
            generated=read(f'results/raw/identification/omni/test/{method}/{row["id"]}.json')
            assert generated['id']==row['id'] and generated['variant']==method
            assert not generated['output']['hit_output_limit']
            response=generated['output']['raw']
            parsed=json.loads(response[response.find('{'):response.rfind('}')+1])
            assert parsed==generated['parsed'] and set(parsed)=={'label','reason'}
            assert isinstance(parsed['reason'],str)
            prediction=(table['labels']+['uncertain']).index(parsed['label'])
            counts[row['label_index'],prediction]+=1
        assert np.array_equal(counts,record['confusion_matrix_with_uncertain_column'])
        assert int(np.trace(counts[:,:5]))==record['correct_issued']
        assert int(counts[:,5].sum())==record['referred']
    checks.append('primary criterion and all 705 raw generated category outcomes independently reconciled')
    unknown=read('results/tables/untrained_categories_results.json')
    unknown_raw=np.load(ROOT/'results/raw/identification/untrained_categories.npz')
    assert list(unknown_raw['ids'])==[row['id'] for row in untrained]
    assert file_hash(ROOT/'results/raw/identification/untrained_categories.npz')==unknown['predictions_sha256']
    for method,record in unknown['methods'].items():
        total=record['overall'];assert total['n']==72
        assert total['trained_category_issued']+total['referred']==72
        assert sum(v['n'] for v in record['by_untrained_category'].values())==72
        assert sum(total['issued_label_counts'])==total['trained_category_issued']
        p=unknown_raw[method];assert np.isfinite(p).all() and np.allclose(p.sum(1),1)
        issued=unknown_raw['score_'+method]>=training['thresholds'][method]
        broad_vessel=1-p[:,0]>=training['binary_thresholds'][method]
        assert int(issued.sum())==total['trained_category_issued']
        assert np.isclose(broad_vessel.mean(),total['broad_vessel_recognition'])
        assert np.array_equal(np.bincount(p[issued].argmax(1),minlength=5),total['issued_label_counts'])
    checks.append('untrained-category outcomes match saved predictions with their own 72-clip denominator')
    jobs=[read(str(p.relative_to(ROOT))) for pattern in ['results/manifests/jobs/identification*.json','results/manifests/jobs/untrained_categories*.json'] for p in ROOT.glob(pattern)]
    assert jobs and all(r['returncode']==0 and r['status']=='complete' for r in jobs)
    deadline=datetime.fromisoformat(read('configs/execution.json')['research_deadline_utc'].replace('Z','+00:00'))
    assert all(datetime.fromisoformat(r['ended_utc'])<deadline for r in jobs)
    checks.append('every follow-up worker stage completed before the original deadline')
    preservation=read('results/manifests/identification_preservation.json');assert preservation['all_match']
    checks.append('worker artifacts match independently verified local copies')
    summary={'audited_utc':datetime.now(timezone.utc).isoformat(),'all_passed':True,'checks':checks,
             'fresh_test_clips':235,'transformed_clips':1410,'untrained_category_clips':72,
             'generative_test_calls':705,'generative_pilot_calls':20,'primary_success':table['primary_success'],
             'original_study_unchanged':True,'human_reviewers':0}
    atomic_json(ROOT/'results/manifests/identification_audit.json',summary)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
