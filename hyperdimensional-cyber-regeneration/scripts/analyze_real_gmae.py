#!/usr/bin/env python3
"""Audit and exploratory paired analysis of two CTU-13 GraphMAE variants.

All learning/inference was performed on CUDA. This script only audits recorded
outputs, computes CPU statistics and produces the human-readable results report.
The test captures were inspected previously: nominal multiplicity adjustment
cannot turn this follow-up into a fresh confirmatory evaluation.
"""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from analyze_real import (average_precision, metrics, ranked, WeightedAP, sha,
                          SEEDS, SCENARIOS, CONDITIONS)

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/real_gmae'
BASE=ROOT/'results/real'
IMPORTED=['NB','LOGIT','DAE','DENSE','GMAE','HDC','SCR_FLOW','CTX_FLOW','LOCAL_FLOW']
NEW=['EDGE_GMAE','EDGE_DENSE','CONS_GMAE','CONS_PRE']
VARIANTS=['EDGE_GMAE','CONS_GMAE']
METHODS=IMPORTED+NEW
LABELS={'NB':'Gaussian naive Bayes','LOGIT':'Logistic classifier','DAE':'Denoising autoencoder',
'DENSE':'Graph classifier without pretraining','GMAE':'Original GraphMAE-adapted baseline',
'HDC':'HDC-augmented masked graph classifier','SCR_FLOW':'Shared-context mixture',
'CTX_FLOW':'Single expert with context','LOCAL_FLOW':'Single expert without context',
'EDGE_GMAE':'Observed-overlap gated GraphMAE','EDGE_DENSE':'Observed-overlap gate without pretraining',
'CONS_GMAE':'Corruption-consistent GraphMAE','CONS_PRE':'Consistency pretraining only'}


def load_rows(path=OUT/'summary.csv'):
    with Path(path).open() as stream:return list(csv.DictReader(stream))


def select(rows,**criteria):
    return [r for r in rows if all(str(r.get(k))==str(v) for k,v in criteria.items())]


def raw_path(seed,method,condition,scenario):
    origin='real_gmae' if method in NEW else 'real'
    return ROOT/'results/runs'/origin/str(seed)/f'{method}_{condition}_{scenario}.npz'


def write_csv(path,rows):
    with Path(path).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)


def audit(manifest,rows):
    if manifest['settings'].get('pilot'):raise ValueError('Pilot outputs are not study results')
    if manifest['settings']['seeds']!=SEEDS:raise ValueError('Unexpected training seeds')
    if set(manifest['methods'])!=set(METHODS) or set(manifest['new_methods'])!=set(NEW):raise ValueError('Method catalogue mismatch')
    if set(manifest['imported_methods'])!=set(IMPORTED):raise ValueError('Imported method catalogue mismatch')
    for method in METHODS:
        if manifest['method_origins'][method]!=('real_gmae' if method in NEW else 'real'):raise ValueError('Invalid method origin')
    if sha(BASE/'manifest.json')!=manifest['base_manifest_sha256']:raise ValueError('Prior manifest changed')
    if sha(BASE/'summary.csv')!=manifest['base_summary_sha256']:raise ValueError('Prior summary changed')
    if sha(BASE/'curves.csv')!=manifest['base_curves_sha256']:raise ValueError('Prior curves changed')
    if sha(BASE/'calibration.csv')!=manifest['base_calibration_sha256']:raise ValueError('Prior calibration changed')
    base=json.loads((BASE/'manifest.json').read_text())
    source_checks=0
    for record in [base,manifest]:
        for relative,digest in record['source_hashes'].items():
            if sha(ROOT/relative)!=digest:raise ValueError('Scientific source changed: '+relative)
            source_checks+=1
    for key in ['case_hashes','train_hash','validation_hashes','preprocessing','data_manifest_sha256']:
        if manifest[key]!=base[key]:raise ValueError('Changed paired input/preprocessing contract: '+key)
    dataset_path=BASE/'ctu13_data_manifest.json'
    if sha(dataset_path)!=manifest['data_manifest_sha256']:raise ValueError('Changed data preparation manifest')
    dataset=json.loads(dataset_path.read_text());temporal=dataset['temporal_split_check']
    if not dataset['complete'] or not temporal['passed']:raise ValueError('Incomplete or overlapping dataset')
    if not (temporal['latest_train_completion']<temporal['earliest_validation_start'] and temporal['latest_validation_completion']<temporal['earliest_test_start']):raise ValueError('Actual completion-time split failed')
    for record in dataset['files']:
        if sha(ROOT/record['prepared_path'])!=record['prepared_sha256']:raise ValueError('Changed prepared capture')
    expected=set(itertools.product(SEEDS,METHODS,CONDITIONS,SCENARIOS))
    key=lambda r:(int(r['seed']),r['method'],r['condition'],int(r['scenario']))
    if len(rows)!=975 or {key(r) for r in rows}!=expected:raise ValueError('Expected 975 unique evaluation records')
    prior_rows={key(r):r for r in load_rows(BASE/'summary.csv')}
    for row in rows:
        if row['method'] in IMPORTED:
            reference=prior_rows[key(row)]
            for column,value in reference.items():
                if column in ['method','condition']:
                    if row[column]!=value:raise ValueError('Altered imported record')
                elif float(row[column])!=float(value):raise ValueError('Altered imported measurement: '+column)
    counts={r['scenario']:r['rows_retained'] for r in dataset['files']}
    reference={};raw_hashes={};primary={};calibration=[];cal_cases=[];metric_checks=0
    for row in rows:
        seed,method,condition,scenario=key(row);path=raw_path(seed,method,condition,scenario)
        relative=str(path.relative_to(ROOT));digest=sha(path);raw_hashes[relative]=digest
        if manifest['raw_hashes'].get(relative)!=digest:raise ValueError('Raw archive changed: '+relative)
        if method in IMPORTED and base['raw_hashes'].get(relative)!=digest:raise ValueError('Changed imported raw archive')
        with np.load(path,allow_pickle=False) as archive:
            y,q=archive['labels'],archive['prob'];block=archive['block'];window=archive['window'];row_id=archive['row_id']
            if y.ndim!=1 or any(v.shape!=y.shape for v in [q,block,window,row_id]):raise ValueError('Invalid saved array shapes')
            if len(y)!=int(row['n']) or len(y)!=counts[scenario] or int(y.sum())!=int(row['positives']):raise ValueError('Incorrect flow counts')
            if not np.isin(y,[0,1]).all() or not np.isfinite(q).all() or (q < -1e-6).any() or (q > 1+1e-6).any():raise ValueError('Invalid labels/probabilities')
            if len(np.unique(row_id))!=len(y) or not np.array_equal(block,window//64):raise ValueError('Invalid pairing/block metadata')
            current=tuple(v.copy() for v in [y,block,window,row_id])
            if scenario not in reference:reference[scenario]=current
            elif any(not np.array_equal(a,b) for a,b in zip(reference[scenario],current)):raise ValueError('Methods/seeds do not share paired flow metadata')
            computed=metrics(y,q,float(row['threshold']))
            for metric,value in computed.items():
                if not np.isclose(value,float(row[metric]),atol=2e-6,rtol=3e-6):raise ValueError(f'Raw/summary mismatch {relative}/{metric}')
                metric_checks+=1
            display=np.clip(q,0,1);excess=max(0.,float(np.maximum(-q,q-1).max()));out_of_range=int(((q<0)|(q>1)).sum())
            for j in range(10):
                keep=(display>=j/10)&((display<(j+1)/10) if j<9 else (display<=1))
                calibration.append(dict(seed=seed,method=method,condition=condition,scenario=scenario,bin=j,
                    count=int(keep.sum()),confidence=float(display[keep].mean()) if keep.any() else (j+.5)/10,
                    accuracy=float(y[keep].mean()) if keep.any() else float('nan')))
            binned=sum(r['count'] for r in calibration[-10:])
            if binned!=len(y):raise ValueError('Reliability bins lost observations')
            cal_cases.append(dict(seed=seed,method=method,condition=condition,scenario=scenario,n=len(y),binned=binned,
                                  out_of_range=out_of_range,max_excess=excess))
            if condition=='clean' and method in VARIANTS+['GMAE']:
                primary[(seed,method,scenario)]={'y':y.copy(),'q':q.copy(),'block':block.copy()}
    if raw_hashes!=manifest['raw_hashes']:raise ValueError('Raw archive catalogue mismatch')
    records=manifest['checkpoints'];expected_checkpoints=set(itertools.product(SEEDS,METHODS))
    if len(records)!=65 or {(int(r['seed']),r['method']) for r in records}!=expected_checkpoints:raise ValueError('Expected 65 checkpoint records')
    for record in records:
        expected_path=f"results/checkpoints/{manifest['method_origins'][record['method']]}/{record['seed']}_{record['method']}.pt"
        if record['checkpoint_path']!=expected_path or sha(ROOT/expected_path)!=record['checkpoint_sha256']:raise ValueError('Checkpoint provenance mismatch')
    parameters={(int(r['seed']),r['method']):int(r['parameters']) for r in records}
    for seed in SEEDS:
        if any(parameters[(seed,m)]!=parameters[(seed,'GMAE')] for m in NEW):raise ValueError('Graph parameter counts are not matched')
    thresholds=manifest['thresholds']
    if len(thresholds)!=65 or {(int(r['seed']),r['method']) for r in thresholds}!=expected_checkpoints:raise ValueError('Expected 65 threshold records')
    lookup={(int(r['seed']),r['method']):float(r['threshold']) for r in thresholds}
    for row in rows:
        if float(row['threshold'])!=lookup[(int(row['seed']),row['method'])]:raise ValueError('Test threshold differs from frozen validation value')
    checks=dict(summary_records=len(rows),imported_records=len(prior_rows),new_records=len(rows)-len(prior_rows),
        raw_archives_verified=len(raw_hashes),raw_summary_metric_checks=metric_checks,checkpoint_hashes_verified=len(records),
        source_hashes_verified=source_checks,prepared_capture_hashes_verified=len(dataset['files']),
        completion_time_split_verified=True,prior_baseline_records_unchanged=True,
        capture_counts={str(c):dict(flows=len(v[0]),attack_flows=int(v[0].sum()),blocks=len(np.unique(v[1])),windows=len(np.unique(v[2]))) for c,v in reference.items()})
    return checks,primary,raw_hashes,calibration,cal_cases


def bootstrap(raw,repetitions=1000):
    rng=np.random.default_rng(20261126)
    prepared={key:WeightedAP(v['y'],v['q'],v['block']) for key,v in raw.items()}
    blocks={c:len(np.unique(raw[(SEEDS[0],'GMAE',c)]['block'])) for c in SCENARIOS}
    original={key:average_precision(v['y'],v['q']) for key,v in raw.items()}
    draws={m:[] for m in VARIANTS};rejected=0;attempts=0
    while len(draws[VARIANTS[0]])<repetitions:
        attempts+=1
        if attempts>100*repetitions:raise RuntimeError('Too many zero-positive bootstrap samples')
        seed_counts=np.bincount(rng.integers(len(SEEDS),size=len(SEEDS)),minlength=len(SEEDS))
        counts={c:np.bincount(rng.integers(blocks[c],size=blocks[c]),minlength=blocks[c]) for c in SCENARIOS}
        if not all(np.dot(counts[c],prepared[(SEEDS[0],'GMAE',c)].positive_by_block)>0 for c in SCENARIOS):
            rejected+=1;continue
        effect={m:0. for m in VARIANTS}
        for i,seed in enumerate(SEEDS):
            if not seed_counts[i]:continue
            reference={c:prepared[(seed,'GMAE',c)](counts[c]) for c in SCENARIOS}
            for method in VARIANTS:
                differences=[prepared[(seed,method,c)](counts[c])-reference[c] for c in SCENARIOS]
                effect[method]+=seed_counts[i]*np.mean(differences)/len(SEEDS)
        for method in VARIANTS:draws[method].append(effect[method])
    primary={}
    for method in VARIANTS:
        effects=[float(np.mean([original[(s,method,c)]-original[(s,'GMAE',c)] for c in SCENARIOS])) for s in SEEDS]
        primary[method]=dict(method=method,comparator='GMAE',metric='macro_attack_average_precision',condition='clean',
            estimate=float(np.mean(effects)),higher_is_better=True,
            percentile_95_ci=np.quantile(draws[method],[.025,.975]).tolist(),
            bonferroni_97_5_ci=np.quantile(draws[method],[.0125,.9875]).tolist(),
            means={m:float(np.mean([original[(s,m,c)] for s in SEEDS for c in SCENARIOS])) for m in [method,'GMAE']},
            seed_effects=dict(zip(map(str,SEEDS),effects)),
            per_capture_effects={str(c):float(np.mean([original[(s,method,c)]-original[(s,'GMAE',c)] for s in SEEDS])) for c in SCENARIOS})
    design=dict(bootstrap_repetitions=repetitions,bootstrap_rng_seed=20261126,rejected_zero_positive_draws=rejected,
        blocks_per_capture={str(c):blocks[c] for c in SCENARIOS},shared_draws_across_two_contrasts=True,
        design='Cross five training seeds with contiguous 64-window blocks resampled within each fixed capture. Share block draws across methods/seeds and both contrasts; fixed equal capture weights. Exact tie-aware weighted AP. Jointly reject zero-positive capture draws.',
        multiplicity='Two prespecified-for-this-follow-up contrasts use 97.5% marginal percentile intervals (quantiles .0125/.9875), a nominal Bonferroni target of 95% simultaneous coverage if underlying intervals are valid. 95% intervals are also descriptive.',
        uncertainty_scope='Exploratory reuse of previously inspected test captures. Multiplicity adjustment does not undo adaptive dataset reuse. Intervals condition on these three captures, fixed training data and verified labels; block populations 60/6/5 are small in the latter captures. No confirmatory or unseen-organization inference.')
    return primary,design


def mean(rows,method,metric,condition='clean',scenario=None):
    chosen=select(rows,method=method,condition=condition)
    if scenario is not None:chosen=select(chosen,scenario=scenario)
    return float(np.mean([float(r[metric]) for r in chosen]))


def contrasts(rows):
    pairs=[('EDGE_GMAE','GMAE'),('EDGE_GMAE','EDGE_DENSE'),('EDGE_DENSE','DENSE'),
           ('CONS_GMAE','GMAE'),('CONS_GMAE','CONS_PRE'),('CONS_PRE','GMAE'),
           ('EDGE_GMAE','DENSE'),('CONS_GMAE','DENSE')]
    result=[]
    for first,second in pairs:
        for condition in CONDITIONS:
            values={}
            for metric in ['ap','auc','brier','balanced_accuracy','f1','recall','fpr']:
                effects=[mean(select(rows,seed=s),first,metric,condition)-mean(select(rows,seed=s),second,metric,condition) for s in SEEDS]
                error=2.776445*np.std(effects,ddof=1)/np.sqrt(5)
                values[metric]=dict(estimate=float(np.mean(effects)),seed_t_95_ci=[float(np.mean(effects)-error),float(np.mean(effects)+error)],seed_effects=dict(zip(map(str,SEEDS),map(float,effects))))
            result.append(dict(method=first,comparator=second,condition=condition,metrics=values))
    return result


def report(rows,result,manifest):
    design=result['bootstrap'];cal=result['calibration_audit']
    lines=['# Two GraphMAE variants on captured CTU-13 traffic','','This is an exploratory follow-up to the completed captured-traffic study. The same test captures were already inspected when choosing to investigate GraphMAE. All four new methods were frozen before their new test predictions and use validation-only checkpoint/threshold selection, but this is not a fresh confirmatory test. Nominal multiplicity adjustment cannot remove this adaptive reuse.','',
    '## Variants and matched controls','','**EDGE_GMAE** changes message passing using an observed-overlap compatibility weight on existing endpoint-sharing edges. It reduces messages between flows with sparse shared observations or incompatible measured features, keeps self loops, and uses the same trainable parameter count and reconstruction/supervised objectives as GMAE. **EDGE_DENSE** removes its masked pretraining. Gate values use only the inputs available to each encoder view, not hidden masked targets.','',
    '**CONS_GMAE** retains GMAE inference architecture while averaging reconstruction over two independently node-masked views and adding symmetric stop-gradient cosine alignment of their latent encodings over all valid nodes, then prediction consistency under additional feature masking during supervised training. The original view is the supplied observation, which can already be corrupted; it is not oracle clean truth. **CONS_PRE** keeps the consistency pretraining but removes the supervised consistency term. Stop-gradient views use the same current model, not a separately trained or EMA teacher.','',
    'These are task-specific combinations of established masked reconstruction, input-dependent message weighting and consistency regularization. No first-ever architecture claim is made. The [method and equations](REAL_GMAE_VARIANTS.md) document the lineage, coefficients and limitations.','',
    '## Data and evaluation scope','','The data, features, 32-flow windows, observed endpoint graphs, preprocessing, train/validation/test captures and artificial test corruptions match the preceding study exactly. There are 799,777 verified-label flows among 19,976,700 original CTU-13 rows (4.00%): 442,084 training, 214,880 validation and 142,813 test. Only From-Normal and From-Botnet labels enter; background and other uncertain labels remain excluded. Windows/graphs are formed after that filtering.','',
    'All training flows complete before validation begins, and validation completes before testing begins. Hosts may recur, so this is capture/time-disjoint, not host-disjoint. Capture 12 contains NSIS.ay, absent from training; two test captures contain Rbot. These are 2011 university recordings with controlled botnet executions, not arbitrary operational incidents. The target is retrospective completed-flow classification, not measured course-of-action success.','',
    'Nine earlier baselines are imported without changing their predictions, summaries or checkpoints. Four new methods are trained from scratch with the same five seeds and maximum 40 supervised epochs, validation patience 8, AdamW learning rate 0.001 and batch 256. EDGE_GMAE, CONS_GMAE and CONS_PRE receive five pretraining epochs; EDGE_DENSE receives none. All have the same trainable parameter count as GMAE, but consistency introduces additional forward passes and is not equal-FLOP training.','',
    '## Two clean-data contrasts against GMAE','','Each primary difference gives three test captures equal weight within each seed. Positive differences favor the new variant. The wider interval is the 97.5% marginal percentile interval for a nominal two-comparison Bonferroni adjustment.','',
    '| Variant | Clean AP | GMAE AP | Difference | 95% interval | 97.5% interval |','|---|---:|---:|---:|---|---|']
    for method,p in result['primary'].items():
        lo,hi=p['percentile_95_ci'];blo,bhi=p['bonferroni_97_5_ci']
        lines.append(f'| {method} | {p["means"][method]:.5f} | {p["means"]["GMAE"]:.5f} | {p["estimate"]:+.5f} | [{lo:+.5f}, {hi:+.5f}] | [{blo:+.5f}, {bhi:+.5f}] |')
    for method,p in result['primary'].items():
        lo,hi=p['bonferroni_97_5_ci']
        conclusion=('has a positive adjusted interval on these fixed captures; this is exploratory evidence, not confirmation' if lo>0 else
        'has a negative adjusted interval, favoring GMAE on these fixed captures' if hi<0 else
        'has an adjusted interval containing zero; neither improvement nor equivalence is established')
        lines+=['',f'{method} {conclusion}.']
    lines+=['',f'The paired bootstrap uses {design["bootstrap_repetitions"]:,} accepted shared draws and rejects {design["rejected_zero_positive_draws"]} draws without positives. Captures 10/11/12 contain {design["blocks_per_capture"]["10"]}/{design["blocks_per_capture"]["11"]}/{design["blocks_per_capture"]["12"]} contiguous 64-window blocks (up to 2,048 flows each). It crosses training seeds with within-capture blocks, not independent individual flows. Captures themselves are fixed and not population-resampled; the last two block populations are small.','',
    '## All clean capture-macro comparisons','','| Method | Attack AP | ROC-AUC | Brier | Balanced accuracy | Recall | FPR |','|---|---:|---:|---:|---:|---:|---:|']
    for method in METHODS:lines.append('| '+method+' | '+' | '.join(f'{mean(rows,method,k):.5f}' for k in ['ap','auc','brier','balanced_accuracy','recall','fpr'])+' |')
    best=max(METHODS,key=lambda m:mean(rows,m,'ap'))
    lines+=['',f'The numerically largest clean mean AP is {best} ({mean(rows,best,"ap"):.5f}). This descriptive ranking is not an additional multiplicity-adjusted superiority test. Recall/FPR use clean-validation F1 thresholds fixed across every test condition. AP depends on the selected class prevalence; fixed thresholds face substantial prior shift.','',
    '## Capture-specific paired differences','','| Capture | EDGE minus GMAE | CONS minus GMAE |','|---|---:|---:|']
    for c in SCENARIOS:lines.append(f'| {c} | '+ ' | '.join(f'{result["primary"][m]["per_capture_effects"][str(c)]:+.5f}' for m in VARIANTS)+' |')
    lines+=['','## Artificial stress and controls','','| Condition | GMAE | DENSE | EDGE | EDGE no pretrain | CONS | CONS pretrain only |','|---|---:|---:|---:|---:|---:|---:|']
    for c in CONDITIONS:lines.append('| '+c+' | '+' | '.join(f'{mean(rows,m,"ap",c):.5f}' for m in ['GMAE','DENSE']+NEW)+' |')
    lines+=['',f'EDGE_GMAE improves the clean mean AP from {mean(rows,"GMAE","ap"):.5f} to {mean(rows,"EDGE_GMAE","ap"):.5f}, while the no-pretraining EDGE_DENSE control reaches {mean(rows,"EDGE_DENSE","ap"):.5f}. The gated family\'s clean gain cannot be attributed to masked pretraining alone. At 50% missingness, EDGE_GMAE reaches {mean(rows,"EDGE_GMAE","ap","missing50"):.5f}, compared with GMAE {mean(rows,"GMAE","ap","missing50"):.5f} and EDGE_DENSE {mean(rows,"EDGE_DENSE","ap","missing50"):.5f}; this is a descriptive severity comparison, not another primary test.', '',f'CONS_GMAE reaches clean AP {mean(rows,"CONS_GMAE","ap"):.5f}, compared with CONS_PRE {mean(rows,"CONS_PRE","ap"):.5f}. The small numerical increment does not by itself establish a benefit from the extra supervised consistency term.']
    lines+=['','Missing20/Missing50 and Spoof20/Spoof40 are artificial feature removal or training-normal-centroid replacement on recorded data. They are not observed adversarial deception. The endpoint graph is retained during corruption; inference-time EDGE reweights its existing edges. Matching the representation and objective does not guarantee that feature similarity equals class similarity.','',
    'EDGE_GMAE versus EDGE_DENSE measures the pretraining contribution within the gated architecture; EDGE_DENSE versus DENSE changes the gate at no pretraining. CONS_PRE versus GMAE changes both reconstruction-view count and latent alignment, so it does not isolate the cosine term; CONS_GMAE versus CONS_PRE adds supervised consistency. These secondary contrasts use descriptive seed-t intervals conditional on the same test pool.','',
    '## Recorded compute and evidence audit','','| Method | Trainable parameters / fitted NB statistics | Mean train seconds | Warm CUDA window ms | Incremental VRAM MiB |','|---|---:|---:|---:|---:|']
    for m in METHODS:lines.append('| '+m+' | '+f'{mean(rows,m,"parameters"):.0f}'+' | '+' | '.join(f'{mean(rows,m,k):.3f}' for k in ['train_seconds','latency_ms','peak_vram_mb'])+' |')
    gpu=json.loads((OUT/'gpu_audit.json').read_text());timing={r['method']:r for r in gpu['timing_rows']}
    lines+=['','The separate common-session audit holds the timing window and measurement loop fixed across all thirteen methods. For the primary variants:','','| Method | Host-wall median ms | Host-wall p95 ms | CUDA median ms |','|---|---:|---:|---:|']
    for method in ['GMAE']+VARIANTS:
        t=timing[method];lines.append(f'| {method} | {t["wall_median_ms"]:.3f} | {t["wall_p95_ms"]:.3f} | {t["cuda_median_ms"]:.3f} |')
    lines+=['',f'EDGE_GMAE lowers descriptive clean FPR from {mean(rows,"GMAE","fpr"):.5f} to {mean(rows,"EDGE_GMAE","fpr"):.5f}, with recall {mean(rows,"EDGE_GMAE","recall"):.5f} versus {mean(rows,"GMAE","recall"):.5f}. It adds pairwise aggregation work: bounded wall median {timing["EDGE_GMAE"]["wall_median_ms"]:.3f} ms versus {timing["GMAE"]["wall_median_ms"]:.3f} ms. CONS_GMAE retains the original inference architecture and records {timing["CONS_GMAE"]["wall_median_ms"]:.3f} ms. These are observed operating-point/cost tradeoffs, not additional adjusted hypotheses.']
    lines+=['','Baseline timings were recorded in the prior run and are imported; new-method timings come from the follow-up run. They are measurements on the same Ada device, not simultaneous randomized hardware trials. Training totals include pretraining and supervision. The separate common-session GPU audit remeasures all thirteen inference architectures and is used in the resource figures. Its bounded host-wall path includes decoded-input transfers, normalization, graph construction and prediction return; neither timing is full operational detection latency. NB repeats a deterministic closed-form fit in five seed slots, not five stochastic learning outcomes.','',
    f'The audit checked {result["audit"]["summary_records"]} summary records, {result["audit"]["raw_archives_verified"]} raw archives, {result["audit"]["raw_summary_metric_checks"]} recomputed metrics and {result["audit"]["checkpoint_hashes_verified"]} checkpoint hashes. Imported baseline records remain unchanged. Reliability bins contain all {cal["predictions_total"]:,} model-flow predictions. {cal["out_of_range_predictions"]} raw scores exceed a probability boundary within the 1e-6 roundoff allowance (maximum excess {cal["maximum_bound_excess"]:.3g}); only reliability binning/display clips them to [0,1]. Raw rankings and metrics remain as executed.','',
    'Raw data, predictions, checkpoints, logs and PDF review renders remain ignored by Git. Code, concise measurements, hashes, four main figures and a three-page supplement with two extra figures are retained. This extension does not establish operational planning, actual deception handling, network compression, unseen-organization performance or complete mission-state regeneration.','',
    '## Sources and artifacts','','- [GraphMAE](https://arxiv.org/abs/2205.10803), Hou et al. (2022); [GraphMAE2](https://arxiv.org/abs/2304.04779), Hou et al. (2023). The variants are adaptations, not reproductions or claims to have originated their general ingredients.',
    '- [Official CTU-13 dataset](https://www.stratosphereips.org/datasets-ctu13); García et al. (2014), [An empirical comparison of botnet detection methods](https://doi.org/10.1016/j.cose.2014.05.011).',
    '- [Four main figures](../figures/real_gmae/index.html); [three-page supplement](../supplemental/real_gmae/Supplement_real_gmae.pdf).',
    '- `results/real_gmae/manifest.json` records source/input/checkpoint hashes and baseline origins; `comparisons.json` records statistics and uncertainty scope; `calibration_audit.json` records every bin count.','']
    (ROOT/'docs/RESULTS_REAL_GMAE.md').write_text('\n'.join(lines))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repetitions',type=int,default=1000);args=ap.parse_args()
    if args.repetitions!=1000:raise ValueError('The frozen exploratory analysis uses 1000 bootstrap draws')
    path=OUT/'manifest.json'
    if not path.is_file():raise RuntimeError('Wait for the completed GraphMAE variant run')
    manifest=json.loads(path.read_text());rows=load_rows()
    gpu_path=OUT/'gpu_audit.json';gpu=json.loads(gpu_path.read_text())
    if not gpu['passed'] or gpu['manifest_sha256']!=sha(path):raise ValueError('Completed current GPUaudit required before final reporting')
    checks,raw,hashes,calibration,cases=audit(manifest,rows)
    write_csv(OUT/'calibration.csv',calibration)
    cal=dict(manifest_sha256=sha(path),summary_sha256=sha(OUT/'summary.csv'),
        raw_hash_catalogue_sha256=hashlib.sha256(json.dumps(manifest['raw_hashes'],sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        analysis_source_sha256=sha(__file__),calibration_sha256=sha(OUT/'calibration.csv'),raw_archives=len(cases),calibration_rows=len(calibration),
        predictions_total=sum(r['n'] for r in cases),binned_total=sum(r['binned'] for r in cases),
        out_of_range_predictions=sum(r['out_of_range'] for r in cases),maximum_bound_excess=max(r['max_excess'] for r in cases),
        tolerance=1e-6,policy='Preserve raw scores and ranking metrics. Clip only reliability bin membership/display; require each flow exactly once.',per_case=cases)
    (OUT/'calibration_audit.json').write_text(json.dumps(cal,indent=2)+'\n')
    primary,design=bootstrap(raw,args.repetitions)
    result=dict(primary=primary,bootstrap=design,contrasts=contrasts(rows),audit=checks,
        calibration_audit={k:v for k,v in cal.items() if k!='per_case'},seeds=SEEDS,test_scenarios=SCENARIOS,conditions=CONDITIONS,
        raw_input_sha256=hashes,manifest_sha256=sha(path),summary_sha256=sha(OUT/'summary.csv'),
        calibration_sha256=sha(OUT/'calibration.csv'),calibration_audit_sha256=sha(OUT/'calibration_audit.json'),
        analysis_source_sha256=sha(__file__),analysis_dependency_hashes={'scripts/analyze_real.py':sha(ROOT/'scripts/analyze_real.py')},
        gpu_audit_sha256=sha(gpu_path),exploratory_test_reuse=True)
    (OUT/'comparisons.json').write_text(json.dumps(result,indent=2)+'\n');report(rows,result,manifest)
    print(json.dumps(dict(primary=primary,bootstrap=design,audit=checks),indent=2))


if __name__=='__main__':main()
