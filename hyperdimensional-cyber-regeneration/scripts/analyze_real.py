#!/usr/bin/env python3
"""Audit CTU-13 predictions and estimate paired held-out-capture comparisons.

No model inference is performed here. CPU statistics consume recorded CUDA
predictions. The primary interval crosses training seeds and contiguous flow
blocks within each fixed held-out capture; captures are not IID resampled.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/real'
SEEDS = [11, 23, 37, 51, 71]
SCENARIOS = [10, 11, 12]
METHODS = ['NB', 'LOGIT', 'DAE', 'DENSE', 'GMAE', 'HDC', 'SCR_FLOW', 'CTX_FLOW', 'LOCAL_FLOW']
CONDITIONS = ['clean', 'missing20', 'missing50', 'spoof20', 'spoof40']
LABELS = {'NB': 'Gaussian naive Bayes', 'LOGIT': 'Logistic classifier', 'DAE': 'Denoising autoencoder',
          'DENSE': 'Dense graph classifier', 'GMAE': 'Masked-pretrained graph classifier',
          'HDC': 'HDC-augmented masked graph classifier', 'SCR_FLOW': 'Shared-context mixture',
          'CTX_FLOW': 'Single expert with context', 'LOCAL_FLOW': 'Single expert without context'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_rows(path=OUT/'summary.csv'):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def select(rows, **criteria):
    return [r for r in rows if all(str(r.get(k)) == str(v) for k, v in criteria.items())]


def ranked(y, q):
    """Stable descending ranking and ends of exact prediction-tie groups."""
    y, q = np.asarray(y, dtype=np.float64), np.asarray(q, dtype=np.float64)
    if y.ndim != 1 or q.shape != y.shape or not len(y):
        raise ValueError('Probability and label arrays must be nonempty vectors')
    order = np.argsort(-q, kind='stable')
    ends = np.r_[np.flatnonzero(np.diff(q[order]) != 0), len(q)-1]
    return order, ends


def average_precision(y, q, weight=None):
    order, ends = ranked(y, q)
    weight = np.ones(len(y), dtype=np.float64) if weight is None else np.asarray(weight, dtype=np.float64)
    positive = weight * np.asarray(y)
    total = positive.sum()
    if total <= 0:
        return float('nan')
    tp = np.cumsum(positive[order])[ends]
    count = np.cumsum(weight[order])[ends]
    precision = np.divide(tp, count, out=np.zeros_like(tp), where=count > 0)
    return float((np.diff(np.r_[0., tp]) * precision).sum() / total)


def roc_auc(y, q):
    order, ends = ranked(y, q)
    truth = np.asarray(y, dtype=np.float64)
    positives, negatives = truth.sum(), len(truth)-truth.sum()
    if min(positives, negatives) <= 0:
        return float('nan')
    tp = np.cumsum(truth[order])[ends]
    fp = ends + 1 - tp
    return float(np.trapezoid(np.r_[0., tp/positives], np.r_[0., fp/negatives]))


def metrics(y, q, threshold):
    y, q = np.asarray(y, dtype=bool), np.asarray(q, dtype=float)
    decision = q >= threshold
    tp = int((decision & y).sum()); fp = int((decision & ~y).sum())
    fn = int((~decision & y).sum()); tn = int((~decision & ~y).sum())
    recall = tp / (tp+fn) if tp+fn else float('nan')
    fpr = fp / (fp+tn) if fp+tn else float('nan')
    return {'ap': average_precision(y, q), 'auc': roc_auc(y, q), 'brier': float(((q-y)**2).mean()),
            'balanced_accuracy': (recall+1-fpr)/2,
            'f1': 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0., 'recall': recall, 'fpr': fpr}


def raw_path(seed, method, condition, scenario):
    return ROOT / 'results/runs/real' / str(seed) / f'{method}_{condition}_{scenario}.npz'


def audit(manifest, rows):
    if manifest.get('settings', {}).get('pilot'):
        raise ValueError('Pilot results cannot supply final evidence')
    if sorted(manifest.get('methods', [])) != sorted(METHODS):
        raise ValueError('Expected all nine frozen methods')
    if sorted({int(r['seed']) for r in rows}) != SEEDS or len(rows) != 675:
        raise ValueError('Expected 675 records: nine methods, five seeds, five conditions, three captures')
    keys = [(int(r['seed']), r['method'], r['condition'], int(r['scenario'])) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError('Repeated seed/method/condition/capture records')
    expected = {(s, m, c, d) for s in SEEDS for m in METHODS for c in CONDITIONS for d in SCENARIOS}
    if set(keys) != expected:
        raise ValueError('The record catalogue differs from the frozen design')
    sources = manifest.get('source_hashes', {})
    if not sources:
        raise ValueError('Scientific source hashes are required')
    for path, digest in sources.items():
        if sha(ROOT/path) != digest:
            raise ValueError(f'Scientific source changed: {path}')
    data_path=OUT/'ctu13_data_manifest.json'
    if sha(data_path)!=manifest.get('data_manifest_sha256'):
        raise ValueError('Dataset preparation manifest changed')
    dataset=json.loads(data_path.read_text())
    temporal=dataset.get('temporal_split_check',{})
    if not dataset.get('complete') or not temporal.get('passed'):
        raise ValueError('Complete data and temporal split audit required')
    if not (temporal['latest_train_completion']<temporal['earliest_validation_start'] and
            temporal['latest_validation_completion']<temporal['earliest_test_start']):
        raise ValueError('Training/validation flow completion overlaps a later split')
    prepared_checked=0
    for record in dataset['files']:
        if sha(ROOT/record['prepared_path'])!=record['prepared_sha256']:
            raise ValueError('Prepared capture changed: '+record['prepared_path'])
        prepared_checked+=1
    expected_counts={r['scenario']:r['rows_retained'] for r in dataset['files']}
    reference = {}; raw_hashes = {}; primary_raw = {}; metric_checks = 0
    calibration=[];calibration_cases=[];roundoff_count=0;max_probability_excess=0.
    for row in rows:
        seed, method, condition, scenario = int(row['seed']), row['method'], row['condition'], int(row['scenario'])
        path = raw_path(seed, method, condition, scenario)
        relative=str(path.relative_to(ROOT));raw_hashes[relative] = sha(path)
        if manifest.get('raw_hashes',{}).get(relative)!=raw_hashes[relative]:
            raise ValueError('Raw archive differs from executed manifest: '+relative)
        with np.load(path, allow_pickle=False) as archive:
            y, q = archive['labels'], archive['prob']
            block, window, row_id = archive['block'], archive['window'], archive['row_id']
            if y.ndim != 1 or q.shape != y.shape or block.shape != y.shape or window.shape != y.shape:
                raise ValueError(f'Invalid raw shapes: {path}')
            if not np.isin(y, [0, 1]).all() or not np.isfinite(q).all() or (q < -1e-6).any() or (q > 1+1e-6).any():
                raise ValueError(f'Invalid labels/probabilities: {path}')
            out_of_range=int(((q<0)|(q>1)).sum());excess=max(0.,float(np.maximum(-q,q-1).max()))
            roundoff_count+=out_of_range;max_probability_excess=max(max_probability_excess,excess)
            # Convex mixtures may exceed one by an FP32 ulp. Preserve raw metric
            # scores; clip only reliability-display probabilities/bin membership.
            display=np.clip(q,0.,1.)
            for j in range(10):
                keep=(display>=j/10)&((display<(j+1)/10) if j<9 else (display<=1))
                calibration.append(dict(seed=seed,method=method,condition=condition,scenario=scenario,bin=j,
                    count=int(keep.sum()),confidence=float(display[keep].mean()) if keep.any() else (j+.5)/10,
                    accuracy=float(y[keep].mean()) if keep.any() else float('nan')))
            binned=sum(r['count'] for r in calibration[-10:])
            if binned!=len(y):raise ValueError('Calibration bins lost observations')
            calibration_cases.append(dict(seed=seed,method=method,condition=condition,scenario=scenario,n=len(y),binned=binned,out_of_range=out_of_range,max_excess=excess))
            if len(y) != int(row['n']) or len(y)!=expected_counts[scenario] or int(y.sum()) != int(row['positives']):
                raise ValueError(f'Raw count mismatch: {path}')
            if len(np.unique(row_id)) != len(y):
                raise ValueError(f'Duplicate original source rows: {path}')
            ref = (y.copy(), block.copy(), window.copy(), row_id.copy())
            if scenario not in reference:
                reference[scenario] = ref
            elif any(not np.array_equal(a, b) for a, b in zip(reference[scenario], ref)):
                raise ValueError(f'Flow labels or temporal blocks are not paired: {path}')
            if not np.array_equal(block, window//64):
                raise ValueError(f'Expected 64 consecutive 32-flow windows per block: {path}')
            observed = metrics(y, q, float(row['threshold']))
            for key, value in observed.items():
                if not np.isclose(value, float(row[key]), atol=2e-6, rtol=3e-6):
                    raise ValueError(f'Raw/summary {key} mismatch: {path}: {value} vs {row[key]}')
                metric_checks += 1
            if condition == 'clean' and method in ['SCR_FLOW', 'GMAE']:
                primary_raw[(seed, method, scenario)] = {'y': y.copy(), 'q': q.copy(), 'block': block.copy()}
    checks = {'probability_bound_tolerance':1e-6,'roundoff_bound_exceedance_predictions':roundoff_count,'maximum_probability_bound_excess':max_probability_excess,'calibration_policy':'Retain raw scores for AP/AUC/Brier parity; clip to[0,1] only for reliability display and bin membership. Regenerated from raw to include every prediction.', 'summary_records': len(rows), 'source_hashes_verified': len(sources),'prepared_capture_hashes_verified':prepared_checked,'completion_time_split_verified':True,
              'raw_archives_verified': len(raw_hashes), 'raw_summary_metric_checks': metric_checks,
              'capture_counts': {str(k): {'flows': len(v[0]), 'attack_flows': int(v[0].sum()),
                'blocks': len(np.unique(v[1])), 'windows': len(np.unique(v[2]))} for k, v in reference.items()}}
    checkpoints = manifest.get('checkpoints', [])
    if len(checkpoints)!=45 or {(int(r['seed']),r['method']) for r in checkpoints}!={(s,m) for s in SEEDS for m in METHODS}:
        raise ValueError('Expected forty neural and five repeated conventional checkpoint records')
    checked = 0
    for record in checkpoints:
        method, seed = record['method'], record['seed']
        path = ROOT / 'results/checkpoints/real' / f'{seed}_{method}.pt'
        if not path.is_file() or sha(path) != record['checkpoint_sha256']:
            raise ValueError(f'Checkpoint changed or absent: {path}')
        checked += 1
    checks['checkpoint_hashes_verified'] = checked
    return checks, primary_raw, raw_hashes, calibration, calibration_cases


class WeightedAP:
    """Reuse score ordering for exact block-weighted, tie-aware AP."""
    def __init__(self, labels, scores, blocks):
        order, self.ends = ranked(labels, scores)
        self.y = np.asarray(labels, dtype=np.float64)[order]
        self.block = np.asarray(blocks, dtype=np.int64)[order]
        self.positive_by_block = np.bincount(self.block, weights=self.y)

    def __call__(self, counts):
        w = counts[self.block]
        tp = np.cumsum(w*self.y)[self.ends]
        seen = np.cumsum(w)[self.ends]
        precision = np.divide(tp, seen, out=np.zeros_like(tp), where=seen > 0)
        total = tp[-1]
        if total <= 0:
            return float('nan')
        return float((np.diff(np.r_[0., tp])*precision).sum()/total)


def primary_bootstrap(raw, repetitions=1000):
    rng = np.random.default_rng(20261125)
    prepared = {key: WeightedAP(v['y'], v['q'], v['block']) for key, v in raw.items()}
    blocks = {}
    for scenario in SCENARIOS:
        sample = raw[(SEEDS[0], 'SCR_FLOW', scenario)]
        unique = np.unique(sample['block'])
        if not np.array_equal(unique, np.arange(len(unique))):
            raise ValueError('Block IDs must be dense and start at zero within each capture')
        blocks[scenario] = len(unique)
    original = {key: average_precision(v['y'], v['q']) for key, v in raw.items()}
    seed_effects = [np.mean([original[(s, 'SCR_FLOW', c)] - original[(s, 'GMAE', c)] for c in SCENARIOS]) for s in SEEDS]
    draws = []; rejected = 0; attempts = 0
    while len(draws) < repetitions:
        attempts += 1
        if attempts > repetitions*100:
            raise RuntimeError('Too many resampled captures without positives; AP interval unidentified')
        seed_counts = np.bincount(rng.integers(5, size=5), minlength=5)
        counts = {c: np.bincount(rng.integers(blocks[c], size=blocks[c]), minlength=blocks[c]) for c in SCENARIOS}
        valid = all(np.dot(counts[c], prepared[(SEEDS[0], 'SCR_FLOW', c)].positive_by_block) > 0 for c in SCENARIOS)
        if not valid:
            rejected += 1
            continue
        effect = 0.
        for i, seed in enumerate(SEEDS):
            if not seed_counts[i]:
                continue
            differences = [prepared[(seed, 'SCR_FLOW', c)](counts[c]) - prepared[(seed, 'GMAE', c)](counts[c]) for c in SCENARIOS]
            effect += seed_counts[i]*np.mean(differences)/5
        draws.append(effect)
    return {'method': 'SCR_FLOW', 'comparator': 'GMAE', 'metric': 'macro_attack_average_precision',
            'condition': 'clean', 'estimate': float(np.mean(seed_effects)), 'higher_is_better': True,
            'percentile_95_ci': np.quantile(draws, [.025, .975]).tolist(),
            'means': {m: float(np.mean([original[(s, m, c)] for s in SEEDS for c in SCENARIOS])) for m in ['SCR_FLOW', 'GMAE']},
            'seed_effects': dict(zip(map(str, SEEDS), map(float, seed_effects))),
            'per_capture_effects': {str(c): float(np.mean([original[(s, 'SCR_FLOW', c)]-original[(s, 'GMAE', c)] for s in SEEDS])) for c in SCENARIOS},
            'bootstrap_repetitions': repetitions, 'bootstrap_rng_seed': 20261125,
            'rejected_zero_positive_draws': rejected, 'blocks_per_capture': {str(k): v for k, v in blocks.items()},
            'bootstrap_design': 'Cross five training seeds with contiguous 64-window blocks independently resampled within each fixed test capture; the same block draw is shared across methods and seeds. Each capture has fixed equal weight. Recompute exact tie-aware weighted AP; jointly reject draws with no positive flow in any capture.',
            'uncertainty_scope': 'Conditional on three fixed held-out captures, fixed training data and the selected verified labels. Captures, unseen organizations and unknown background labels are not population-resampled.'}


def summary_contrasts(rows):
    contrasts = []
    for condition in CONDITIONS:
        for comparator in ['GMAE', 'HDC', 'DENSE', 'CTX_FLOW', 'LOCAL_FLOW', 'DAE', 'LOGIT', 'NB']:
            record = {'method': 'SCR_FLOW', 'comparator': comparator, 'condition': condition, 'metrics': {}}
            for metric in ['ap', 'auc', 'brier', 'balanced_accuracy', 'f1', 'recall', 'fpr']:
                values = []
                for seed in SEEDS:
                    a = select(rows, method='SCR_FLOW', seed=seed, condition=condition)
                    b = select(rows, method=comparator, seed=seed, condition=condition)
                    values.append(np.mean([float(r[metric]) for r in a])-np.mean([float(r[metric]) for r in b]))
                error = 2.776445*np.std(values, ddof=1)/np.sqrt(5)
                record['metrics'][metric] = {'estimate': float(np.mean(values)), 'seed_t_95_ci': [float(np.mean(values)-error), float(np.mean(values)+error)],
                    'seed_effects': dict(zip(map(str, SEEDS), map(float, values))), 'higher_is_better': metric not in ['brier', 'fpr']}
            contrasts.append(record)
    return contrasts


def mean(rows, method, metric, condition='clean', scenario=None):
    chosen = select(rows, method=method, condition=condition)
    if scenario is not None:
        chosen = select(chosen, scenario=scenario)
    return float(np.mean([float(r[metric]) for r in chosen]))


def report(rows, result, manifest):
    p = result['primary']; lo, hi = p['percentile_95_ci']
    dataset=json.loads((OUT/'ctu13_data_manifest.json').read_text())
    retained=sum(r['rows_retained'] for r in dataset['files']);original=sum(r['rows_total'] for r in dataset['files'])
    train=dataset['totals']['train'];val=dataset['totals']['validation'];test=dataset['totals']['test']
    conclusion = ('The conditional primary interval favors SCR_FLOW.' if lo > 0 else
                  'The conditional primary interval favors the masked-pretrained graph baseline.' if hi < 0 else
                  'The interval contains zero and establishes neither an improvement nor equivalence.')
    lines = ['# External application to captured CTU-13 traffic', '',
        'This experiment retrains explicit adaptations on real captured network-flow measurements. CTU-13 mixes normal university traffic with controlled executions of real botnets. It is not a record of arbitrary operational incidents, and its labels do not supply host readiness or course-of-action outcomes.', '',
        '## Data and task boundaries', '',
        'Only author-labeled `From-Normal` and `From-Botnet` flows enter the binary task. Background and other uncertain labels are excluded rather than treated as benign. Exclusion makes this a selected verified-label benchmark; success does not establish performance on the omitted background distribution.', '',
        'Captures 1-8 and 13 (August 10 through August 17 morning, 2011) supply training, capture 9 (August 17 midday/afternoon) supplies validation, and captures 10-12 (August 18-19) supply the held-out test. Scenario numbering is not chronological: capture 13 belongs in the earlier training period. Test captures 10/11 contain Rbot and capture 12 contains NSIS.ay, which is not represented in training. Hosts may recur, so the split is capture/time-disjoint, not host-disjoint.', '',
        f'The selected subset contains {retained:,} of {original:,} original flows ({100*retained/original:.2f}%). Training/validation/test retain {train["rows_retained"]:,}/{val["rows_retained"]:,}/{test["rows_retained"]:,} flows. Their pooled attack prevalences are {train["rows_attack"]/train["rows_retained"]:.3f}, {val["rows_attack"]/val["rows_retained"]:.3f} and {test["rows_attack"]/test["rows_retained"]:.3f}; this substantial prior shift is part of the held-out challenge. The parser verified that every retained training flow completed before validation starts, and every validation flow completed before test starts.', '',
        'Models receive fourteen derived flow measurements and fourteen observation indicators. Raw addresses, absolute timestamps, scenario IDs and labels are not input features. Addresses are used only to construct observable within-window endpoint-sharing edges. Windows and graphs are constructed after verified-label filtering; they are not an unfiltered live-stream context. Context is fixed by consecutive 32-flow windows rather than arbitrary minibatch composition. Predictions require all flow statistics in the window, including final durations/byte totals, and are retrospective classifications after those measurements become available, not causal packet-by-packet decisions.', '',
        'The clean condition uses real recorded measurements. Missing-value and feature-spoofing conditions are explicitly artificial corruptions applied to those measurements with original labels held fixed. The observed endpoint graph is retained during feature corruption. These are stress tests, not evidence of naturally occurring telemetry deception. No synthetic conjunction is presented as a measured cyber course of action.', '',
        '## Primary held-out result', '',
        f"Clean macro attack average precision: SCR_FLOW **{p['means']['SCR_FLOW']:.5f}**, GMAE **{p['means']['GMAE']:.5f}**. Paired difference: **{p['estimate']:+.5f}**, 95% interval **[{lo:+.5f}, {hi:+.5f}]**. Higher is better. {conclusion}", '',
        'The primary average gives each held-out capture equal weight before averaging training seeds. The interval crosses five training seeds and contiguous 64-window blocks within each of three fixed test captures. It recomputes tie-aware attack AP for every weighted bootstrap sample; individual flows are not treated as independent. The interval conditions on these captures and training data and does not quantify uncertainty over organizations or capture selection.', '',
        f"There were {p['bootstrap_repetitions']:,} accepted bootstrap draws and {p['rejected_zero_positive_draws']} rejected draws lacking any positive flow in a resampled capture. The fixed captures provide {p['blocks_per_capture']['10']}, {p['blocks_per_capture']['11']} and {p['blocks_per_capture']['12']} blocks respectively; the last two are small block populations.", '',
        '## Clean capture-macro comparisons', '',
        '| Method | Attack AP ↑ | ROC-AUC ↑ | Brier ↓ | Balanced accuracy ↑ | Recall ↑ | FPR ↓ |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for method in METHODS:
        lines.append('| '+LABELS[method]+' ('+method+') | '+' | '.join(f'{mean(rows,method,k):.5f}' for k in ['ap','auc','brier','balanced_accuracy','recall','fpr'])+' |')
    lines += ['', 'Threshold metrics use validation-selected thresholds, frozen before test evaluation and retained across all stress conditions. AP/AUC are threshold-free but depend on the selected data distribution; AP in particular changes with attack prevalence.', '',
        f'Within this recorded test pool, DENSE has the largest descriptive clean macro AP ({mean(rows,"DENSE","ap"):.5f}); this is not a separate population-level superiority test. Context access is useful in the matched single-expert comparison: CTX_FLOW {mean(rows,"CTX_FLOW","ap"):.5f} versus LOCAL_FLOW {mean(rows,"LOCAL_FLOW","ap"):.5f}. The conventional NB reference reaches {mean(rows,"NB","ap"):.5f}, exceeding the small DAE ({mean(rows,"DAE","ap"):.5f}) and LOGIT ({mean(rows,"LOGIT","ap"):.5f}) in this dataset. These secondary rankings are descriptive.', '',
        '## Capture-specific outcomes', '', '| Capture | Flows | Attack prevalence | SCR_FLOW AP | GMAE AP | DAE AP | LOGIT AP |', '|---|---:|---:|---:|---:|---:|---:|']
    for scenario in SCENARIOS:
        row=select(rows,seed=SEEDS[0],method='SCR_FLOW',condition='clean',scenario=scenario)[0]
        lines.append(f"| {scenario} | {int(row['n']):,} | {int(row['positives'])/int(row['n']):.4f} | "+' | '.join(f'{mean(rows,m,"ap",scenario=scenario):.5f}' for m in ['SCR_FLOW','GMAE','DAE','LOGIT'])+' |')
    lines += ['', '## Artificial corruption on captured telemetry', '', '| Condition | SCR_FLOW AP | GMAE AP | HDC AP | DAE AP | LOGIT AP |', '|---|---:|---:|---:|---:|---:|']
    for condition in CONDITIONS:
        lines.append('| '+condition+' | '+' | '.join(f'{mean(rows,m,"ap",condition):.5f}' for m in ['SCR_FLOW','GMAE','HDC','DAE','LOGIT'])+' |')
    lines += ['', f'Clean SCR_FLOW-minus-GMAE AP differences are {p["per_capture_effects"]["10"]:+.5f}, {p["per_capture_effects"]["11"]:+.5f} and {p["per_capture_effects"]["12"]:+.5f} for captures 10, 11 and 12. The largest observed deficit occurs in capture12 (NSIS.ay, absent from training). Capture identity, traffic mix and family change together, so this is not an isolated causal effect of an unseen malware family.', '',
        '## Method and compute interpretation', '',
        'SCR_FLOW retains the shared latent mixture and fixed observation-window context from the third synthetic study, adapting the output to flow attack probabilities. CTX_FLOW and LOCAL_FLOW compare single-expert context and local-only alternatives. They are newly trained adaptations, not a zero-shot transfer of the original synthetic checkpoints. DENSE/GMAE separate graph-masked pretraining; GMAE/HDC match the supervised objective and pretraining while changing input augmentation. The supplied graph expresses shared observed endpoints, not child-implies-parent logic. No conditional-neurosymbolic consistency guarantee is invented for these labels.', '',
        'LOGIT and DAE provide inexpensive neural alternatives; Gaussian naive Bayes supplies a conventional conditional-independence reference with 58 fitted statistics; it has no gradient-trained parameters. Its deterministic fit is repeated in the five seed slots for paired bookkeeping, not five stochastic training outcomes. Parameter counts, actual training effort and measured inference costs must accompany accuracy. DAE reconstructs observed flow features during denoising pretraining; there is no no-pretraining DAE control, so its comparison does not isolate a causal pretraining benefit.', '',
        '| Method | Mean training seconds | Mean CUDA window latency (ms) | Incremental peak VRAM (MiB) |', '|---|---:|---:|---:|']
    for method in METHODS:
        lines.append(f'| {method} | '+ ' | '.join(f'{mean(rows,method,k):.3f}' for k in ['train_seconds','latency_ms','peak_vram_mb'])+' |')
    lines += ['', 'GPU timing is warmed model inference for one fixed observation window; it excludes storage/download and is not an operational network latency measurement. The separate prepared-evidence wall audit records its own included operations. Models retain differing pretraining budgets and capacities.', '',
        '## Evidence audit and scope', '',
        f"The analysis checked {result['audit']['summary_records']} unique records and {result['audit']['raw_archives_verified']} raw archives, recomputed {result['audit']['raw_summary_metric_checks']} metrics, verified paired labels/window blocks, and checked recorded scientific source/checkpoint hashes. The raw archives and downloaded captures remain ignored by Git.", '',
        f'Floating-point audit: {result["audit"]["roundoff_bound_exceedance_predictions"]:,} model-flow predictions exceeded a probability boundary within the accepted 1e-6 rounding tolerance; maximum excess was {result["audit"]["maximum_probability_bound_excess"]:.3g}. Raw scores and metric calculations are preserved. Reliability bins are regenerated with display-only clipping to [0,1] so no boundary-roundoff prediction is omitted.', '',
        'This is external evidence for representation, reconstruction and classification under shifted recorded traffic. It is not a validation of multi-step cyber planning, autonomous execution, hyperdimensional wire compression or operational mission outcomes. Author labels are partly host-derived; excluded identities reduce direct leakage but do not remove all dataset-specific artifacts. Only one dataset, three test captures and five model seeds are covered. Secondary seed-t intervals are descriptive and conditional on the fixed test pool, not the primary crossed-block interval.', '',
        '## Sources and reproducibility', '',
        '- García, Grill, Stiborek and Zunino (2014), *An empirical comparison of botnet detection methods*, Computers & Security 45, 100-123. [DOI](https://doi.org/10.1016/j.cose.2014.05.011).',
        '- [Official CTU-13 description](https://www.stratosphereips.org/datasets-ctu13) and [author usage/citation FAQ](https://www.stratosphereips.org/datasets-faq).',
        '- `results/real/manifest.json` records the executed protocol, hardware, source and data/checkpoint hashes. `results/real/comparisons.json` records the analysis design, exact raw hashes and uncertainty scope.', '']
    (ROOT/'docs/RESULTS_REAL.md').write_text('\n'.join(lines))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repetitions',type=int,default=1000);args=ap.parse_args()
    if args.repetitions != 1000:
        raise ValueError('Frozen real-data analysis uses 1,000 bootstrap repetitions')
    path=OUT/'manifest.json'
    if not path.is_file():
        raise RuntimeError('Wait for a complete real-data study manifest')
    manifest=json.loads(path.read_text());rows=load_rows()
    checks, raw, hashes, calibration, calibration_cases=audit(manifest,rows)
    with (OUT/'calibration.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(calibration[0]),lineterminator='\n');writer.writeheader();writer.writerows(calibration)
    calibration_audit=dict(manifest_sha256=sha(path),summary_sha256=sha(OUT/'summary.csv'),
        raw_hash_catalogue_sha256=hashlib.sha256(json.dumps(manifest['raw_hashes'],sort_keys=True,separators=(',',':')).encode('utf-8')).hexdigest(),
        analysis_source_sha256=sha(__file__),calibration_sha256=sha(OUT/'calibration.csv'),raw_archives=len(calibration_cases),
        calibration_rows=len(calibration),predictions_total=sum(r['n'] for r in calibration_cases),binned_total=sum(r['binned'] for r in calibration_cases),
        out_of_range_predictions=checks['roundoff_bound_exceedance_predictions'],maximum_bound_excess=checks['maximum_probability_bound_excess'],
        tolerance=1e-6,policy=checks['calibration_policy'],per_case=calibration_cases)
    (OUT/'calibration_audit.json').write_text(json.dumps(calibration_audit,indent=2)+'\n')
    primary=primary_bootstrap(raw,args.repetitions)
    result={'primary':primary,'contrasts':summary_contrasts(rows),'audit':checks,'seeds':SEEDS,
            'test_scenarios':SCENARIOS,'conditions':CONDITIONS,'raw_input_sha256':hashes,
            'manifest_sha256':sha(path),'summary_sha256':sha(OUT/'summary.csv'),'calibration_sha256':sha(OUT/'calibration.csv'),'analysis_source_sha256':sha(__file__),'calibration_audit_sha256':sha(OUT/'calibration_audit.json')}
    (OUT/'comparisons.json').write_text(json.dumps(result,indent=2)+'\n')
    report(rows,result,manifest)
    print(json.dumps({'primary':primary,'audit':checks},indent=2))


if __name__=='__main__':
    main()
