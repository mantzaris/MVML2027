"""Frozen object-category evaluation with paired source-cluster uncertainty."""
from __future__ import annotations
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
from scipy.special import softmax
from sklearn.metrics import f1_score
from underwater_reasoning.identification import (prototype_evidence, review_score, identification_metrics,
                                                 confusion, macro_f1)
from underwater_reasoning.io import atomic_json, file_hash
from train_identification import inputs, calibrated, HEADS

ROOT = Path(__file__).resolve().parents[1]
LABELS = ['background', 'cargo', 'passenger', 'tanker', 'tug_tow']


def check_freeze():
    frozen = json.loads((ROOT/'protocol/identification_freeze.json').read_text())
    for path, expected in frozen['files'].items():
        if file_hash(ROOT/path) != expected:
            raise ValueError('Freeze mismatch: '+path)
    return frozen


def bootstrap(rows, labels, probabilities, comparator, repetitions, cluster_field='source_group'):
    groups = sorted({r[cluster_field] for r in rows})
    members = [np.array([i for i, r in enumerate(rows) if r[cluster_field]==g]) for g in groups]
    rng = np.random.default_rng(1729)
    selected = rng.integers(0, len(groups), (repetitions, len(groups)))
    values, intervals = {}, {}
    for method, p in probabilities.items():
        prediction = p.argmax(1)
        matrices = np.array([confusion(labels[i], prediction[i]) for i in members])
        sums = matrices[selected].sum(1)
        diagonal = np.diagonal(sums, axis1=1, axis2=2)
        denominator = sums.sum(1)+sums.sum(2)
        f1 = np.divide(2*diagonal, denominator, out=np.zeros_like(diagonal, dtype=float), where=denominator>0).mean(1)
        accuracy = diagonal.sum(1)/sums.sum((1, 2))
        values[method] = {'macro_f1':f1, 'accuracy':accuracy}
        intervals[method] = {key+'_ci95':np.quantile(v, [.025, .975]).tolist() for key, v in values[method].items()}
    contrast = {}
    for metric in ['macro_f1', 'accuracy']:
        difference = values['fusion'][metric]-values[comparator][metric]
        left = identification_metrics(labels, probabilities['fusion'])[metric]
        right = identification_metrics(labels, probabilities[comparator])[metric]
        contrast[metric] = {'difference':left-right, 'ci95':np.quantile(difference, [.025, .975]).tolist()}
    contrast.update(comparator=comparator, clusters=len(groups), cluster_field=cluster_field, repetitions=repetitions)
    return intervals, contrast


def curves(labels, p, score):
    correct = p.argmax(1)==labels
    order = np.argsort(-score, kind='stable')
    result = []
    for coverage in np.linspace(.1, 1., 10):
        k = int(np.ceil(coverage*len(labels)))
        selected = order[:k]
        result.append({'target':float(coverage), 'issued':k, 'coverage':k/len(labels),
                       'conditional_accuracy':float(correct[selected].mean()),
                       'correct_per_100':float(100*correct[selected].sum()/len(labels)),
                       'wrong_per_100':float(100*(~correct[selected]).sum()/len(labels))})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predict-only', action='store_true')
    args = parser.parse_args()
    frozen = check_freeze()
    config = json.loads((ROOT/'configs/identification_study.json').read_text())
    rows = json.loads((ROOT/'data/private/identification_evaluation_manifest.json').read_text())['records']
    labels = np.array([r['label_index'] for r in rows])
    model = joblib.load(ROOT/'checkpoints/identification_fusion.joblib')
    seeds = [np.load(ROOT/f'data/derived/identification_seed{s}.npz') for s in [17,29,43]]
    for data in seeds:
        assert list(data['ids']) == [r['id'] for r in rows]
    physical = np.load(ROOT/'data/derived/identification_test_physical.npz')
    assert list(physical['ids']) == [r['id'] for r in rows]
    anchor = softmax(seeds[0]['logits']/float(seeds[0]['temperature']), axis=1)
    exemplar, proto_probs = prototype_evidence(seeds[0]['embeddings'], model['prototypes'], model['prototype_classes'],
        [r['source_group'] for r in rows], model['prototype_groups'])
    acoustic = np.c_[seeds[0]['pca'], np.log(np.clip(anchor, 1e-8, 1))]
    xx = inputs(acoustic, physical['features'], exemplar)
    predictions = {name:calibrated(model['heads'][name], xx[name]) for name in HEADS}
    predictions.update(anchor=anchor, prototype_vote=proto_probs,
                       ensemble=np.mean([softmax(z['logits']/float(z['temperature']), axis=1) for z in seeds], axis=0))
    predictions['fusion_maxprob'] = predictions['fusion']
    scores = {name:p.max(1) for name,p in predictions.items()}
    scores['fusion'] = review_score(predictions['fusion'], predictions['physical_only'], proto_probs, model['penalty'])
    raw_dir = ROOT/'results/raw/identification'
    raw_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(raw_dir/'evaluation_predictions.npz', ids=np.array([r['id'] for r in rows]), **predictions,
                        **{'score_'+name:s for name,s in scores.items()})
    if args.predict_only:
        print('Fresh predictions saved; no outcome summaries requested', flush=True)
        return
    conditions = {}
    comparator = model['primary_comparator']
    for condition in ['clean']+config['stress_conditions']:
        use = np.array([r['variant']==condition for r in rows])
        selected = [r for r, keep in zip(rows, use) if keep]
        metrics = {name:identification_metrics(labels[use], p[use], scores[name][use], model['thresholds'][name],
                                               model['binary_thresholds'][name]) for name,p in predictions.items()}
        intervals, contrast = bootstrap(selected, labels[use], {name:p[use] for name,p in predictions.items()},
                                        comparator, config['bootstrap_repetitions'])
        for name in metrics:
            metrics[name].update(intervals[name])
        conditions[condition] = {'methods':metrics, 'primary_contrast':contrast,
                                'curves':{name:curves(labels[use], p[use], scores[name][use]) for name,p in predictions.items()}}
    clean = np.array([r['variant']=='clean' for r in rows])
    selected = [r for r, use in zip(rows, clean) if use]
    _, day_contrast = bootstrap(selected, labels[clean], {name:p[clean] for name,p in predictions.items()},
                                comparator, config['bootstrap_repetitions'], 'date')
    corrupted_exemplar = np.concatenate([np.roll(exemplar[clean, i:i+5], 1, axis=1) for i in [0,5,10]], axis=1)
    corrupt_x = inputs(acoustic[clean], physical['features'][clean], corrupted_exemplar)['fusion']
    corrupt_p = calibrated(model['heads']['fusion'], corrupt_x)
    correct = predictions['fusion'][clean].argmax(1)==labels[clean]
    corrupt_correct = corrupt_p.argmax(1)==labels[clean]
    corruption = {'intervention':'cyclic permutation of the five prototype class channels',
                  'metrics':identification_metrics(labels[clean], corrupt_p),
                  'changed_labels':int((corrupt_p.argmax(1)!=predictions['fusion'][clean].argmax(1)).sum()),
                  'correct_to_wrong':int((correct & ~corrupt_correct).sum()),
                  'wrong_to_correct':int((~correct & corrupt_correct).sum())}
    omni = {}
    for variant in ['raw', 'acoustic_prior', 'fused_evidence']:
        records = []
        for row in selected:
            path = ROOT/f'results/raw/identification/omni/test/{variant}/{row["id"]}.json'
            if path.exists():
                records.append(json.loads(path.read_text()))
        if len(records)!=len(selected):
            omni[variant] = {'requested':len(selected), 'saved':len(records), 'complete':False}
            continue
        predicted = np.array([LABELS.index(r['parsed']['label']) if r['parsed'] and r['parsed']['label'] in LABELS else -1 for r in records])
        truth = labels[clean]
        issued = predicted>=0
        correct_omni = predicted==truth
        original_method = 'anchor' if variant=='acoustic_prior' else 'fusion'
        original_correct = predictions[original_method][clean].argmax(1)==truth
        counts = np.zeros((5,6), dtype=int)
        np.add.at(counts, (truth, np.where(issued,predicted,5)), 1)
        omni[variant] = {'requested':len(selected), 'saved':len(records), 'complete':True,
            'valid':sum(r['parsed'] is not None for r in records), 'accuracy':float(correct_omni.mean()),
            'macro_f1':float(f1_score(truth,predicted,labels=range(5),average='macro',zero_division=0)),
            'per_class_recall':[float(correct_omni[truth==i].mean()) for i in range(5)],
            'confusion_matrix_with_uncertain_column':counts.tolist(), 'issued':int(issued.sum()),
            'correct_issued':int(correct_omni.sum()), 'wrong_issued':int((issued & ~correct_omni).sum()),
            'referred':int((~issued).sum()), 'conditional_accuracy':float(correct_omni[issued].mean()) if issued.any() else None,
            'prior_method_for_override_counts':original_method if variant!='raw' else None,
            'correct_prior_to_wrong_or_withheld':int((original_correct & ~correct_omni).sum()) if variant!='raw' else None,
            'wrong_prior_corrected':int((~original_correct & correct_omni).sum()) if variant!='raw' else None}
    contrast = conditions['clean']['primary_contrast']['macro_f1']
    summary = {'recorded_utc':datetime.now(timezone.utc).isoformat(), 'fresh_clean_clips':int(clean.sum()),
               'source_groups':len({r['source_group'] for r in selected}), 'utc_days':len({r['date'] for r in selected}),
               'labels':LABELS, 'class_counts':np.bincount(labels[clean],minlength=5).tolist(),
               'primary_comparator':comparator, 'primary_success':contrast['difference']>=.03 and contrast['ci95'][0]>0,
               'conditions':conditions, 'utc_day_sensitivity':day_contrast, 'prototype_intervention':corruption,
               'omni':omni, 'frozen_before_new_test_inference':True, 'freeze_utc':frozen['frozen_utc'],
               'model_sha256':file_hash(ROOT/'checkpoints/identification_fusion.joblib'),
               'raw_predictions_sha256':file_hash(raw_dir/'evaluation_predictions.npz'),
               'interpretation_limits':['same collection and prior test dates','no individual hull identification',
                   'dataset category labels, not human operator judgments','artificial stress, not continuous detection',
                   'review fraction is not measured human time savings','unknown parent-recording and deployment IDs']}
    atomic_json(ROOT/'results/tables/identification_results.json', summary)
    print(json.dumps({'primary_success':summary['primary_success'],'primary_comparator':comparator,
                      'clean_macro_f1_contrast':contrast,'clean_accuracy':{name:v['accuracy'] for name,v in conditions['clean']['methods'].items()},
                      'omni_accuracy':{name:v.get('accuracy') for name,v in omni.items()}}, indent=2))


if __name__ == '__main__':
    main()
