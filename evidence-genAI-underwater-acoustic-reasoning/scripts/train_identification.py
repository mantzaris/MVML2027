"""Fit the identification follow-up exclusively on original non-test partitions."""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, softmax
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from underwater_reasoning.identification import (source_weights, make_prototypes, prototype_evidence,
    review_score, coverage_threshold, identification_metrics)
from underwater_reasoning.io import atomic_json, file_hash

ROOT = Path(__file__).resolve().parents[1]
HEADS = ['acoustic_refit', 'physical_only', 'no_physical', 'no_prototypes', 'fusion']


def inputs(acoustic, physical, exemplar):
    return {'acoustic_refit': acoustic, 'physical_only': physical,
            'no_physical': np.c_[acoustic, exemplar], 'no_prototypes': np.c_[acoustic, physical],
            'fusion': np.c_[acoustic, physical, exemplar]}


def calibrated(head, x):
    logits = head['model'].decision_function(head['scaler'].transform(x))
    return softmax(logits/head['temperature'], axis=1)


def main():
    if (ROOT/'protocol/identification_freeze.json').exists():
        raise ValueError('Follow-up is frozen; refusing to retrain')
    config = json.loads((ROOT/'configs/identification_study.json').read_text())
    manifest = json.loads((ROOT/'data/private/experiment_manifest.json').read_text())
    rows = manifest['records']
    all_data = np.load(ROOT/'data/derived/classifier_seed17.npz')
    assert list(all_data['ids']) == [r['id'] for r in rows]
    keep = np.array([r['partition'] in ['policy_fit', 'development', 'calibration'] for r in rows])
    bank = np.array([r['partition'] == 'acoustic_fit' for r in rows])
    selected = [r for r, use in zip(rows, keep) if use]
    labels = np.array([r['label_index'] for r in selected])
    fit = np.array([r['partition'] == 'policy_fit' for r in selected])
    dev = np.array([r['partition'] == 'development' for r in selected])
    cal = np.array([r['partition'] == 'calibration' for r in selected])
    physical_data = np.load(ROOT/'data/derived/identification_fitting_physical.npz')
    assert list(physical_data['ids']) == [r['id'] for r in selected]
    prototypes, classes, bank_groups = make_prototypes(all_data['embeddings'][bank],
        [r['label_index'] for r, use in zip(rows, bank) if use],
        [r['source_group'] for r, use in zip(rows, bank) if use])
    exemplar, prototype_probabilities = prototype_evidence(all_data['embeddings'][keep], prototypes, classes,
        [r['source_group'] for r in selected], bank_groups)
    probs = softmax(all_data['logits'][keep]/float(all_data['temperature']), axis=1)
    acoustic = np.c_[all_data['pca'][keep], np.log(np.clip(probs, 1e-8, 1))]
    x = inputs(acoustic, physical_data['features'], exemplar)
    weights = source_weights(labels[fit], [r['source_group'] for r, use in zip(selected, fit) if use])
    heads, search, predictions = {}, {}, {'anchor': probs, 'prototype_vote': prototype_probabilities}
    ensemble = []
    for seed in [17, 29, 43]:
        data = np.load(ROOT/f'data/derived/classifier_seed{seed}.npz')
        assert list(data['ids']) == [r['id'] for r in rows]
        ensemble.append(softmax(data['logits'][keep]/float(data['temperature']), axis=1))
    predictions['ensemble'] = np.mean(ensemble, axis=0)
    for name in HEADS:
        scaler = StandardScaler().fit(x[name][fit], sample_weight=weights)
        xx = scaler.transform(x[name])
        trials = []
        for c in config['regularization_C']:
            model = LogisticRegression(C=c, max_iter=2000, solver='lbfgs', random_state=config['seed'])
            model.fit(xx[fit], labels[fit], sample_weight=weights)
            if not np.array_equal(model.classes_, np.arange(5)):
                raise ValueError('Unexpected class ordering')
            result = identification_metrics(labels[dev], model.predict_proba(xx[dev]))
            trials.append((result['macro_f1'], c, model))
            print(name, c, 'development macro-F1', result['macro_f1'], flush=True)
        score, c, model = max(trials, key=lambda t: (t[0], -t[1]))
        z = model.decision_function(xx[dev])
        solution = minimize_scalar(lambda logt: float(np.mean(logsumexp(z/np.exp(logt), axis=1)-
            (z/np.exp(logt))[np.arange(len(z)), labels[dev]])), bounds=(-3, 3), method='bounded')
        head = {'model': model, 'scaler': scaler, 'temperature': float(np.exp(solution.x)), 'C': c}
        heads[name] = head
        predictions[name] = calibrated(head, x[name])
        search[name] = {'trials': [{'C': trial[1], 'dev_macro_f1': trial[0],
                                  'iterations': trial[2].n_iter_.tolist()} for trial in trials],
                        'selected_C': c, 'temperature': head['temperature'], 'input_dimensions': x[name].shape[1]}
    # A fixed, development-only choice; never rank methods on the fresh cohort.
    development = {name: identification_metrics(labels[dev], p[dev]) for name, p in predictions.items()}
    comparator = max(['anchor', 'ensemble', 'acoustic_refit'], key=lambda name: development[name]['macro_f1'])
    penalties = []
    for penalty in config['disagreement_penalties']:
        score = review_score(predictions['fusion'], predictions['physical_only'], prototype_probabilities, penalty)
        accepted = np.argsort(-score[dev], kind='stable')[:int(np.ceil(config['primary_report_coverage']*dev.sum()))]
        correct = predictions['fusion'][dev].argmax(1) == labels[dev]
        penalties.append({'penalty': penalty, 'dev_issued_accuracy': float(correct[accepted].mean())})
    penalty = max(penalties, key=lambda t: (t['dev_issued_accuracy'], -t['penalty']))['penalty']
    scores = {name: p.max(1) for name, p in predictions.items()}
    scores['fusion_maxprob'] = scores['fusion'].copy()
    predictions['fusion_maxprob'] = predictions['fusion']
    scores['fusion'] = review_score(predictions['fusion'], predictions['physical_only'], prototype_probabilities, penalty)
    thresholds = {name: coverage_threshold(score[cal], config['primary_report_coverage']) for name, score in scores.items()}
    binary_thresholds = {}
    for name, p in predictions.items():
        options = [(identification_metrics(labels[dev], p[dev], binary_threshold=t)['vessel_background_balanced_accuracy'], t)
                   for t in np.linspace(.05, .95, 19)]
        binary_thresholds[name] = max(options, key=lambda v: (v[0], -abs(v[1]-.5)))[1]
    model_path = ROOT/'checkpoints/identification_fusion.joblib'
    joblib.dump({'heads': heads, 'prototypes': prototypes, 'prototype_classes': classes, 'prototype_groups': bank_groups,
                 'penalty': penalty, 'thresholds': thresholds, 'binary_thresholds': binary_thresholds,
                 'primary_comparator': comparator}, model_path)
    private_path = ROOT/'results/raw/identification'
    private_path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(private_path/'development_and_calibration.npz', ids=np.array([r['id'] for r in selected]),
                        **{name: p for name, p in predictions.items()})
    mixture_diagnostic = {name:{str(a):identification_metrics(labels[dev],
        ((1-a)*predictions['ensemble']+a*predictions[name])[dev])['macro_f1']
        for a in [0, .25, .5, .75, 1]} for name in ['acoustic_refit','fusion','no_physical','no_prototypes']}
    summary = {'completed_utc': datetime.now(timezone.utc).isoformat(), 'fit_clips': int(fit.sum()),
               'fit_sources': len({r['source_group'] for r, use in zip(selected, fit) if use}),
               'development_clips': int(dev.sum()), 'calibration_clips': int(cal.sum()),
               'training_source_prototypes': len(prototypes), 'search': search, 'development_metrics': development,
               'penalty_search': penalties, 'selected_penalty': penalty, 'primary_comparator': comparator,
               'thresholds': thresholds, 'binary_thresholds': binary_thresholds,
               'checkpoint_sha256': file_hash(model_path), 'fresh_test_outcomes_examined': False,
               'pre_freeze_development_mixture_diagnostic':mixture_diagnostic,
               'mixture_diagnostic_note':'Pre-freeze diagnostic prompted by weaker development fusion. No mixture exceeded the ensemble on development; mixtures were not adopted and are not additional test-selected methods.'}
    atomic_json(ROOT/'results/manifests/identification_training.json', summary)
    print('IDENTIFICATION FIT COMPLETE; comparator', comparator, flush=True)


if __name__ == '__main__':
    main()
