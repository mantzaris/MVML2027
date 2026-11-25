#!/usr/bin/env python3
"""Verify external-study artifacts, GPU audits and preserved prior sources.

This checks already-recorded outputs and hashes. It performs no training,
model inference, bootstrap analysis or GPU operations.
"""
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [11, 23, 37, 51, 71]
SCENARIOS = [10, 11, 12]
CONDITIONS = ['clean', 'missing20', 'missing50', 'spoof20', 'spoof40']
METHODS = {'LOGIT', 'DAE', 'DENSE', 'GMAE', 'HDC', 'SCR_FLOW', 'CTX_FLOW', 'LOCAL_FLOW', 'NB'}
REPLAY_METHODS = {'SCR_FLOW', 'GMAE', 'DAE'}
MAIN = ['real_fig01_external_quality', 'real_fig02_capture_calibration', 'real_fig03_corruption_transfer', 'real_fig04_resource_cost']
EXTRA = ['real_figS1_training', 'real_figS2_seed_operating_points']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inputs(record):
    for relative, digest in record['inputs'].items():
        assert sha(ROOT/relative) == digest, relative


def sources(record, expected=None):
    hashes = record['source_hashes']
    if expected is not None:
        assert set(hashes) == set(expected), set(hashes)
    for relative, digest in hashes.items():
        assert sha(ROOT/relative) == digest, relative


def finite_between(value, lower=0., upper=1.):
    return math.isfinite(float(value)) and lower <= float(value) <= upper


def verify_calibration(data, manifest, comparison, summary):
    """Check display-only clipping provenance and complete per-case bin support."""
    audit_path = data/'calibration_audit.json'
    audit = json.loads(audit_path.read_text())
    assert comparison['calibration_audit_sha256'] == sha(audit_path)
    assert audit['manifest_sha256'] == sha(data/'manifest.json')
    assert audit['summary_sha256'] == sha(data/'summary.csv')
    assert audit['analysis_source_sha256'] == sha(ROOT/'scripts/analyze_real.py')
    assert audit['calibration_sha256'] == sha(data/'calibration.csv') == comparison['calibration_sha256']
    catalogue = json.dumps(manifest['raw_hashes'], sort_keys=True, separators=(',', ':')).encode('utf-8')
    assert audit['raw_hash_catalogue_sha256'] == hashlib.sha256(catalogue).hexdigest()
    assert audit['raw_archives'] == len(manifest['raw_hashes']) == 675
    assert audit['tolerance'] == comparison['audit']['probability_bound_tolerance'] == 1e-6
    assert audit['policy']
    cases = {(int(r['seed']), r['method'], r['condition'], int(r['scenario'])): r for r in audit['per_case']}
    assert len(cases) == len(audit['per_case']) == 675 and set(cases) == set(summary)
    calibration = list(csv.DictReader((data/'calibration.csv').open()))
    assert len(calibration) == audit['calibration_rows'] == 6750
    bins = {}
    for row in calibration:
        case = (int(row['seed']), row['method'], row['condition'], int(row['scenario']))
        index = int(row['bin'])
        assert case in summary and 0 <= index <= 9
        selected = bins.setdefault(case, {})
        assert index not in selected, (case, index)
        selected[index] = row
        count = int(row['count'])
        assert count >= 0 and finite_between(row['confidence'])
        if count:
            confidence, accuracy = float(row['confidence']), float(row['accuracy'])
            assert finite_between(accuracy)
            assert index/10-1e-6 <= confidence <= (index+1)/10+1e-6
        else:
            assert math.isnan(float(row['accuracy']))
    assert set(bins) == set(summary)
    predicted_total = binned_total = out_of_range_total = 0
    maximum_excess = 0.
    for case, record in cases.items():
        assert set(bins[case]) == set(range(10))
        expected = int(summary[case]['n'])
        binned = sum(int(r['count']) for r in bins[case].values())
        assert int(record['n']) == int(record['binned']) == binned == expected
        out_of_range = int(record['out_of_range'])
        assert 0 <= out_of_range <= expected
        excess = float(record['max_excess'])
        assert finite_between(excess, 0., audit['tolerance'])
        assert (excess == 0.) == (out_of_range == 0)
        # Accuracy means use FP32 reductions; allow their documented arithmetic
        # rounding while catching lost labels or missing probability bins.
        positive_mass = sum(int(r['count'])*float(r['accuracy']) for r in bins[case].values() if int(r['count']))
        assert abs(positive_mass-int(summary[case]['positives'])) <= max(.001, 5e-7*expected)
        predicted_total += expected
        binned_total += binned
        out_of_range_total += out_of_range
        maximum_excess = max(maximum_excess, excess)
    assert predicted_total == audit['predictions_total'] == audit['binned_total'] == binned_total
    assert out_of_range_total == audit['out_of_range_predictions'] == comparison['audit']['roundoff_bound_exceedance_predictions']
    assert maximum_excess == audit['maximum_bound_excess'] == comparison['audit']['maximum_probability_bound_excess']
    return {'calibration_predictions': predicted_total,
            'calibration_bound_roundoff_predictions': out_of_range_total}


def verify_gpu_audits(data, manifest, dataset, summary):
    manifest_digest = sha(data/'manifest.json')
    summary_digest = sha(data/'summary.csv')
    checkpoint = {(int(r['seed']), r['method']): r for r in manifest['checkpoints']}
    captured_counts = {r['scenario']: r['rows_retained'] for r in dataset['files']}
    replay = json.loads((data/'replay.json').read_text())
    timing = json.loads((data/'pipeline_timing.json').read_text())

    assert replay['passed'] is True and replay['no_training_or_refitting'] is True
    assert replay['training_seed'] == 37 and set(replay['methods']) == REPLAY_METHODS
    assert replay['test_scenarios'] == SCENARIOS and replay['conditions'] == CONDITIONS
    assert replay['source_manifest_sha256'] == manifest_digest
    assert replay['summary_sha256'] == summary_digest
    assert replay['data_manifest_sha256'] == manifest['data_manifest_sha256']
    assert replay['hardware'] == manifest['hardware']
    assert replay['replay_script_sha256'] == sha(ROOT/'scripts/replay_real.py')
    sources(replay, ['scripts/replay_real.py', 'scripts/measure_real_pipeline.py',
                     'src/real_data.py', 'src/real_models.py', 'src/real_metrics.py'])
    assert 0 < replay['probability_tolerance'] <= 1e-5
    assert 0 < replay['metric_tolerance'] <= 1e-8
    expected = set(itertools.product(REPLAY_METHODS, CONDITIONS, SCENARIOS))
    observed = {(r['method'], r['condition'], int(r['scenario'])) for r in replay['rows']}
    assert observed == expected and len(replay['rows']) == replay['model_condition_replays'] == 45
    expected_flows = dataset['totals']['test']['rows_retained'] * len(CONDITIONS) * len(REPLAY_METHODS)
    assert replay['model_flow_evaluations'] == expected_flows == 2142195
    assert sum(int(r['flows']) for r in replay['rows']) == expected_flows
    assert set(replay['maximum_absolute_errors']) == {'prob', 'ap', 'brier'}
    for row in replay['rows']:
        method, condition, scenario = row['method'], row['condition'], int(row['scenario'])
        assert int(row['flows']) == captured_counts[scenario]
        assert row['paired_metadata_equal'] is True
        assert row['input_data_sha256'] == manifest['case_hashes'][f'{condition}_{scenario}']
        expected_checkpoint_path = f'results/checkpoints/real/37_{method}.pt'
        assert row['checkpoint_path'] == expected_checkpoint_path
        assert row['checkpoint_sha256'] == checkpoint[(37, method)]['checkpoint_sha256']
        assert sha(ROOT/expected_checkpoint_path) == row['checkpoint_sha256']
        expected_raw_path = f'results/runs/real/37/{method}_{condition}_{scenario}.npz'
        assert row['raw_path'] == expected_raw_path
        assert row['raw_sha256'] == manifest['raw_hashes'][expected_raw_path]
        assert sha(ROOT/expected_raw_path) == row['raw_sha256']
        assert set(row['max_absolute_errors']) == {'prob', 'ap', 'brier'}
        for metric, error in row['max_absolute_errors'].items():
            tolerance = replay['probability_tolerance'] if metric == 'prob' else replay['metric_tolerance']
            assert finite_between(error, 0., tolerance), (method, condition, scenario, metric, error)
        reference = summary[(37, method, condition, scenario)]
        for metric in ['ap', 'brier']:
            value = row[f'replayed_{metric}']
            assert finite_between(value)
            assert abs(float(value)-float(reference[metric])) <= 2*replay['metric_tolerance']
    for metric, maximum in replay['maximum_absolute_errors'].items():
        assert maximum == max(r['max_absolute_errors'][metric] for r in replay['rows'])

    assert timing['seed'] == 37 and timing['scenario'] == 10 and timing['condition'] == 'clean'
    assert timing['window'] == 0 and timing['flows'] == 32 and timing['warmups'] == 5
    assert timing['hardware'] == manifest['hardware']
    assert timing['manifest_sha256'] == manifest_digest
    assert timing['data_manifest_sha256'] == manifest['data_manifest_sha256']
    assert timing['case_input_sha256'] == manifest['case_hashes']['clean_10']
    assert timing['labels_in_measured_inputs'] is False and timing['model_and_statistics_resident'] is True
    sources(timing, ['scripts/measure_real_pipeline.py', 'src/real_data.py', 'src/real_models.py', 'src/real_metrics.py'])
    component_bytes = {'features': 32*14*4, 'observed': 32*14, 'src': 32*8, 'dst': 32*8, 'valid': 32}
    assert timing['input_component_bytes'] == component_bytes
    assert timing['input_bytes'] == sum(component_bytes.values()) == 2784
    assert timing['output_bytes'] == 32*4 and timing['resident_normalization_bytes'] == 2*14*4
    assert timing['input_roundtrip'] and timing['prediction_roundtrip']
    assert set(timing['checkpoint_hashes']) == METHODS
    for method, record in timing['checkpoint_hashes'].items():
        path = f'results/checkpoints/real/37_{method}.pt'
        assert record['path'] == path
        assert record['sha256'] == checkpoint[(37, method)]['checkpoint_sha256']
        assert sha(ROOT/path) == record['sha256']
    assert 0 < timing['original_batch_probability_tolerance'] <= 1e-5
    assert set(timing['original_batch_first_window_max_absolute_errors']) == METHODS
    assert all(finite_between(error, 0., timing['original_batch_probability_tolerance'])
               for error in timing['original_batch_first_window_max_absolute_errors'].values())
    assert len(timing['rows']) == 9 and {r['method'] for r in timing['rows']} == METHODS
    for row in timing['rows']:
        assert row['repeats'] == 50 and row['returned_probabilities'] == 32 and row['output_bytes'] == 128
        values = [row['p10_ms'], row['median_ms'], row['p95_ms']]
        assert all(math.isfinite(float(v)) and float(v) >= 0 for v in values)
        assert values[0] <= values[1] <= values[2]
    return {'model_flow_evaluations': expected_flows, 'replay_cases': len(replay['rows']),
            'timing_methods': len(timing['rows']), 'timing_repeats_per_method': 50}


def main():
    data = ROOT/'results/real'
    manifest = json.loads((data/'manifest.json').read_text())
    assert not manifest['settings']['pilot'] and manifest['settings']['seeds'] == SEEDS
    assert set(manifest['methods']) == METHODS
    assert len(manifest['checkpoints']) == 45 and len(manifest['raw_hashes']) == 675 and len(manifest['case_hashes']) == 15
    sources(manifest)
    for name in ['results/study_manifest.json', 'results/v2/manifest.json', 'results/v3/manifest.json']:
        sources(json.loads((ROOT/name).read_text()))
    dataset = json.loads((data/'ctu13_data_manifest.json').read_text())
    assert dataset['complete'] and dataset['temporal_split_check']['passed'] and len(dataset['files']) == 13
    assert sha(data/'ctu13_data_manifest.json') == manifest['data_manifest_sha256']
    assert sum(r['rows_retained'] for r in dataset['files']) == 799777
    assert dataset['totals']['test']['rows_retained'] == 142813
    rows = list(csv.DictReader((data/'summary.csv').open()))
    assert len(rows) == 675 and {r['method'] for r in rows} == METHODS
    summary = {(int(r['seed']), r['method'], r['condition'], int(r['scenario'])): r for r in rows}
    assert len(summary) == 675 and set(summary) == set(itertools.product(SEEDS, METHODS, CONDITIONS, SCENARIOS))
    for row in rows:
        for key in ['ap', 'auc', 'brier', 'balanced_accuracy', 'f1', 'recall', 'fpr']:
            assert finite_between(row[key], 0., 1.+1e-7), (key, row)
        for key in ['latency_ms', 'p95', 'peak_vram_mb', 'train_seconds']:
            assert math.isfinite(float(row[key])) and float(row[key]) >= 0, (key, row)
    comparison = json.loads((data/'comparisons.json').read_text())
    assert sha(data/'manifest.json') == comparison['manifest_sha256']
    assert sha(data/'summary.csv') == comparison['summary_sha256']
    assert sha(ROOT/'scripts/analyze_real.py') == comparison['analysis_source_sha256']
    assert comparison['seeds'] == SEEDS and comparison['test_scenarios'] == SCENARIOS
    assert comparison['conditions'] == CONDITIONS
    assert comparison['raw_input_sha256'] == manifest['raw_hashes']
    primary = comparison['primary']
    assert primary['method'] == 'SCR_FLOW' and primary['comparator'] == 'GMAE'
    assert primary['condition'] == 'clean' and primary['metric'] == 'macro_attack_average_precision'
    assert primary['bootstrap_repetitions'] == 1000 and primary['higher_is_better'] is True
    assert set(primary['means']) == {'SCR_FLOW', 'GMAE'} and all(finite_between(v) for v in primary['means'].values())
    assert abs(primary['estimate']-(primary['means']['SCR_FLOW']-primary['means']['GMAE'])) <= 1e-12
    lo, hi = primary['percentile_95_ci']
    assert finite_between(lo, -1., 1.) and finite_between(hi, -1., 1.) and lo <= hi
    checks = comparison['audit']
    assert checks['summary_records'] == checks['raw_archives_verified'] == 675
    assert checks['raw_summary_metric_checks'] == 675*7
    assert checks['checkpoint_hashes_verified'] == 45 and checks['prepared_capture_hashes_verified'] == 13
    for directory, stems in [(ROOT/'figures/real', MAIN), (ROOT/'supplemental/real', EXTRA)]:
        for stem in stems:
            record = json.loads((directory/f'{stem}.provenance.json').read_text())
            inputs(record)
            assert record['script'] == 'scripts/make_real_results.py' and sha(ROOT/record['script']) == record['script_sha256']
            assert record['seeds'] == SEEDS and record['test_captures'] == SCENARIOS
            assert set(record['output_sha256']) == {'pdf', 'png'}
            for extension, digest in record['output_sha256'].items():
                assert sha(directory/f'{stem}.{extension}') == digest
            assert len(PdfReader(directory/f'{stem}.pdf').pages) == 1
    folder = ROOT/'supplemental/real'
    pdf = folder/'Supplement_real.pdf'
    pages = PdfReader(pdf).pages
    assert len(pages) == 3 and all(len(page.extract_text()) > 350 for page in pages)
    record = json.loads((folder/'Supplement_real.provenance.json').read_text())
    inputs(record)
    assert record['script'] == 'scripts/make_real_supplement.py'
    assert sha(pdf) == record['pdf_sha256'] and sha(ROOT/record['script']) == record['script_sha256']
    assert record['pages'] == 3 and record['figures'] == EXTRA
    calibration_checked = verify_calibration(data, manifest, comparison, summary)
    audited = verify_gpu_audits(data, manifest, dataset, summary)
    print(json.dumps(dict(verified=True, flow_records=799777, heldout_flows=142813, summary_rows=675,
        main_figures=4, extra_figures=2, supplement_pages=3,
        prior_scientific_sources_preserved=True, **audited, **calibration_checked)))


if __name__ == '__main__':
    main()
