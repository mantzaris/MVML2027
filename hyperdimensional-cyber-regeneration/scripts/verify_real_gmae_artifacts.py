#!/usr/bin/env python3
"""Verify recorded GraphMAE follow-up artifacts without learning or inference.

This is a read-only provenance/count audit. It checks files already produced by
the CUDA runner/audit and CPU analyzer/renderers, preserving all earlier studies.
"""
import csv
import hashlib
import itertools
import json
import math
import subprocess
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'results/real_gmae'
BASE = ROOT/'results/real'
SEEDS = [11, 23, 37, 51, 71]
SCENARIOS = [10, 11, 12]
CONDITIONS = ['clean', 'missing20', 'missing50', 'spoof20', 'spoof40']
IMPORTED = {'NB', 'LOGIT', 'DAE', 'DENSE', 'GMAE', 'HDC', 'SCR_FLOW', 'CTX_FLOW', 'LOCAL_FLOW'}
NEW = {'EDGE_GMAE', 'EDGE_DENSE', 'CONS_GMAE', 'CONS_PRE'}
VARIANTS = {'EDGE_GMAE', 'CONS_GMAE'}
METHODS = IMPORTED | NEW
REPLAY = NEW | {'GMAE'}
MAIN = ['gmae_fig01_quality', 'gmae_fig02_capture_calibration', 'gmae_fig03_stress_ablations', 'gmae_fig04_resource_cost']
EXTRA = ['gmae_figS1_training', 'gmae_figS2_seed_effects']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def key(row):
    return (int(row['seed']), row['method'], row['condition'], int(row['scenario']))


def bounded(value, lower=0., upper=1.):
    return math.isfinite(float(value)) and lower <= float(value) <= upper


def hashes(catalogue):
    assert catalogue
    for relative, digest in catalogue.items():
        assert sha(ROOT/relative) == digest, relative


def source_commit(manifest):
    """Check the frozen scientific source files also exist in the recorded commit."""
    project_prefix = subprocess.check_output(['git', 'rev-parse', '--show-prefix'], cwd=ROOT, text=True).strip()
    for relative, digest in manifest['source_hashes'].items():
        content = subprocess.check_output(['git', 'show', manifest['source_commit']+':'+project_prefix+relative], cwd=ROOT)
        assert hashlib.sha256(content).hexdigest() == digest, relative


def verify_study():
    manifest = read(DATA/'manifest.json')
    prior = read(BASE/'manifest.json')
    assert manifest['settings']['pilot'] is False
    assert manifest['settings']['seeds'] == SEEDS
    for setting, expected in [('epochs', 40), ('pretrain_epochs', 5), ('patience', 8), ('batch', 256), ('lr', .001)]:
        assert manifest['settings'][setting] == expected
    assert set(manifest['methods']) == METHODS and len(manifest['methods']) == 13
    assert set(manifest['new_methods']) == NEW and set(manifest['imported_methods']) == IMPORTED
    assert manifest['hardware'] == 'NVIDIA RTX 5000 Ada Generation'
    assert manifest['hardware'] == prior['hardware']
    for method in METHODS:
        assert manifest['method_origins'][method] == ('real_gmae' if method in NEW else 'real')
    assert 0 < manifest['all_training_complete_seconds'] < manifest['wall_seconds']
    assert manifest['interpretation'] and 'exploratory' in manifest['interpretation'].lower()
    hashes(manifest['source_hashes'])
    source_commit(manifest)
    config = read(ROOT/'configs/study_real_gmae.json')
    assert config == manifest['frozen_protocol']
    assert config['seeds'] == SEEDS and set(config['methods']) == NEW
    assert config['test_scenarios'] == SCENARIOS and config['conditions'] == CONDITIONS
    assert config['test_reuse'] and config['selection']

    for name in ['results/study_manifest.json', 'results/v2/manifest.json', 'results/v3/manifest.json', 'results/real/manifest.json']:
        hashes(read(ROOT/name)['source_hashes'])
    for name, field in [('manifest.json', 'base_manifest_sha256'), ('summary.csv', 'base_summary_sha256'),
                        ('curves.csv', 'base_curves_sha256'), ('calibration.csv', 'base_calibration_sha256')]:
        assert sha(BASE/name) == manifest[field], field
    for field in ['case_hashes', 'train_hash', 'validation_hashes', 'preprocessing', 'data_manifest_sha256', 'train_windows', 'train_flows']:
        assert manifest[field] == prior[field], field
    assert len(manifest['case_hashes']) == 15
    assert set(manifest['case_hashes']) == {f'{c}_{s}' for c in CONDITIONS for s in SCENARIOS}
    dataset = read(BASE/'ctu13_data_manifest.json')
    assert sha(BASE/'ctu13_data_manifest.json') == manifest['data_manifest_sha256']
    assert dataset['complete'] and dataset['temporal_split_check']['passed'] and len(dataset['files']) == 13
    temporal = dataset['temporal_split_check']
    assert temporal['latest_train_completion'] < temporal['earliest_validation_start']
    assert temporal['latest_validation_completion'] < temporal['earliest_test_start']
    assert sum(r['rows_retained'] for r in dataset['files']) == 799777
    assert dataset['totals']['test']['rows_retained'] == 142813
    for record in dataset['files']:
        assert sha(ROOT/record['prepared_path']) == record['prepared_sha256']

    raw = manifest['raw_hashes']
    assert len(raw) == 975
    hashes(raw)
    for relative, digest in prior['raw_hashes'].items():
        assert raw[relative] == digest
    records = rows(DATA/'summary.csv')
    summary = {key(r): r for r in records}
    assert len(records) == len(summary) == 975
    assert set(summary) == set(itertools.product(SEEDS, METHODS, CONDITIONS, SCENARIOS))
    previous = {key(r): r for r in rows(BASE/'summary.csv')}
    imported = {k: r for k, r in summary.items() if r['method'] in IMPORTED}
    assert imported == previous and len(imported) == 675
    old_curves = rows(BASE/'curves.csv')
    new_curves = rows(DATA/'curves.csv')
    assert [r for r in new_curves if r['method'] in IMPORTED] == old_curves
    captured_counts = {r['scenario']: r['rows_retained'] for r in dataset['files']}
    for row in records:
        seed, method, condition, scenario = key(row)
        assert int(row['n']) == captured_counts[scenario]
        for field in ['ap', 'auc', 'brier', 'balanced_accuracy', 'f1', 'recall', 'fpr']:
            assert bounded(row[field], 0., 1.+1e-6), (key(row), field)
        for field in ['latency_ms', 'p95', 'peak_vram_mb', 'train_seconds']:
            assert math.isfinite(float(row[field])) and float(row[field]) >= 0
        origin = manifest['method_origins'][method]
        assert f'results/runs/{origin}/{seed}/{method}_{condition}_{scenario}.npz' in raw

    checkpoint = {(int(r['seed']), r['method']): r for r in manifest['checkpoints']}
    expected = set(itertools.product(SEEDS, METHODS))
    assert len(checkpoint) == len(manifest['checkpoints']) == 65 and set(checkpoint) == expected
    assert sum(m in NEW for _, m in checkpoint) == 20
    assert sum(m in IMPORTED for _, m in checkpoint) == 45
    old_checkpoint = {(int(r['seed']), r['method']): r for r in prior['checkpoints']}
    for pair, record in checkpoint.items():
        seed, method = pair
        path = f"results/checkpoints/{manifest['method_origins'][method]}/{seed}_{method}.pt"
        assert record['checkpoint_path'] == path and sha(ROOT/path) == record['checkpoint_sha256']
        if method in IMPORTED:
            assert all(record[k] == v for k, v in old_checkpoint[pair].items())
        else:
            assert record['parameters'] == 121643
            assert record['pretrain_epochs'] == (0 if method == 'EDGE_DENSE' else 5)
            assert 1 <= record['epochs_run'] <= 40 and 0 <= record['best_epoch'] < record['epochs_run']
            assert math.isfinite(record['validation_loss']) and record['validation_loss'] >= 0
    threshold = {(int(r['seed']), r['method']): r for r in manifest['thresholds']}
    assert len(threshold) == len(manifest['thresholds']) == 65 and set(threshold) == expected
    old_threshold = {(int(r['seed']), r['method']): r for r in prior['thresholds']}
    for pair, record in threshold.items():
        assert bounded(record['threshold'], -1e-6, 1.+1e-6)
        assert record['selection'] == 'clean validation F1'
        if pair[1] in IMPORTED:
            assert record == old_threshold[pair]
    for row in records:
        assert float(row['threshold']) == float(threshold[(int(row['seed']), row['method'])]['threshold'])
    return manifest, dataset, summary, checkpoint


def verify_analysis(manifest, summary):
    comparison = read(DATA/'comparisons.json')
    assert comparison['exploratory_test_reuse'] is True
    assert comparison['manifest_sha256'] == sha(DATA/'manifest.json')
    assert comparison['summary_sha256'] == sha(DATA/'summary.csv')
    assert comparison['analysis_source_sha256'] == sha(ROOT/'scripts/analyze_real_gmae.py')
    hashes(comparison['analysis_dependency_hashes'])
    assert comparison['raw_input_sha256'] == manifest['raw_hashes']
    assert comparison['seeds'] == SEEDS and comparison['test_scenarios'] == SCENARIOS
    assert comparison['conditions'] == CONDITIONS and set(comparison['primary']) == VARIANTS
    design = comparison['bootstrap']
    assert design['bootstrap_repetitions'] == 1000 and design['bootstrap_rng_seed'] == 20261126
    assert design['shared_draws_across_two_contrasts'] is True
    assert design['blocks_per_capture'] == {'10': 60, '11': 6, '12': 5}
    assert design['rejected_zero_positive_draws'] >= 0 and design['multiplicity'] and design['uncertainty_scope']
    for method, record in comparison['primary'].items():
        assert record['method'] == method and record['comparator'] == 'GMAE'
        assert record['metric'] == 'macro_attack_average_precision' and record['condition'] == 'clean'
        assert record['higher_is_better'] is True and set(record['means']) == {method, 'GMAE'}
        assert set(record['seed_effects']) == set(map(str, SEEDS))
        assert set(record['per_capture_effects']) == set(map(str, SCENARIOS))
        for m, mean in record['means'].items():
            assert bounded(mean)
            csv_mean = sum(float(summary[(s, m, 'clean', c)]['ap']) for s in SEEDS for c in SCENARIOS)/15
            assert abs(mean-csv_mean) <= 2e-6
        assert abs(record['estimate']-(record['means'][method]-record['means']['GMAE'])) <= 1e-12
        for s in SEEDS:
            effect = sum(float(summary[(s, method, 'clean', c)]['ap'])-float(summary[(s, 'GMAE', 'clean', c)]['ap']) for c in SCENARIOS)/3
            assert abs(record['seed_effects'][str(s)]-effect) <= 4e-6
        lo, hi = record['percentile_95_ci']
        blo, bhi = record['bonferroni_97_5_ci']
        assert all(bounded(v, -1., 1.) for v in [lo, hi, blo, bhi])
        assert blo <= lo <= hi <= bhi
    audit = comparison['audit']
    assert audit['summary_records'] == audit['raw_archives_verified'] == 975
    assert audit['imported_records'] == 675 and audit['new_records'] == 300
    assert audit['raw_summary_metric_checks'] == 975*7
    assert audit['checkpoint_hashes_verified'] == 65 and audit['prepared_capture_hashes_verified'] == 13
    assert audit['completion_time_split_verified'] is True and audit['prior_baseline_records_unchanged'] is True
    return comparison


def verify_calibration(manifest, comparison, summary):
    path = DATA/'calibration_audit.json'
    audit = read(path)
    assert comparison['calibration_audit_sha256'] == sha(path)
    assert comparison['calibration_audit'] == {k: v for k, v in audit.items() if k != 'per_case'}
    assert audit['manifest_sha256'] == sha(DATA/'manifest.json')
    assert audit['summary_sha256'] == sha(DATA/'summary.csv')
    assert audit['analysis_source_sha256'] == sha(ROOT/'scripts/analyze_real_gmae.py')
    assert audit['calibration_sha256'] == comparison['calibration_sha256'] == sha(DATA/'calibration.csv')
    catalogue = json.dumps(manifest['raw_hashes'], sort_keys=True, separators=(',', ':')).encode()
    assert audit['raw_hash_catalogue_sha256'] == hashlib.sha256(catalogue).hexdigest()
    assert audit['raw_archives'] == 975 and audit['calibration_rows'] == 9750
    assert audit['tolerance'] == 1e-6 and audit['policy']
    cases = {key(r): r for r in audit['per_case']}
    assert len(cases) == len(audit['per_case']) == 975 and set(cases) == set(summary)
    calibration = rows(DATA/'calibration.csv')
    assert len(calibration) == 9750
    bins = {}
    for row in calibration:
        case, index = key(row), int(row['bin'])
        assert case in summary and 0 <= index <= 9
        selected = bins.setdefault(case, {})
        assert index not in selected
        selected[index] = row
        assert int(row['count']) >= 0 and bounded(row['confidence'])
        if int(row['count']):
            assert bounded(row['accuracy'])
            assert index/10-1e-6 <= float(row['confidence']) <= (index+1)/10+1e-6
        else:
            assert math.isnan(float(row['accuracy']))
    assert set(bins) == set(summary)
    total = excess_count = 0
    maximum_excess = 0.
    for case, record in cases.items():
        assert set(bins[case]) == set(range(10))
        count = int(summary[case]['n'])
        assert record['n'] == record['binned'] == sum(int(r['count']) for r in bins[case].values()) == count
        assert 0 <= record['out_of_range'] <= count and bounded(record['max_excess'], 0., 1e-6)
        assert (record['max_excess'] == 0.) == (record['out_of_range'] == 0)
        mass = sum(int(r['count'])*float(r['accuracy']) for r in bins[case].values() if int(r['count']))
        assert abs(mass-int(summary[case]['positives'])) <= max(.001, 5e-7*count)
        total += count
        excess_count += record['out_of_range']
        maximum_excess = max(maximum_excess, record['max_excess'])
    assert total == audit['predictions_total'] == audit['binned_total'] == 46414225
    assert excess_count == audit['out_of_range_predictions'] and maximum_excess == audit['maximum_bound_excess']
    return total


def verify_gpu(manifest, dataset, checkpoint):
    audit = read(DATA/'gpu_audit.json')
    assert audit['passed'] is True
    assert audit['manifest_sha256'] == sha(DATA/'manifest.json')
    assert audit['summary_sha256'] == sha(DATA/'summary.csv')
    assert audit['base_manifest_sha256'] == manifest['base_manifest_sha256']
    assert audit['hardware'] == manifest['hardware'] and audit['seed'] == 37
    expected_sources = {'scripts/audit_real_gmae.py', 'scripts/measure_real_pipeline.py', 'scripts/run_real_gmae.py',
                        'scripts/run_real.py', 'src/real_gmae_variants.py', 'src/real_models.py', 'src/real_data.py', 'src/real_metrics.py'}
    assert set(audit['source_hashes']) == expected_sources
    hashes(audit['source_hashes'])
    assert audit['input_bytes'] == 2784 and audit['output_bytes'] == 128 and audit['warmups'] == 5
    assert audit['labels_in_timing_inputs'] is False and audit['model_and_statistics_resident'] is True
    assert audit['timing_scope']
    timing = {r['method']: r for r in audit['timing_rows']}
    assert len(timing) == len(audit['timing_rows']) == 13 and set(timing) == METHODS
    for method, row in timing.items():
        record = checkpoint[(37, method)]
        assert row['checkpoint_path'] == record['checkpoint_path'] and row['checkpoint_sha256'] == record['checkpoint_sha256']
        assert row['parameters'] == record['parameters']
        assert row['wall_repeats'] == 50 and row['cuda_repeats'] == 30
        values = [row['wall_p10_ms'], row['wall_median_ms'], row['wall_p95_ms']]
        assert all(math.isfinite(v) and v >= 0 for v in values) and values[0] <= values[1] <= values[2]
        assert 0 <= row['cuda_median_ms'] <= row['cuda_p95_ms'] and math.isfinite(row['cuda_p95_ms'])
        assert math.isfinite(row['peak_vram_mb']) and row['peak_vram_mb'] >= 0
    replay = {(r['method'], r['condition'], int(r['scenario'])): r for r in audit['replay_rows']}
    assert len(replay) == len(audit['replay_rows']) == 75
    assert set(replay) == set(itertools.product(REPLAY, CONDITIONS, SCENARIOS))
    counts = {r['scenario']: r['rows_retained'] for r in dataset['files']}
    for (method, condition, scenario), row in replay.items():
        record = checkpoint[(37, method)]
        assert row['flows'] == counts[scenario]
        assert row['input_sha256'] == manifest['case_hashes'][f'{condition}_{scenario}']
        assert row['checkpoint_path'] == record['checkpoint_path'] and row['checkpoint_sha256'] == record['checkpoint_sha256']
        origin = manifest['method_origins'][method]
        expected = f'results/runs/{origin}/37/{method}_{condition}_{scenario}.npz'
        assert row['raw_path'] == expected and row['raw_sha256'] == manifest['raw_hashes'][expected]
        assert bounded(row['probability_error'], 0., 1e-5)
        assert bounded(row['ap_error'], 0., 1e-8) and bounded(row['brier_error'], 0., 1e-8)
    assert sum(r['flows'] for r in replay.values()) == audit['model_flow_evaluations'] == 3570325
    assert set(audit['maximum_errors']) == {'prob', 'ap', 'brier'}
    for metric, field in [('prob', 'probability_error'), ('ap', 'ap_error'), ('brier', 'brier_error')]:
        assert audit['maximum_errors'][metric] == max(r[field] for r in replay.values())
    return audit['model_flow_evaluations']


def verify_artifacts():
    for folder, stems in [(ROOT/'figures/real_gmae', MAIN), (ROOT/'supplemental/real_gmae', EXTRA)]:
        for stem in stems:
            record = read(folder/f'{stem}.provenance.json')
            hashes(record['inputs'])
            hashes(record['dependency_sha256'])
            assert record['script'] == 'scripts/make_real_gmae_results.py'
            assert record['script_sha256'] == sha(ROOT/record['script'])
            assert record['seeds'] == SEEDS and record['test_captures'] == SCENARIOS
            assert set(record['output_sha256']) == {'png', 'pdf'}
            for suffix, digest in record['output_sha256'].items():
                assert sha(folder/f'{stem}.{suffix}') == digest
            assert len(PdfReader(folder/f'{stem}.pdf').pages) == 1
    folder = ROOT/'supplemental/real_gmae'
    path = folder/'Supplement_real_gmae.pdf'
    pages = PdfReader(path).pages
    assert len(pages) == 3 and all(len(page.extract_text()) > 350 for page in pages)
    record = read(folder/'Supplement_real_gmae.provenance.json')
    hashes(record['inputs'])
    hashes(record['dependency_sha256'])
    assert record['script'] == 'scripts/make_real_gmae_supplement.py'
    assert record['script_sha256'] == sha(ROOT/record['script']) and record['pdf_sha256'] == sha(path)
    assert record['pages'] == 3 and record['figures'] == EXTRA
    assert (ROOT/'figures/real_gmae/index.html').is_file()
    assert (ROOT/'docs/RESULTS_REAL_GMAE.md').is_file()


def main():
    manifest, dataset, summary, checkpoint = verify_study()
    comparison = verify_analysis(manifest, summary)
    calibration = verify_calibration(manifest, comparison, summary)
    evaluations = verify_gpu(manifest, dataset, checkpoint)
    verify_artifacts()
    print(json.dumps(dict(verified=True, summary_records=975, imported_records=675, new_records=300,
        fitted_checkpoints=65, new_training_runs=20, clean_variant_contrasts=2,
        primary_interval_levels=[95, 97.5], exploratory_test_reuse=True,
        prior_scientific_sources_preserved=True, imported_baseline_records_unchanged=True,
        timing_methods=13, replay_cases=75, model_flow_evaluations=evaluations,
        calibration_predictions=calibration, main_figures=4, extra_figures=2, supplement_pages=3)))


if __name__ == '__main__':
    main()
