"""Record worker status and the completed identification deliverables.

This is an editorial/accounting utility; it never starts or stops GPU work.
"""
import argparse
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

from underwater_reasoning.io import atomic_json, file_hash

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT / path).read_text())


def utc(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00'))


def worker_status():
    gpu = subprocess.check_output([
        'nvidia-smi', '--query-gpu=name,memory.total,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True).strip()
    processes = subprocess.check_output([
        'nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
        '--format=csv,noheader,nounits'], text=True).strip().splitlines()
    disk = shutil.disk_usage(ROOT)
    record = {
        'recorded_utc': datetime.now(timezone.utc).isoformat(),
        'pod_running': True,
        'gpu_csv_fields': ['name', 'memory_total_mib', 'memory_used_mib', 'utilization_percent'],
        'gpu_csv': gpu,
        'active_compute_processes': processes,
        'gpu_idle_at_observation': not processes and float(gpu.rsplit(',', 1)[1]) == 0,
        'workspace_total_bytes': disk.total,
        'workspace_free_bytes': disk.free,
        'pod_stop_requested': False,
        'pod_finish_instruction': 'leave_running',
    }
    atomic_json(ROOT / 'results/manifests/identification_worker_finish.json', record)
    print(json.dumps(record, indent=2))


def complete():
    audit = read('results/manifests/identification_audit.json')
    original = read('results/manifests/freeze_integrity.json')
    preserved = read('results/manifests/identification_preservation.json')
    old_preserved = read('results/manifests/final_preservation.json')
    inspection = read('results/manifests/manuscript_inspection.json')
    worker = read('results/manifests/identification_worker_finish.json')
    config = read('configs/execution.json')
    results = read('results/tables/identification_results.json')
    assert audit['all_passed'] and original['method_files_and_models_match']
    assert original['main_references_and_reviews_match']
    assert preserved['all_match'] and old_preserved['all_match']
    assert worker['pod_running'] and worker['gpu_idle_at_observation']
    assert inspection['all_pages_reviewed'] and not inspection['clipped_text_observed']
    assert inspection['pages'] == 8
    for name in ['pdf', 'docx']:
        assert inspection[name + '_sha256'] == file_hash(ROOT / f'manuscript/manuscript.{name}')
    info = subprocess.check_output(['pdfinfo', str(ROOT / 'manuscript/manuscript.pdf')], text=True)
    assert int(next(line.split(':', 1)[1] for line in info.splitlines() if line.startswith('Pages:'))) == 8
    assert subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip() == 'main'
    suites = ET.parse(ROOT / '.cache/identification_full_tests.xml').getroot().findall('testsuite')
    counts = {key: sum(int(suite.attrib[key]) for suite in suites)
              for key in ['tests', 'failures', 'errors', 'skipped']}
    assert counts['tests'] > 0 and counts['failures'] == counts['errors'] == 0
    counts['passed'] = counts['tests'] - counts['skipped']
    jobs = []
    for pattern in ['identification*.json', 'untrained_categories*.json']:
        for path in sorted((ROOT / 'results/manifests/jobs').glob(pattern)):
            row = json.loads(path.read_text())
            assert row['status'] == 'complete' and row['returncode'] == 0
            jobs.append({key: row[key] for key in ['name', 'started_utc', 'ended_utc', 'elapsed_seconds', 'status']}
                        | {'record_sha256': file_hash(path)})
    assert len(jobs) >= 7
    last = max(utc(row['ended_utc']) for row in jobs)
    start = utc(config['research_accounting_start_utc'])
    deadline = utc(config['research_deadline_utc'])
    assert start < last < deadline
    clean = results['conditions']['clean']
    record = {
        'recorded_utc': datetime.now(timezone.utc).isoformat(),
        'stage': 'completed_identification_followup',
        'fresh_clips': 235, 'fresh_source_groups': 218,
        'transformed_clips': 1410, 'untrained_category_clips': 72,
        'untrained_category_source_groups': 65,
        'generative_test_calls': 705, 'generative_development_calls': 20,
        'primary_comparator': results['primary_comparator'],
        'primary_success': results['primary_success'],
        'primary_macro_f1_contrast': clean['primary_contrast']['macro_f1'],
        'portable_tests': counts,
        'test_report_sha256': file_hash(ROOT / '.cache/identification_full_tests.xml'),
        'followup_audit_passed': True,
        'original_scientific_artifacts_unchanged': True,
        'local_preservation': {'original_study': old_preserved, 'followup': preserved},
        'research_accounting_start_utc': start.isoformat(),
        'last_followup_job_ended_utc': last.isoformat(),
        'conservative_window_through_last_worker_job_hours': (last - start).total_seconds() / 3600,
        'authorized_deadline_utc': deadline.isoformat(),
        'within_original_ten_hour_window': True,
        'recorded_guarded_worker_stages': jobs,
        'recorded_stage_elapsed_seconds_sum': sum(row['elapsed_seconds'] for row in jobs),
        'timing_scope': 'Wall time through the last recorded follow-up worker job, not GPU utilization or billed allocation. Fitting summaries precede the prospective freeze; these seven guarded stages cover the pilot and fresh evaluations.',
        'actual_pod_billing_hours': None, 'actual_usd_cost': None,
        'worker_status': worker,
        'human_reviewers': 0,
        'identification_targets': 'released dataset categories, not AI or human-review labels',
        'pdf_pages': 8, 'maximum_conference_pages': 8,
        'abstract_words': inspection['abstract_words'],
        'manuscript_authorship': 'anonymous by user instruction',
        'no_push_or_submission_performed': True,
        'large_artifacts_excluded_from_git': True,
        'submission_formatting_item': 'Official template geometry and its prose specify different horizontal margins; organizer resolution remains necessary before submission.',
    }
    atomic_json(ROOT / 'results/manifests/identification_completion.json', record)
    print(json.dumps({key: record[key] for key in [
        'stage', 'primary_success', 'portable_tests', 'pdf_pages',
        'conservative_window_through_last_worker_job_hours']}, indent=2))


def artifacts():
    completion = read('results/manifests/identification_completion.json')
    # Retain the original study's required checks without changing its frozen builder.
    subprocess.run([str(ROOT / '.venv/bin/python'), str(ROOT / 'scripts/build_artifact_manifest.py')], check=True, cwd=ROOT)
    path = ROOT / 'results/manifests/research_artifacts.json'
    prior = json.loads(path.read_text())
    records = prior['artifacts']
    record = {
        'stage': 'completed_identification_followup_with_original_study_retained',
        'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
        'active_completion_record': 'results/manifests/identification_completion.json',
        'active_completion_sha256': file_hash(ROOT / 'results/manifests/identification_completion.json'),
        'pdf_pages': completion['pdf_pages'],
        'abstract_words': completion['abstract_words'],
        'primary_identification_success': completion['primary_success'],
        'human_reviewers': 0,
        'portable_tests_passed': completion['portable_tests']['passed'],
        'conservative_research_elapsed_hours': completion['conservative_window_through_last_worker_job_hours'],
        'authorized_ceiling_hours': 10,
        'actual_pod_billing_hours': None, 'actual_usd_cost': None,
        'pod_finish_instruction': 'leave_running',
        'no_push_or_submission_performed': True,
        'original_study_summary': {key: value for key, value in prior.items()
                                   if key not in ['artifacts', 'artifact_bytes_excluding_this_manifest',
                                                  'pdf_pages', 'abstract_words', 'manuscript_status',
                                                  'manuscript_authorship', 'recorded_at_utc']},
        'artifact_bytes_excluding_this_manifest': sum(row['bytes'] for row in records),
        'artifacts': records,
    }
    atomic_json(path, record)
    print(len(records), 'compact deliverables;', record['artifact_bytes_excluding_this_manifest'], 'bytes')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--worker-status', action='store_true')
    mode.add_argument('--complete', action='store_true')
    mode.add_argument('--artifacts', action='store_true')
    args = parser.parse_args()
    if args.worker_status:
        worker_status()
    elif args.complete:
        complete()
    else:
        artifacts()
