"""Check actual research deliverables. Never starts/stops compute or unlocks test."""
import json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(relative):
    path=ROOT/relative
    return json.loads(path.read_text()) if path.exists() else {}
config=read('configs/execution.json');unlock=read('protocol/test_unlock_log.json')
ledger=read('results/manifests/execution_ledger.json');main=read('results/tables/main_results.json')
pdf=ROOT/'manuscript/manuscript.pdf';pages=None
if pdf.exists():
    info=subprocess.check_output(['pdfinfo',str(pdf)],text=True)
    pages=int(next(line.split(':',1)[1] for line in info.splitlines() if line.startswith('Pages:')))
gates={
    'authorized_ten_hour_window':config.get('approved_allocated_gpu_hours')==10,
    'local_artifact_preservation':config.get('persistent_storage_verified') is True,
    'source_integrity':config.get('primary_data_verified') is True,
    'ai_reference_assessments':config.get('ai_reviewer_assessments_complete') is True,
    'development_gate':read('results/manifests/pilot_gate.json').get('passed') is True,
    'prospective_test_freeze':unlock.get('unlocked') is True and unlock.get('test_outcomes_examined_at_freeze') is False,
    'frozen_method_integrity':read('results/manifests/freeze_integrity.json').get('method_files_and_models_match') is True,
    'clean_test_analysis':main.get('assessment_clips')==100,
    'waveform_stress_analysis':read('results/tables/robustness_results.json').get('transformed_clips')==360,
    'report_audits':bool(read('results/tables/report_audits.json')),
    'editable_manuscript':(ROOT/'manuscript/manuscript.docx').exists(),
    'pdf_within_eight_pages':pages is not None and 1<=pages<=8,
    'finished_within_compute_ceiling':ledger.get('finished') is True and ledger.get('within_authorized_window') is True,
}
for name,passed in gates.items():print(('PASS ' if passed else 'PENDING ')+name)
print('Measured adaptive-selection success:',main.get('preregistered_success_criteria_met','not yet evaluated'))
print('No human reviewers; Pod must remain running; no push or submission.')
print('Manuscript is anonymous by instruction. Submission formatting item: template margin discrepancy resolution.')
if config.get('actual_pod_hourly_usd') is None:print('Dollar cost and prior billed allocation remain unknown.')
raise SystemExit(0 if all(gates.values()) else 2)
