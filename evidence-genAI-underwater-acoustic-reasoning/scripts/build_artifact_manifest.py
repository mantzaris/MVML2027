"""Hash completed research deliverables; preserve the historical preparation audit."""
import json,subprocess
from datetime import datetime,timezone
from pathlib import Path
from underwater_reasoning.io import atomic_json,file_hash
ROOT=Path(__file__).resolve().parents[1]
def read(path):return json.loads((ROOT/path).read_text())
main=read('results/tables/main_results.json');stress=read('results/tables/robustness_results.json')
audits=read('results/tables/report_audits.json');reviews=read('results/tables/reviewer_quality.json')
ledger=read('results/manifests/execution_ledger.json');validation=read('results/manifests/software_validation.json')
inspection=read('results/manifests/manuscript_inspection.json');preservation=read('results/manifests/final_preservation.json')
unlock=read('protocol/test_unlock_log.json')
assert main['assessment_clips']==100 and stress['transformed_clips']==360
assert audits['paraphrase']['n']==100 and audits['paraphrase_2']['n']==100
intervention_count=sum(v['n'] for v in audits['evidence_interventions'].values())
assert 125<=intervention_count<=175
assert all(audits['evidence_interventions'][k]['n']==25 for k in ('missing_A','missing_B','swapped_A','false_A','corrupt_D_labels'))
assert sum(v['requested'] for v in audits['prose'].values())==120 and reviews['counts']['requested']==540
assert ledger['finished'] and ledger['within_authorized_window'] and preservation['all_match']
assert unlock['unlocked'] and not unlock['test_outcomes_examined_at_freeze']
assert inspection['all_pages_reviewed'] and not inspection['clipped_text_observed']
pdf=ROOT/'manuscript/manuscript.pdf';info=subprocess.check_output(['pdfinfo',str(pdf)],text=True)
pages=int(next(line.split(':',1)[1] for line in info.splitlines() if line.startswith('Pages:')))
assert 1<=pages<=8 and inspection['pages']==pages
text=(ROOT/'manuscript/manuscript.txt').read_text();abstract=text.split('Abstract — ',1)[1].split('\n\nKeywords:',1)[0]
assert len(abstract.split())<=300
destination=ROOT/'results/manifests/research_artifacts.json'
paths=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','-z','.'],cwd=ROOT).decode().split('\0')
files=sorted({p for p in paths if p and (ROOT/p).is_file() and ROOT/p!=destination})
for path in files:
    assert not any(part in ('raw','private','checkpoints','.venv','.cache') for part in Path(path).parts),path
    assert (ROOT/path).stat().st_size<5_000_000,(path,'unexpected large Git artifact')
records=[{'path':p,'bytes':(ROOT/p).stat().st_size,'sha256':file_hash(ROOT/p)} for p in files]
record={'stage':'completed_real_recording_research_with_automated_proxy_assessment','recorded_at_utc':datetime.now(timezone.utc).isoformat(),
    'human_reviewers':0,'adaptive_success_criteria_met':main['preregistered_success_criteria_met'],
    'assessment_test_clips':100,'assessment_test_source_groups':main['assessment_source_groups'],'classifier_test_clips':2240,
    'main_ai_review_calls':540,'main_ai_valid_reviews':reviews['counts']['valid'],'transformed_clips':360,
    'prose_audits_requested':120,'evidence_intervention_reports':intervention_count,'prompt_paraphrase_reports':200,
    'portable_tests_passed':validation['local']['passed'],'worker_tests_passed':validation['remote']['passed'],
    'synthetic_tool_fixtures':500,'structural_interventions':100,'analysis_math_checks':validation['analysis_math']['checks_passed'],
    'conservative_research_elapsed_hours':ledger['conservative_research_elapsed_hours'],'authorized_ceiling_hours':10,
    'actual_pod_billing_hours':None,'actual_usd_cost':None,'pod_finish_instruction':'leave_running',
    'manuscript_status':'empirical_manuscript_for_author_review','pdf_pages':pages,'abstract_words':len(abstract.split()),
    'manuscript_authorship':'anonymous by user instruction; affiliation/contact omitted',
    'submission_items_remaining':['organizer resolution of template margin discrepancy'],
    'study_limitations':['missing parent-recording/deployment provenance','automated and correlated same-model assessments','small development/calibration sets','single region and anchor generator'],
    'no_push_or_submission_performed':True,'historical_preparation_record':'results/manifests/preparation_artifacts.json',
    'artifact_bytes_excluding_this_manifest':sum(r['bytes'] for r in records),'artifacts':records}
atomic_json(destination,record)
print(len(records),'compact artifacts;',record['artifact_bytes_excluding_this_manifest'],'bytes;',pages,'PDF pages')
