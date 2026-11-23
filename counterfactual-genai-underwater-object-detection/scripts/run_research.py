#!/usr/bin/env python3
"""One-command, sequential GPU research continuation with durable stage receipts."""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path
from uac.runtime import RunLock,EventLedger,atomic_json,atomic_bytes,sha256_file,fingerprint,utc_now

from stage_provenance import stage_inputs,native_adoption_evidence

ROOT=Path(__file__).resolve().parents[1];os.chdir(ROOT)
PYTHON=str(ROOT/'.venv/bin/python');ledger=EventLedger();child=None;download_child=None;stopping=False

def read(path):return json.loads((ROOT/path).read_text())
def handle(signum,frame):
    global stopping
    stopping=True
    for process in (child,download_child):
        if process is not None and process.poll() is None:
            try:os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError:pass

def wait_for_owned_group(group):
    """Give signaled descendants a checkpoint boundary before leaving a stage."""
    deadline=time.monotonic()+60
    while True:
        try:os.killpg(group,0)
        except ProcessLookupError:return
        if time.monotonic()>=deadline:
            try:os.killpg(group,signal.SIGKILL)
            except ProcessLookupError:pass
            ledger.append('owned_process_group_forced_stop',process_group=group,grace_seconds=60)
            return
        time.sleep(.2)


def invoke(script,*args,gpu=True):
    command=[PYTHON,script,*map(str,args)]
    return [PYTHON,'scripts/run_on_ada.py','--',*command] if gpu else command

def complete(path):
    try:return read(path).get('status')=='complete'
    except (FileNotFoundError,ValueError):return False

def stage(name,commands,outputs,adopt=False):
    global child
    directory=ROOT/'runs/stages';directory.mkdir(parents=True,exist_ok=True);receipt=directory/(name+'.json')
    signature=fingerprint(commands)
    inputs=stage_inputs(name,commands);input_signature=fingerprint(inputs)
    if receipt.exists():
        old=read(receipt)
        if old['command_hash']!=signature:raise RuntimeError('Restart command changed for '+name)
        if set(old['outputs'])!=set(outputs) or not all((ROOT/p).is_file() and sha256_file(p)==h for p,h in old['outputs'].items()):
            raise RuntimeError('Completed stage output changed: '+name+'; preserve and audit before resuming')
        if old.get('schema_version',1)<2:
            # Preserve exact legacy bytes. Native receipts prove their recorded
            # core inputs; expanded dependencies are first pinned now, explicitly.
            evidence=native_adoption_evidence(name)
            original=receipt.read_bytes();old_hash=sha256_file(receipt)
            archive=directory/'migration_originals'/(name+'.'+old_hash+'.json')
            if archive.exists() and archive.read_bytes()!=original:raise RuntimeError('Migration archive conflict')
            if not archive.exists():atomic_bytes(archive,original)
            migration={'migrated_utc':utc_now(),'original_receipt':str(archive.relative_to(ROOT)),
                       'original_receipt_sha256':old_hash,'evidence':evidence,
                       'reason':'Legacy receipt had output and command hashes but no stage input signature'}
            updated={**old,'schema_version':2,'inputs':inputs,'input_hash':input_signature,
                     'migration':migration,'supervisor_sha256':sha256_file('scripts/run_research.py')}
            atomic_json(receipt,updated)
            ledger.append('horizon_receipt_migrated',stage=name,original_receipt_sha256=old_hash,input_hash=input_signature,evidence=evidence)
            print(json.dumps({'stage':name,'status':'verified_legacy_migration','original_receipt':migration['original_receipt']}),flush=True)
            return
        if old.get('input_hash')!=input_signature or old.get('inputs')!=inputs:
            changed=sorted(p for p in set(old.get('inputs',{}))|set(inputs) if old.get('inputs',{}).get(p)!=inputs.get(p))
            raise RuntimeError('Completed stage inputs changed: '+name+' '+repr(changed)+'; preserve and audit before resuming')
        print(json.dumps({'stage':name,'status':'verified_skip'}),flush=True);return
    if stopping:raise SystemExit(75)
    evidence=native_adoption_evidence(name) if adopt else None
    if not adopt:
        log=ROOT/'logs'/('horizon_'+name+'.log');log.parent.mkdir(parents=True,exist_ok=True)
        for number,command in enumerate(commands):
            atomic_json('runs/horizon_status.json',{'status':'running','stage':name,'command_number':number,'command':command,'log':str(log.relative_to(ROOT)),'updated_utc':utc_now()})
            ledger.append('horizon_stage_started',stage=name,command_number=number,command=command,input_hash=input_signature)
            with log.open('ab',buffering=0) as stream:
                child=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,env=os.environ.copy(),start_new_session=True)
                worker_group=child.pid;status=child.wait()
                if stopping:wait_for_owned_group(worker_group)
                child=None
            if status!=0:
                atomic_json('runs/horizon_status.json',{'status':'interrupted' if stopping or status==75 else 'failed','stage':name,'returncode':status,'log':str(log.relative_to(ROOT)),'updated_utc':utc_now()})
                ledger.append('horizon_stage_stopped',stage=name,returncode=status)
                raise SystemExit(status if status>0 else 1)
    if stage_inputs(name,commands)!=inputs:raise RuntimeError('Stage inputs changed while running '+name+'; do not commit completion')
    hashes={p:sha256_file(p) for p in outputs}
    atomic_json(receipt,{'schema_version':2,'stage':name,'command_hash':signature,'commands':commands,'outputs':hashes,
                         'inputs':inputs,'input_hash':input_signature,'supervisor_sha256':sha256_file('scripts/run_research.py'),
                         'adopted_verified_existing_artifacts':adopt,'adoption_evidence':evidence,'completed_utc':utc_now()})
    ledger.append('horizon_stage_completed',stage=name,outputs=hashes,adopted=adopt,input_hash=input_signature)
    print(json.dumps({'stage':name,'status':'complete'}),flush=True)

def reviewed_completion(root=None):
    """Retain completed status only while the hash-bound final review stays valid."""
    root=Path(root or ROOT)
    required={'manuscript/main_paper.pdf','manuscript/supplement.pdf','manuscript/build_manifest.json',
              'reports/main_pdf_visual_review.json','reports/supplement_pdf_visual_review.json',
              'reports/final_integrity_audit.json','results/tables/final_real_results.json','runs/test_freeze.json'}
    try:
        record=json.loads((root/'runs/research_complete.json').read_text())
        artifacts=record['artifacts']
        if record['status']!='complete' or not required.issubset(artifacts):return False
        for name,digest in artifacts.items():
            path=(root/name).resolve()
            if not path.is_relative_to(root.resolve()) or not path.is_file() or sha256_file(path)!=digest:return False
        build=json.loads((root/'manuscript/build_manifest.json').read_text())
        return build.get('visual_inspection_complete') is True and build.get('page_counts')=={'main_paper.pdf':8,'supplement.pdf':4}
    except (OSError,ValueError,KeyError,TypeError):return False

def policy_outputs(split):return [f'results/raw/policies/{split}/{mode}_seed{s}/all.json' for mode in ('agent','agent_no_generation') for s in (17,29,43)]

def ensure_raw(split=None):
    target='data/manifests/real'+('_'+split if split else '')+'.jsonl'
    while not (ROOT/target).exists():
        if stopping:raise SystemExit(75)
        if download_child is not None and download_child.poll() is not None:
            raise RuntimeError('Acquisition exited before publishing '+target+'; inspect logs/horizon_raw_download.log')
        atomic_json('runs/horizon_status.json',{'status':'waiting_for_raw_split','split':split or 'all','updated_utc':utc_now()})
        time.sleep(30)


def main():
    global download_child
    parser=argparse.ArgumentParser();parser.add_argument('--skip-download-if-active',action='store_true',help='Wait for current acquisition to publish all raw manifests; normal restart should omit this')
    args=parser.parse_args()
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,handle)
    with RunLock('runs/active.lock'):
        ledger.read(repair_tail=True)
        stage('gpu_identity',[invoke('scripts/verify_cuda.py')],['manifests/gpu_verification.json'],adopt=(ROOT/'manifests/gpu_verification.json').is_file())
        if not (ROOT/'data/manifests/real.jsonl').exists() and not args.skip_download_if_active:
            command=invoke('scripts/prepare_data.py','--stage','download','--workers','3',gpu=False)
            with (ROOT/'logs/horizon_raw_download.log').open('ab',buffering=0) as stream:
                download_child=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,env=os.environ.copy(),start_new_session=True)
            ledger.append('horizon_background_acquisition_started',pid=download_child.pid)
        # Already completed artifacts are individually checked by their native receipts or consumers.
        stage('background_flow',[invoke('scripts/train_flow.py')],['checkpoints/background_flow/final.pt','manifests/background_flow_validation.json'],adopt=(ROOT/'runs/background_flow_complete.json').exists())
        temporal=[f'checkpoints/synthetic_temporal/seed_{s}/best.pt' for s in (17,29,43)]
        stage('synthetic_training',[invoke('scripts/train_synthetic_detector.py')],temporal,adopt=all((ROOT/f'runs/synthetic_temporal_seed_{s}_complete.json').exists() for s in (17,29,43)))
        stage('synthetic_train_features',[invoke('scripts/generate_synthetic_features.py','--split','train','development')],[f'data/features/synthetic/{s}/index.json' for s in ('train','development')],adopt=all(complete(f'data/features/synthetic/{s}/index.json') for s in ('train','development')))
        stage('supplemental_training',[invoke('scripts/supplemental_views.py','--stage','train')],['manifests/supplemental_model_index.json','manifests/supplemental_feature_cost.json'],adopt=complete('manifests/supplemental_model_index.json'))
        for split in ('train','development'):
            ensure_raw(split)
            stage('features_'+split,[invoke('scripts/extract_real_features.py','--split',split)],[f'data/features/real/{split}/index.json'])
            stage('foundation_'+split,[invoke('scripts/extract_foundation.py','--split',split)],[f'data/features/foundation/{split}/index.json'])
        stage('real_training',[invoke('scripts/train_detectors.py'),invoke('scripts/train_augmented.py')],['manifests/detector_index.json'])
        for split in ('development','calibration'):
            if split=='calibration':
                ensure_raw(split)
                stage('features_'+split,[invoke('scripts/extract_real_features.py','--split',split)],[f'data/features/real/{split}/index.json'])
                stage('foundation_'+split,[invoke('scripts/extract_foundation.py','--split',split)],[f'data/features/foundation/{split}/index.json'])
            for probe,folder in [('central','data/interventions'),('full','data/interventions_full')]:
                stage('interventions_'+split+'_'+probe,[invoke('scripts/generate_interventions.py','--split',split,'--probe',probe)],[f'{folder}/{split}/index.json'])
            for mode in ('agent','agent_no_generation'):
                outputs=[p for p in policy_outputs(split) if '/'+mode+'_seed' in p]
                stage('policies_'+split+'_'+mode,[invoke('scripts/run_policies.py','--split',split,'--mode',mode)],outputs)
        stage('calibration',[invoke('scripts/analyze_real.py','--stage','calibrate')],['manifests/calibration.json','manifests/final_development_selection.json'])
        stage('live_benchmark',[invoke('scripts/benchmark_live.py')],['results/tables/runtime_benchmark.json'])
        ensure_raw()
        stage('raw_acquisition',[invoke('scripts/prepare_data.py','--stage','download','--workers','3',gpu=False)],['data/manifests/real.jsonl']+[f'data/manifests/real_{s}.jsonl' for s in ('train','development','calibration','test')])
        stage('freeze',[invoke('scripts/freeze_protocol.py',gpu=False)],['runs/test_freeze.json'])
        from uac.protocol import require_frozen_test
        require_frozen_test()
        stage('features_test',[invoke('scripts/extract_real_features.py','--split','test','--allow-test-features')],['data/features/real/test/index.json'])
        stage('foundation_test',[invoke('scripts/extract_foundation.py','--split','test','--allow-test-features')],['data/features/foundation/test/index.json'])
        for probe,folder in [('central','data/interventions'),('full','data/interventions_full')]:
            stage('interventions_test_'+probe,[invoke('scripts/generate_interventions.py','--split','test','--probe',probe)],[f'{folder}/test/index.json'])
        for mode in ('agent','agent_no_generation'):
            stage('policies_test_'+mode,[invoke('scripts/run_policies.py','--split','test','--mode',mode)],[p for p in policy_outputs('test') if '/'+mode+'_seed' in p])
        stage('real_test',[invoke('scripts/analyze_real.py','--stage','test')],['results/tables/real_results.json'])
        stage('robustness',[invoke('scripts/evaluate_robustness.py')],['results/tables/robustness.json'])
        # Recheck the full global freeze immediately before this historical evaluator.
        require_frozen_test()
        stage('synthetic_test',[invoke('scripts/evaluate_synthetic.py','--split','test')],['reports/synthetic_results.json'])
        stage('synthetic_test_features',[invoke('scripts/generate_synthetic_features.py','--split','test','heldout_test')],['data/features/synthetic/test/index.json','data/features/synthetic/heldout_test/index.json'])
        stage('supplemental_test',[invoke('scripts/supplemental_views.py','--stage','evaluate')],['results/tables/supplemental_views.json'])
        stage('semisynthetic_test',[invoke('scripts/evaluate_semisynthetic.py')],['results/tables/semisynthetic_results.json'])
        stage('final_assembly',[invoke('scripts/assemble_results.py',gpu=False)],['results/tables/final_real_results.json'])
        stage('manuscript',[invoke('scripts/build_results_report.py','--real','results/tables/final_real_results.json',gpu=False)],['manuscript/main_paper.pdf','manuscript/supplement.pdf'])
        status='complete' if reviewed_completion() else 'experiments_and_drafts_complete_visual_review_pending'
        atomic_json('runs/horizon_status.json',{'status':status,'updated_utc':utc_now()})
        ledger.append('horizon_completion_reverified' if status=='complete' else 'horizon_execution_complete_visual_review_pending')
        print(json.dumps({'status':status}),flush=True)

if __name__=='__main__':
    try:main()
    finally:
        if download_child is not None and download_child.poll() is None:
            try:os.killpg(download_child.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            wait_for_owned_group(download_child.pid)
            download_child.wait()
