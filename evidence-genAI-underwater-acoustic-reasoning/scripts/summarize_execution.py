"""Compact execution ledger; detailed per-job records remain outside Git."""
import argparse,csv,json
from datetime import datetime,timezone
from pathlib import Path
from collections import Counter
from underwater_reasoning.io import atomic_json,file_hash

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--finished',action='store_true');args=parser.parse_args()
    now=datetime.now(timezone.utc);config=json.loads((ROOT/'configs/execution.json').read_text())
    start=datetime.fromisoformat(config['research_accounting_start_utc'].replace('Z','+00:00'))
    jobs=[]
    for path in sorted((ROOT/'results/manifests/jobs').glob('*.json')):
        record=json.loads(path.read_text())
        jobs.append({'record':path.name,**{k:v for k,v in record.items() if k in ('name','attempt','started_utc','ended_utc','elapsed_seconds','status','returncode','command')},'sha256':file_hash(path)})
    counts=Counter();tokens=Counter();limits=Counter();raw_bytes=0
    for path in (ROOT/'results/raw').rglob('*.json'):
        try:record=json.loads(path.read_text())
        except ValueError:continue
        if not isinstance(record,dict) or not isinstance(record.get('result'),dict) or 'raw' not in record['result']:continue
        family='/'.join(path.relative_to(ROOT/'results/raw').parts[:2]);result=record['result']
        counts[family]+=1;tokens[family]+=result.get('output_tokens',0);limits[family]+=bool(result.get('hit_output_limit'));raw_bytes+=path.stat().st_size
    ledger={'recorded_at_utc':now.isoformat(),'finished':args.finished,'research_accounting_start_utc':start.isoformat(),
        'conservative_research_elapsed_hours':(now-start).total_seconds()/3600,'authorized_ceiling_hours':10,
        'within_authorized_window':(now-start).total_seconds()<=36000,'pod_finish_instruction':'leave_running',
        'pod_stop_requested':False,'actual_pod_billing_hours':None,'actual_usd_cost':None,'guarded_stages':jobs,
        'saved_inference_records':dict(counts),'saved_output_tokens':dict(tokens),'output_limit_hits':dict(limits),
        'raw_record_bytes_excluded_from_git':raw_bytes,
        'counting_note':'Counts cover saved individual inference records; smoke batches and planner calls may use separate aggregate files, and interrupted batches may consume compute without a saved output. Guarded elapsed times include loading and CPU stages, not measured GPU utilization.'}
    if args.finished and any(j.get('status')=='running' for j in jobs):raise ValueError('A guarded stage still has running status')
    if args.finished and not ledger['within_authorized_window']:raise ValueError('Research exceeded authorization window')
    atomic_json(ROOT/'results/manifests/execution_ledger.json',ledger)
    with (ROOT/'results/tables/cost_ledger.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['stage','started_utc','ended_utc','elapsed_seconds','status','actual_usd_cost','note']);writer.writeheader()
        for job in jobs:writer.writerow({'stage':job.get('name',job['record']),**{k:job.get(k) for k in ('started_utc','ended_utc','elapsed_seconds','status')},
            'actual_usd_cost':'','note':'Stage wall time includes CPU/loading; do not sum as allocated GPU time. Pod rate and prior billing unknown.'})
    print(json.dumps({k:v for k,v in ledger.items() if k!='guarded_stages'},indent=2))

if __name__=='__main__':main()
