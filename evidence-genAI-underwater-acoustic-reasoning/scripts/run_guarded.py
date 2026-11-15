"""Run one experiment stage under the authorized wall-clock deadline."""
import argparse,json,os,signal,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
from underwater_reasoning.io import atomic_json
from underwater_reasoning.runtime import budget_check,DEADLINE
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--name',required=True);parser.add_argument('command',nargs=argparse.REMAINDER);a=parser.parse_args()
    if not a.command:raise ValueError('A concrete command is required')
    left=budget_check();start=datetime.now(timezone.utc);t=time.monotonic();path=ROOT/f'results/manifests/jobs/{a.name}.json'
    attempt=1
    while path.exists():
        attempt+=1;path=ROOT/f'results/manifests/jobs/{a.name}_attempt{attempt}.json'
    state={'name':a.name,'attempt':attempt,'command':a.command,'started_utc':start.isoformat(),'deadline_utc':DEADLINE.isoformat(),'pod_stop_requested':False,'status':'running'};atomic_json(path,state)
    process=subprocess.Popen(a.command,cwd=ROOT,start_new_session=True)
    try:code=process.wait(timeout=max(1,left-180))
    except subprocess.TimeoutExpired:
        os.killpg(process.pid,signal.SIGTERM)
        try:process.wait(timeout=30)
        except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
        code=124
    state.update(status='complete' if code==0 else 'failed_or_deadline',returncode=code,ended_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-t)
    atomic_json(path,state);raise SystemExit(code)
