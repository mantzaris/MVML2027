"""Record the live worker after experiments and retain ignored diagnostic logs."""
import importlib.metadata
import json
import shutil
import subprocess
from datetime import datetime,timezone
from pathlib import Path
from underwater_reasoning.io import atomic_json

ROOT=Path(__file__).resolve().parents[1]
jobs=[json.loads(p.read_text()) for p in (ROOT/'results/manifests/jobs').glob('*.json')]
if any(j.get('status')=='running' for j in jobs):raise SystemExit('An experiment is still marked running')
processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip()
if processes:raise SystemExit('GPU compute processes remain; inspect before recording completion')
logs=ROOT/'results/logs/worker';logs.mkdir(parents=True,exist_ok=True)
for source in (ROOT/'.cache').glob('*.log'):shutil.copy2(source,logs/source.name)
versions={}
for name in ('torch','torchvision','torchaudio','transformers','xgrammar','apache-tvm-ffi','numpy','scipy','soundfile','scikit-learn'):
    try:versions[name]=importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:versions[name]=None
disk=shutil.disk_usage(ROOT)
record={'observed_at_utc':datetime.now(timezone.utc).isoformat(),'pod_accessible_over_ssh':True,
    'gpu_compute_processes':0,'pod_stop_requested':False,'pod_termination_requested':False,'finish_instruction':'leave_running',
    'gpu_snapshot':subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip(),
    'disk_total_bytes':disk.total,'disk_free_bytes':disk.free,'package_versions':versions,
    'detailed_logs_preserved_outside_git':len(list(logs.glob('*.log')))}
atomic_json(ROOT/'results/manifests/worker_finish.json',record)
print(json.dumps(record,indent=2))
