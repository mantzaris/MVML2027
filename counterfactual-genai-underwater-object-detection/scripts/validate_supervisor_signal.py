#!/usr/bin/env python3
"""Control-plane fixture: group stop reaches a launcher grandchild checkpoint."""
import json,os,signal,subprocess,sys,time
from pathlib import Path
from uac.runtime import atomic_json,sha256_file,EventLedger
root=Path(__file__).resolve().parents[1];folder=root/'tmp/supervisor_signal_fixture';folder.mkdir(parents=True,exist_ok=True)
ready=folder/'ready.json';saved=folder/'checkpoint.json'
for path in (ready,saved):
    if path.exists():path.unlink()
worker=folder/'worker.py';worker.write_text("import os,signal,time\nfrom uac.runtime import atomic_json\nstop=False\ndef handle(s,f):\n global stop\n stop=True\nsignal.signal(signal.SIGTERM,handle)\natomic_json('tmp/supervisor_signal_fixture/ready.json',{'pid':os.getpid()})\nwhile not stop:time.sleep(.02)\natomic_json('tmp/supervisor_signal_fixture/checkpoint.json',{'stop_saved':True,'pid':os.getpid()})\n")
wrapper='import subprocess,sys;subprocess.run([sys.executable,sys.argv[1]],check=False)'
process=subprocess.Popen([sys.executable,'-c',wrapper,str(worker)],cwd=root,start_new_session=True)
deadline=time.monotonic()+10
while not ready.exists() and time.monotonic()<deadline:time.sleep(.02)
if not ready.exists():os.killpg(process.pid,signal.SIGKILL);raise RuntimeError('Fixture did not start')
os.killpg(process.pid,signal.SIGTERM);status=process.wait(timeout=10)
while not saved.exists() and time.monotonic()<deadline:time.sleep(.02)
if not saved.exists() or not json.loads(saved.read_text())['stop_saved']:raise RuntimeError('Grandchild failed to save checkpoint on group stop')
result={'status':'passed','scope':'CPU control-plane signaling only; no ML/GPU computation','wrapper_exit':status,'grandchild_checkpoint_saved':True,'artifact':'tmp/supervisor_signal_fixture/checkpoint.json','sha256':sha256_file(saved),'source_sha256':sha256_file('scripts/validate_supervisor_signal.py'),'supervisor_sha256':sha256_file('scripts/run_research.py')}
atomic_json('manifests/supervisor_signal_validation.json',result);EventLedger().append('supervisor_signal_validation_passed',checkpoint_sha256=result['sha256']);print(json.dumps(result))
