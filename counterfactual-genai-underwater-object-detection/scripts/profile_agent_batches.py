#!/usr/bin/env python3
"""Pre-outcome throughput/format profiling on repeated protocol fixtures only."""
import json
import torch
from uac.agent import BoundedQwenController
from uac.agent_batch import select_actions
from uac.runtime import RunLock,atomic_json,EventLedger,source_fingerprint
from pilot_agent import fixture_states
with RunLock('runs/training.lock'):
    controller=BoundedQwenController();fixtures=fixture_states();states=[fixtures[i%len(fixtures)] for i in range(128)];records=[]
    select_actions(controller,states[:8])
    for size in (8,32,64):
        torch.cuda.reset_peak_memory_stats();calls=[]
        for begin in range(0,len(states),size):calls.extend(select_actions(controller,states[begin:begin+size]))
        row={'batch_size':size,'states':len(states),'valid_fraction':sum(r['valid'] for r in calls)/len(calls),'amortized_seconds_per_state':sum(r['amortized_seconds'] for r in calls)/len(calls),'peak_memory_mib':torch.cuda.max_memory_allocated()/1024**2,'traces':calls}
        records.append(row);atomic_json('manifests/agent_batch_scaling.json',{'scope':'Repeated protocol fixtures only; no acoustic outcomes','records':records,'code_hash':source_fingerprint(['scripts/profile_agent_batches.py','src/uac/agent.py','src/uac/agent_batch.py'])})
        print(json.dumps({k:v for k,v in row.items() if k!='traces'}),flush=True)
    controller.close();EventLedger().append('agent_batch_scaling_completed',records=[{k:v for k,v in r.items() if k!='traces'} for r in records])
