#!/usr/bin/env python3
"""Formatting/runtime pilot for batched controller calls, not detection evidence."""
import json
from uac.agent import BoundedQwenController
from uac.agent_batch import select_actions
from uac.runtime import atomic_json,RunLock,EventLedger,source_fingerprint
from pilot_agent import fixture_states
with RunLock('runs/training.lock'):
    controller=BoundedQwenController();states=fixture_states();rows=[]
    for begin in range(0,len(states),8):
        rows.extend(select_actions(controller,states[begin:begin+8]))
        atomic_json('manifests/agent_batch_pilot.partial.json',{'records':rows,'expected':len(states)})
    fraction=sum(row['valid'] for row in rows)/len(rows)
    result={'status':'passed' if fraction>=.98 else 'failed','valid_fraction':fraction,'records':rows,'n':len(rows),'amortized_seconds_total':sum(r['amortized_seconds'] for r in rows),'peak_memory_mib':max(r['peak_allocated_mib'] for r in rows),'code_hash':source_fingerprint(['src/uac/agent.py','src/uac/agent_batch.py','scripts/validate_agent_batch.py']),'scope':'Protocol fixtures only; no acoustic detection performance or end-to-end acquisition time.'}
    atomic_json('manifests/agent_batch_pilot.json',result);EventLedger().append('batched_agent_protocol_pilot',status=result['status'],valid_fraction=fraction,n=len(rows),seconds=result['amortized_seconds_total'])
    print(json.dumps({key:value for key,value in result.items() if key!='records'}),flush=True);controller.close()
    if fraction<.98:raise SystemExit(2)
