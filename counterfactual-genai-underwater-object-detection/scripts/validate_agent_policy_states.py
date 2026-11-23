#!/usr/bin/env python3
"""Pre-outcome production-shaped state stress check; numeric fixtures only."""
import copy,json
import torch
from uac.agent import BoundedQwenController
from uac.agent_batch import select_actions
from uac.runtime import RunLock,atomic_json,EventLedger
with RunLock('runs/training.lock'):
    controller=BoundedQwenController();records=[]
    for variant in range(4):
        states=[]
        for i in range(32):
            p=.1234567890123456+(i%4)*.2
            state={'candidates':[{'candidate_id':'candidate0','interval_seconds':[.75,2.25],'base_probability':p,'uncertainty':1-2*abs(p-.5)}],'observations':{'candidate0':{'stft':{'detector_logit':-1.234567890123456,'presence_probability':p}}},'history':[],'available_tools':['fourier','wavelet','stop','verify'],'remaining_actions':3,'remaining_verifications':1}
            acts=[[],['fourier'],['fourier','wavelet'],['fourier','verify']][variant]
            for tool in acts:
                state['history'].append({'action':tool,'candidate_id':'candidate0','valid':True});state['available_tools'].remove(tool);state['remaining_actions']-=1
                if tool=='verify':
                    state['remaining_verifications']=0;state['observations']['candidate0']['verify']={'valid':True,'view_delta_medians':{'stft':-2.1234567890123456,'fourier':3.2345678901234567},'background_penalty':.12345678901234567}
                else:state['observations']['candidate0'][tool]={'detector_logit':3.2345678901234567,'presence_probability':.9621234567890123}
            states.append(state)
        torch.cuda.reset_peak_memory_stats();calls=select_actions(controller,states)
        records.append({'variant':variant,'batch_size':32,'valid_fraction':sum(r['valid'] for r in calls)/len(calls),'max_input_tokens':max(r['input_tokens'] for r in calls),'peak_memory_mib':torch.cuda.max_memory_allocated()/1024**2,'calls':calls})
    valid=sum(r['valid'] for row in records for r in row['calls'])/128
    result={'status':'passed' if valid>=.98 else 'failed','scope':'Production-shaped numeric fixtures only; no acoustic outcomes','batch_size':32,'states':128,'valid_fraction':valid,'max_input_tokens':max(r['max_input_tokens'] for r in records),'peak_memory_mib':max(r['peak_memory_mib'] for r in records),'records':records}
    atomic_json('manifests/agent_policy_state_validation.json',result);EventLedger().append('production_shaped_agent_state_validation',**{k:v for k,v in result.items() if k!='records'})
    print(json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True);controller.close()
    if valid<.98:raise SystemExit(2)
