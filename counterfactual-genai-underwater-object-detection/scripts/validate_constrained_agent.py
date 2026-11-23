#!/usr/bin/env python3
"""Actual Ada validation of legal-action constrained generation; no acoustic outcomes.

Includes the observed failure pattern (STFT observed, wavelet acquired, then an
unavailable STFT temptation), valid/invalid verification states, and mixed-length
left-padded batches with stop-only rows. Schema validity is by construction.
"""
from __future__ import annotations
import copy,json,sys,time
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'src'))
from uac.agent import BoundedQwenController,parse_action
from uac.agent_batch import select_actions,CONSTRAINT_VERSION
from uac.runtime import RunLock,atomic_bytes,atomic_json,EventLedger,sha256_file,source_fingerprint,utc_now


def fixture(kind,index):
    probability=[.05,.2,.5,.8,.95][index%5]
    state={'candidates':[{'candidate_id':'candidate0','interval_seconds':[.75,2.25],
      'base_probability':probability,'uncertainty':1-2*abs(probability-.5)}],
      'observations':{'candidate0':{'stft':{'detector_logit':-1.234567890123456,'presence_probability':probability}}},
      'history':[],'available_tools':['fourier','wavelet','stop','verify'],'remaining_actions':3,'remaining_verifications':1}
    if kind!='initial':
        state['observations']['candidate0']['wavelet']={'detector_logit':3.234567890123456,'presence_probability':.9621234567890123}
        state['history'].append({'action':'wavelet','candidate_id':'candidate0','valid':True})
        state['available_tools'].remove('wavelet');state['remaining_actions']-=1
    if kind in ('verification_valid','verification_invalid','stop_only'):
        state['observations']['candidate0']['verify']={'valid':kind!='verification_invalid',
          'view_delta_medians':{'stft':-2.1234567890123456,'wavelet':3.2345678901234567},'background_penalty':.12345678901234567}
        state['history'].append({'action':'verify','candidate_id':'candidate0','valid':True})
        state['available_tools'].remove('verify');state['remaining_actions']-=1;state['remaining_verifications']=0
    if kind=='stop_only':state['available_tools']=['stop']
    return state


def main():
    import torch
    target=PROJECT/'manifests/agent_constrained_validation.json'
    with RunLock('runs/training.lock'):
        if target.exists():atomic_bytes(PROJECT/'manifests/archive'/f'agent_constrained_validation.{time.time_ns()}.json',target.read_bytes())
        controller=BoundedQwenController();records=[];started=time.perf_counter()
        attempt=PROJECT/'runs/agent_constrained_validation'/str(time.time_ns())
        variants=['initial','stft_then_wavelet','verification_valid','verification_invalid','mixed_finished_rows']
        mixed=['initial','stft_then_wavelet','verification_valid','verification_invalid','stop_only']
        for variant in variants:
            states=[fixture(mixed[i%len(mixed)] if variant=='mixed_finished_rows' else variant,i) for i in range(32)]
            torch.cuda.reset_peak_memory_stats();calls=select_actions(controller,states)
            parsed=[parse_action(call['raw_text'],state) for call,state in zip(calls,states)]
            checks={'all_schema_valid':all(call['valid'] for call in calls),
              'all_terminal_eos':all(call['terminal_eos_observed'] for call in calls),
              'independent_parser_agreement':all(a['valid'] and a['action']==b['action'] and a['candidate_id']==b['candidate_id'] for a,b in zip(parsed,calls)),
              'all_marked_constrained':all(call['constrained_decoding']==CONSTRAINT_VERSION and call['schema_validity_by_construction'] for call in calls),
              'within_output_budget':all(call['output_tokens']<=controller.max_output_tokens for call in calls)}
            record={'variant':variant,'checks':checks,'batch_size':32,'max_input_tokens':max(c['input_tokens'] for c in calls),
              'max_output_tokens':max(c['output_tokens'] for c in calls),'peak_memory_mib':torch.cuda.max_memory_allocated()/1024**2,
              'batch_wall_seconds':calls[0]['batch_wall_seconds'],'states':states,'calls':calls}
            path=attempt/(variant+'.json');atomic_json(path,record)
            record['artifact']={'path':str(path.relative_to(PROJECT)),'sha256':sha256_file(path)};records.append(record)
            print(json.dumps({'variant':variant,**checks,'seconds':record['batch_wall_seconds'],'peak_memory_mib':record['peak_memory_mib']}),flush=True)
        passed=all(all(row['checks'].values()) for row in records)
        result={'status':'passed' if passed else 'failed','created_utc':utc_now(),'constrained_decoding':CONSTRAINT_VERSION,
          'schema_validity_by_construction':True,'scope':'160 protocol fixtures including the actual unavailable-STFT failure pattern; no acoustic performance or unconstrained competence claim',
          'states':sum(len(r['calls']) for r in records),'valid_fraction':sum(c['valid'] for r in records for c in r['calls'])/160,
          'batch_size':32,'elapsed_seconds_excluding_load':time.perf_counter()-started,'model_load_seconds':controller.load_seconds,
          'peak_memory_mib':max(r['peak_memory_mib'] for r in records),'memory_scope':'Includes retained constrained generation score tensors used for token log-probability logging',
          'code_fingerprint':source_fingerprint(['src/uac/agent.py','src/uac/agent_batch.py','scripts/validate_constrained_agent.py']),
          'model_revision':controller.revision,'records':records}
        atomic_json(target,result);EventLedger().append('constrained_agent_fixture_validation',status=result['status'],states=160,valid_fraction=result['valid_fraction'],constrained_decoding=CONSTRAINT_VERSION,schema_validity_by_construction=True)
        controller.close();print(json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)
        return 0 if passed else 2

if __name__=='__main__':raise SystemExit(main())
