#!/usr/bin/env python3
"""Run frozen generative policies on numeric evidence; shard-level restart."""
import argparse,json,time
from pathlib import Path
import torch
from uac.agent import BoundedQwenController
from uac.agent_batch import select_actions,CONSTRAINT_VERSION
from uac.verification import load_real_cache,DetectorBank,intervention_components,verification_score,PRIMARY_GROUPS,SUBSET_NAMES
from uac.runtime import RunLock,StopRequested,EventLedger,atomic_json,sha256_file,fingerprint,source_fingerprint

p=argparse.ArgumentParser();p.add_argument('--split',required=True,choices=['development','calibration','test']);p.add_argument('--seeds',type=int,nargs='+',default=[17,29,43]);p.add_argument('--batch-size',type=int,default=32);p.add_argument('--mode',choices=['agent','agent_no_generation'],default='agent');p.add_argument('--max-batches',type=int);args=p.parse_args()
research=json.loads(Path('configs/research.json').read_text())
if args.batch_size!=research['policy_batch_size']:raise RuntimeError('Policy batch must match prospectively selected research configuration')
gate=json.loads(Path('manifests/agent_constrained_validation.json').read_text())
if gate.get('status')!='passed' or gate.get('constrained_decoding')!=CONSTRAINT_VERSION or gate.get('code_fingerprint')!=source_fingerprint(['src/uac/agent.py','src/uac/agent_batch.py','scripts/validate_constrained_agent.py']):raise RuntimeError('Constrained controller integration gate missing or stale')
if args.split=='test':
    from uac.protocol import require_frozen_test
    require_frozen_test()

def load_interventions(split,ids):
    index=json.loads((Path('data/interventions')/split/'index.json').read_text())
    if index['status']!='complete':raise RuntimeError('Intervention cache incomplete')
    parts=[]
    for shard in index['shards']:
        if sha256_file(shard['path'])!=shard['sha256']:raise RuntimeError('Corrupt intervention shard')
        parts.append(torch.load(shard['path'],map_location='cuda',weights_only=False))
    found=[x for row in parts for x in row['ids']]
    if found!=ids:raise RuntimeError('Intervention and real IDs differ')
    return {key:torch.cat([p[key] for p in parts]) for key in parts[0] if key!='ids'},index

with RunLock('runs/training.lock'),StopRequested() as stop:
    dataset=load_real_cache(args.split);training=load_real_cache('train')
    backgrounds=training['features'][training['labels']==0,:32]
    background_statistics={'mean':backgrounds.mean(0),'scale':backgrounds.std(0).clamp_min(1e-4)}
    del backgrounds,training
    cache,intervention_index=load_interventions(args.split,dataset['ids'])
    controller=BoundedQwenController();ledger=EventLedger()
    for seed in args.seeds:
        bank=DetectorBank(seed);view_logits=bank.view_logits(dataset['features']);components=intervention_components(bank,cache,background_statistics)
        settings={'split':args.split,'seed':seed,'mode':args.mode,'batch_size':args.batch_size,'data_fingerprint':dataset['index']['data_fingerprint'],'detectors_hash':sha256_file('manifests/detector_index.json'),'interventions_hash':fingerprint(intervention_index),'model_revision':controller.revision,'controller_decoding':CONSTRAINT_VERSION,'schema_validity_by_construction':True,'prompt_hash':controller.prompt_sha256,'code_hash':source_fingerprint(['scripts/run_policies.py','src/uac/agent.py','src/uac/agent_batch.py','src/uac/verification.py']),'initial_view':'stft','probe_seconds':[.75,2.25],'scoring_scope':'whole_record_presence','max_additional_actions':3,'max_verifications':1,'cost_scope':'model calls live; acoustic features and generator results cached replay'}
        directory=Path('results/raw/policies')/args.split/(args.mode+'_seed'+str(seed));directory.mkdir(parents=True,exist_ok=True)
        records=[];new_batches=0
        for begin in range(0,len(dataset['ids']),args.batch_size):
            end=min(begin+args.batch_size,len(dataset['ids']));path=directory/f'batch-{begin:06d}.json'
            receipt=path.with_suffix('.receipt.json')
            if path.exists() or receipt.exists():
                saved=certificate=None
                try:
                    saved=json.loads(path.read_text());certificate=json.loads(receipt.read_text())
                except (OSError,ValueError):pass
                if saved is not None and saved.get('config_hash')!=fingerprint(settings):raise RuntimeError('Policy cache configuration changed')
                if saved is not None and certificate is not None and certificate.get('config_hash')==fingerprint(settings) and certificate.get('sha256')==sha256_file(path):
                    if [r['id'] for r in saved['records']]!=dataset['ids'][begin:end]:raise RuntimeError('Policy batch identity mismatch')
                    records.extend(saved['records']);continue
                stamp='.corrupt-'+str(time.time_ns())
                for damaged in (path,receipt):
                    if damaged.exists():damaged.rename(damaged.with_name(damaged.name+stamp))
                ledger.append('policy_batch_quarantined_for_replay',split=args.split,mode=args.mode,seed=seed,begin=begin)
            if stop.requested or (args.max_batches is not None and new_batches>=args.max_batches):
                atomic_json(directory/'index.json',{'status':'partial','config':settings,'completed':len(records),'expected':len(dataset['ids'])});raise SystemExit(75 if stop.requested else 0)
            states=[];traces=[[] for _ in range(begin,end)];views=[['stft'] for _ in range(begin,end)];verified=[False]*(end-begin);active=list(range(end-begin))
            for idx in range(begin,end):
                probability=float(torch.sigmoid(view_logits[idx,1]))
                states.append({'candidates':[{'candidate_id':'candidate0','interval_seconds':[.75,2.25],'base_probability':probability,'uncertainty':1-2*abs(probability-.5)}],
                 'observations':{'candidate0':{'stft':{'detector_logit':float(view_logits[idx,1]),'presence_probability':probability}}},
                 'history':[],'available_tools':['fourier','wavelet','stop']+(['verify'] if args.mode=='agent' else []),'remaining_actions':3,'remaining_verifications':int(args.mode=='agent')})
            for round_id in range(3):
                if not active:break
                actions=select_actions(controller,[states[i] for i in active]);next_active=[]
                for local,action in zip(active,actions):
                    traces[local].append(action);state=states[local];tool=action['action'];idx=begin+local
                    if tool=='stop' or not action['valid']:continue
                    state['history'].append({'action':tool,'candidate_id':'candidate0','valid':True});state['remaining_actions']-=1
                    state['available_tools'].remove(tool)
                    if tool=='verify':
                        verified[local]=True;state['remaining_verifications']=0
                        observed={v:float(components['delta'][idx,:,PRIMARY_GROUPS.index(v)].median()) for v in views[local]}
                        state['observations']['candidate0']['verify']={'valid':bool(components['valid'][idx]),'view_delta_medians':observed,'background_penalty':float(components['penalty'][idx])}
                    else:
                        views[local].append(tool);position=PRIMARY_GROUPS.index(tool)
                        state['observations']['candidate0'][tool]={'detector_logit':float(view_logits[idx,position]),'presence_probability':float(torch.sigmoid(view_logits[idx,position]))}
                        if verified[local]:state['observations']['candidate0']['verify']['view_delta_medians'][tool]=float(components['delta'][idx,:,position].median())
                    current=float(torch.sigmoid(bank.subset_logit(dataset['features'][idx:idx+1],views[local])[0]))
                    state['candidates'][0]['base_probability']=current
                    state['candidates'][0]['uncertainty']=1-2*abs(current-.5)
                    if state['remaining_actions']>0:next_active.append(local)
                active=next_active
            batch_records=[]
            for local,idx in enumerate(range(begin,end)):
                selected=views[local];base=float(bank.subset_logit(dataset['features'][idx:idx+1],selected)[0])
                positions=[PRIMARY_GROUPS.index(v) for v in selected]
                delta=components['delta'][idx,:,positions].flatten()
                record={'id':dataset['ids'][idx],'seed':seed,'mode':args.mode,'views':selected,'subset_method':SUBSET_NAMES[frozenset(selected)],'verified':verified[local],'intervention_valid':bool(components['valid'][idx]),'base_logit':base,'delta':delta.cpu().tolist() if verified[local] else [],'penalty':float(components['penalty'][idx]) if verified[local] else 0.,'trace':traces[local],'amortized_llm_seconds':sum(t['amortized_seconds'] for t in traces[local]),'measurement_scope':'cached acoustic/generator replay with actual LLM calls; not end-to-end runtime'}
                batch_records.append(record)
            atomic_json(path,{'config_hash':fingerprint(settings),'records':batch_records})
            atomic_json(receipt,{'config_hash':fingerprint(settings),'sha256':sha256_file(path),'begin':begin,'end':end})
            records.extend(batch_records);new_batches+=1
            atomic_json(directory/'index.json',{'status':'running','config':settings,'completed':len(records),'expected':len(dataset['ids'])})
            ledger.append('policy_batch_complete',split=args.split,mode=args.mode,seed=seed,through=end)
            print(json.dumps({'split':args.split,'mode':args.mode,'seed':seed,'completed':end,'expected':len(dataset['ids'])}),flush=True)
        atomic_json(directory/'all.json',{'config':settings,'records':records});atomic_json(directory/'index.json',{'status':'complete','config':settings,'completed':len(records),'expected':len(dataset['ids']),'records_path':str(directory/'all.json'),'sha256':sha256_file(directory/'all.json')})
        del bank
    controller.close()
