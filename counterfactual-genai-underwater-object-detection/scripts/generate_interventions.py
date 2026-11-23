#!/usr/bin/env python3
"""Create immutable generated-evidence shards, without detector/test labels."""
import argparse,json,time
from pathlib import Path
import torch
from uac.verification import load_flow,load_real_cache,generate_intervention_features
from uac.training import save_torch
from uac.runtime import RunLock,StopRequested,EventLedger,atomic_json,sha256_file,source_fingerprint,fingerprint
p=argparse.ArgumentParser();p.add_argument('--split',choices=['development','calibration','test'],required=True);p.add_argument('--batch-size',type=int,default=8);p.add_argument('--shard-size',type=int,default=64);p.add_argument('--probe',choices=['central','full'],default='central');args=p.parse_args()
if args.split=='test':
    from uac.protocol import require_frozen_test
    require_frozen_test()
config=json.loads(Path('configs/research.json').read_text())
with RunLock('runs/training.lock'),StopRequested() as stop:
    dataset=load_real_cache(args.split);flow=load_flow();directory=Path('data/interventions' if args.probe=='central' else 'data/interventions_full')/args.split;directory.mkdir(parents=True,exist_ok=True)
    settings={'split':args.split,'probe':args.probe,'probe_seconds':[.75,2.25] if args.probe=='central' else [0.,3.],'k':config['counterfactual_samples'],'steps':config['flow_sampling_steps'],'batch_size':args.batch_size,'shard_size':args.shard_size,'generator_hash':sha256_file('checkpoints/background_flow/final.pt'),'data_fingerprint':dataset['index']['data_fingerprint'],'feature_index_hash':sha256_file(Path('data/features/real')/args.split/'index.json'),'code_hash':source_fingerprint(['src/uac/verification.py','src/uac/models.py','src/uac/dsp.py','scripts/generate_interventions.py'])}
    ledger=EventLedger();index={'config':settings,'config_hash':fingerprint(settings),'shards':[],'status':'running','n':len(dataset['ids'])}
    for begin in range(0,len(dataset['ids']),args.shard_size):
        end=min(begin+args.shard_size,len(dataset['ids']));path=directory/f'shard-{begin:06d}.pt';receipt=path.with_suffix('.json')
        if receipt.exists():
            old=json.loads(receipt.read_text())
            if old['config_hash']!=index['config_hash']:raise RuntimeError('Intervention configuration changed; preserve and create new run')
            if not path.exists() or sha256_file(path)!=old['sha256']:
                stamp='.corrupt-'+str(time.time_ns())
                for damaged in (path,receipt):
                    if damaged.exists():damaged.rename(damaged.with_name(damaged.name+stamp))
                ledger.append('corrupt_intervention_quarantined',split=args.split,begin=begin)
            else:
                index['shards'].append(old);continue
        chunks=[];started=time.perf_counter()
        for b in range(begin,end,args.batch_size):
            count=min(args.batch_size,end-b)
            chunk=generate_intervention_features(flow,dataset['audio'][b:b+count],seed=7100000+b,k=settings['k'],steps=settings['steps'],probe=args.probe)
            chunks.append({key:value.cpu() for key,value in chunk.items()})
            if stop.requested:raise SystemExit(75)
        payload={key:torch.cat([c[key] for c in chunks]) for key in chunks[0]};payload['ids']=dataset['ids'][begin:end]
        torch.cuda.synchronize();seconds=time.perf_counter()-started
        save_torch(path,payload)
        record={'path':str(path),'sha256':sha256_file(path),'config_hash':index['config_hash'],'begin':begin,'end':end,'seconds':seconds,'valid_fraction':float(payload['valid'].float().mean())}
        atomic_json(receipt,record);index['shards'].append(record);atomic_json(directory/'index.json',index)
        ledger.append('intervention_shard_complete',split=args.split,**record);print(json.dumps(record),flush=True)
    index['status']='complete';atomic_json(directory/'index.json',index)
    ledger.append('intervention_split_complete',split=args.split,n=index['n'])
