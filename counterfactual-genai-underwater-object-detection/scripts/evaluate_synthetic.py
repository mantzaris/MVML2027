#!/usr/bin/env python3
"""Locked, restartable controlled temporal evaluation on the pinned Ada.

Test requires runs/test_freeze.json containing synthetic_config_sha256 (SHA256
of configs/synthetic_temporal.json) and synthetic_checkpoints, a mapping from
relative best.pt paths to their SHA256 hashes. Parent creates this gate after
all model/development decisions. Fully synthetic results never mix with real
or semisynthetic observations. Shards are atomic, checksummed, and replayable.
"""
from __future__ import annotations
import argparse,json,math,sys,time
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'src'))
import torch
from uac.models import SpectralDetector
from uac.runtime import (atomic_bytes,atomic_json,RunLock,StopRequested,EventLedger,
 fingerprint,sha256_file,project_path,utc_now,completed_stage)
from synthetic_common import safe_gpu,code_hash,scene_fingerprint,make_batch,prediction_rows,summarize_rows

def check_gate(config_path,checkpoints):
    freeze_path=PROJECT/'runs/test_freeze.json'
    if not freeze_path.is_file():raise RuntimeError('Test is locked until runs/test_freeze.json exists')
    freeze=json.loads(freeze_path.read_text())
    if freeze.get('synthetic_config_sha256')!=sha256_file(config_path):raise RuntimeError('Test freeze does not pin the synthetic protocol file')
    for path in checkpoints:
        relative=str(path.relative_to(PROJECT))
        if freeze.get('synthetic_checkpoints',{}).get(relative)!=sha256_file(path):raise RuntimeError(f'Test freeze does not pin checkpoint {relative}')
    return sha256_file(freeze_path)

def condition_run(model,seed,split,family,snr,config,checkpoint_hash,freeze_hash,stop,ledger,max_shards=None):
    condition='mixed_snr' if snr is None else f'snr_{snr:+03d}'
    directory=PROJECT/'predictions/synthetic'/f'seed_{seed}'/split/family/condition
    directory.mkdir(parents=True,exist_ok=True)
    expected={'schema_version':1,'config_hash':fingerprint(config),'code_fingerprint':code_hash(),
       'data_fingerprint':scene_fingerprint(config),'checkpoint_sha256':checkpoint_hash,
       'freeze_sha256':freeze_hash,'seed':seed,'split':split,'family':family,'condition':condition,'snr_db':snr}
    total=config[split+'_scenes'];batches=math.ceil(total/config['batch_size']);shard_batches=8
    all_rows=[];receipts=[];computed=0
    for first in range(0,batches,shard_batches):
        end=min(first+shard_batches,batches);path=directory/f'shard-{first:05d}-{end:05d}.jsonl';sidecar=path.with_suffix('.json')
        wanted={**expected,'first_batch':first,'last_batch_exclusive':end}
        receipt=None
        if sidecar.is_file():
            receipt=json.loads(sidecar.read_text())
            if any(receipt.get(k)!=v for k,v in wanted.items()):raise RuntimeError(f'Incompatible prediction shard: {sidecar}')
            if not path.is_file() or sha256_file(path)!=receipt.get('payload_sha256'):
                stamp=f'.corrupt-{time.time_ns()}'
                for item in (path,sidecar):
                    if item.exists():item.rename(item.with_name(item.name+stamp))
                receipt=None
        if receipt is None:
            if stop.requested:return None
            if max_shards is not None and computed>=max_shards:return None
            rows=[];started=time.monotonic()
            with torch.inference_mode():
                for index in range(first,end):
                    batch=make_batch(config,split,index,family,snr)
                    clip,frames=model(batch['audio'],return_frames=True)
                    rows.extend(prediction_rows(batch,clip,frames,config))
                    if stop.requested:return None
            atomic_bytes(path,b''.join((json.dumps(r,sort_keys=True,allow_nan=False)+'\n').encode() for r in rows))
            receipt={**wanted,'payload':str(path.relative_to(PROJECT)),'payload_sha256':sha256_file(path),
              'payload_bytes':path.stat().st_size,'count':len(rows),'seconds':time.monotonic()-started,'completed_utc':utc_now()}
            atomic_json(sidecar,receipt);computed+=1
            ledger.append('synthetic_prediction_shard',seed=seed,split=split,family=family,condition=condition,first_batch=first,count=len(rows),sha256=receipt['payload_sha256'])
            print(json.dumps({'stage':'synthetic_prediction','seed':seed,'split':split,'family':family,'condition':condition,'through':min(end*config['batch_size'],total),'total':total}),flush=True)
        else:rows=[json.loads(line) for line in path.read_text().splitlines()]
        all_rows.extend(rows);receipts.append(receipt)
        atomic_json(directory/'index.json',{**expected,'status':'running','completed_rows':len(all_rows),'expected_rows':total,'shards':receipts})
    if len(all_rows)!=total or len({r['id'] for r in all_rows})!=total:raise RuntimeError('Incomplete or duplicate synthetic predictions')
    records=summarize_rows(all_rows,config,seed=seed,split=split,family=family,condition=condition,snr=snr)
    index={**expected,'status':'complete','completed_rows':total,'expected_rows':total,'shards':receipts,'records':records,'completed_utc':utc_now()}
    atomic_json(directory/'index.json',index)
    return index

def write_report(config,split):
    records=[];indices=[]
    for path in sorted((PROJECT/'predictions/synthetic').glob(f'seed_*/{split}/*/*/index.json')):
        index=json.loads(path.read_text())
        if index.get('status')!='complete' or index.get('config_hash')!=fingerprint(config):continue
        for receipt in index['shards']:
            if sha256_file(receipt['payload'])!=receipt['payload_sha256']:raise RuntimeError('Prediction artifact checksum mismatch during report')
        records.extend(index['records']);indices.append({'path':str(path.relative_to(PROJECT)),'sha256':sha256_file(path)})
    path=PROJECT/'reports'/('synthetic_results.json' if split=='test' else 'synthetic_development_results.json')
    atomic_json(path,{'schema_version':1,'origin':'fully_synthetic','records':records,'config':config,'config_hash':fingerprint(config),
      'indices':indices,'limitations':['Received union activity, not separated physical objects.','Controlled simulator does not establish realism or transfer.','Heldout source, background and propagation changes are joint.','Same-family SNR conditions reuse scenes and are paired; family conditions are not exactly paired.','Frames use equal-duration scene bins; interval precision is approximately 128 ms.'],
      'completed_conditions':len(indices),'expected_test_conditions':len(config['seeds'])*len(config['test_families'])*(1+len(config['test_snr_db'])),'created_utc':utc_now()})
    return path

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--seeds',nargs='+',type=int,default=[17,29,43])
    parser.add_argument('--split',choices=['development','test'],default='development');parser.add_argument('--max-shards',type=int)
    args=parser.parse_args();safe_gpu();config_path=PROJECT/'configs/synthetic_temporal.json';config=json.loads(config_path.read_text())
    if len(set(args.seeds))!=len(args.seeds) or any(s not in config['seeds'] for s in args.seeds):parser.error('Unknown or duplicate seed')
    checkpoints=[PROJECT/'checkpoints/synthetic_temporal'/f'seed_{seed}'/'best.pt' for seed in args.seeds]
    freeze_hash=check_gate(config_path,checkpoints) if args.split=='test' else None
    conditions=[(family,snr) for family in config['test_families'] for snr in [None]+config['test_snr_db']] if args.split=='test' else [('train',None)]
    with RunLock('runs/training.lock'),StopRequested() as stop:
        ledger=EventLedger();model=SpectralDetector().cuda().eval()
        for seed,path in zip(args.seeds,checkpoints):
            payload=torch.load(path,map_location='cuda',weights_only=False)
            if payload['config']!=config:raise RuntimeError('Checkpoint does not match frozen configuration')
            model.load_state_dict(payload['model'])
            for family,snr in conditions:
                result=condition_run(model,seed,args.split,family,snr,config,sha256_file(path),freeze_hash,stop,ledger,args.max_shards)
                if result is None:return 75 if stop.requested else 0
                report=write_report(config,args.split)
                ledger.append('synthetic_condition_complete',seed=seed,split=args.split,family=family,snr=snr,records=result['records'])
        print(json.dumps({'report':str(report),'split':args.split,'status':'complete'}),flush=True)
    return 0
if __name__=='__main__':raise SystemExit(main())
