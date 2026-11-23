#!/usr/bin/env python3
"""Restartable CUDA temporal-activity CNN; controlled synthetic scenes only.

Run via scripts/run_on_ada.py. --max-steps 32 safely smoke-tests the full intended
configuration; subsequent invocation resumes that same run. No test data are
created or inspected. Fixed generated scene sets are reused across epochs/seeds.
"""
from __future__ import annotations
import argparse,copy,json,math,sys,time
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'src'))
import torch
from uac.models import SpectralDetector
from uac.runtime import (atomic_json,CheckpointStore,EventLedger,RunLock,StopRequested,
  fingerprint,completed_stage,stage_is_complete)
from uac.training import seed_all,save_torch
from synthetic_common import DEFAULTS,safe_gpu,code_hash,scene_fingerprint,make_batch,objective

def train_one(seed,config,args,stop,ledger):
    name=f'synthetic_temporal/seed_{seed}';out=PROJECT/'checkpoints'/name
    settings={**config,'model_seed':seed};ch=fingerprint(settings);dh=scene_fingerprint(config)
    receipt=PROJECT/'runs'/f'synthetic_temporal_seed_{seed}_complete.json'
    if stage_is_complete(receipt,config_hash=ch,data_fingerprint=dh):
        print(json.dumps({'status':'already_complete','seed':seed}),flush=True);return 0
    seed_all(seed);model=SpectralDetector().cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    store=CheckpointStore(out/'resume',config=settings,data_fingerprint=dh,code_fingerprint=code_hash(),torch_module=torch)
    restored=store.load_training(model,optimizer)
    epoch=restored['epoch'] if restored else 0;cursor=restored['batch_cursor'] if restored else 0
    step=restored['global_step'] if restored else 0
    extra=restored['extra'] if restored else {'history':[],'best_loss':1e30,'elapsed_seconds':0.,'phase':'train','dev_cursor':0,'dev_sum':0.,'train_sum':0.,'train_count':0}
    begin=time.monotonic();last_save=begin;invocation_steps=0
    def save(epoch,cursor):
        nonlocal begin,last_save
        torch.cuda.synchronize();extra['elapsed_seconds']+=time.monotonic()-begin;begin=time.monotonic()
        store.save_training(model,optimizer,epoch=epoch,batch_cursor=cursor,global_step=step,extra=extra)
        atomic_json(out/'training.json',extra)
        ledger.append('synthetic_temporal_checkpoint',seed=seed,epoch=epoch,batch_cursor=cursor,global_step=step,phase=extra['phase'])
        print(json.dumps({'stage':'synthetic_temporal','seed':seed,'epoch':epoch,'step':step,'phase':extra['phase'],'seconds':extra['elapsed_seconds'],'peak_memory_mib':torch.cuda.max_memory_allocated()/1024**2}),flush=True)
        last_save=time.monotonic()
    train_batches=math.ceil(config['train_scenes']/config['batch_size']);dev_batches=math.ceil(config['development_scenes']/config['batch_size'])
    for epoch in range(epoch,config['epochs']):
        # Shuffle batch identities (not samples inside a seeded generated batch).
        order=sorted(range(train_batches),key=lambda i:fingerprint({'epoch':epoch,'batch':i,'seed':17}))
        if extra['phase']=='train':
            model.train()
            for position in range(cursor,train_batches):
                batch=make_batch(config,'train',order[position]);optimizer.zero_grad(set_to_none=True)
                clip,frames=model(batch['audio'],return_frames=True);loss=objective(batch,clip,frames,config)
                if not bool(torch.isfinite(loss)):raise RuntimeError('Nonfinite synthetic training loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
                step+=1;invocation_steps+=1;extra['train_sum']+=float(loss.detach())*len(clip);extra['train_count']+=len(clip)
                limit=args.max_steps is not None and invocation_steps>=args.max_steps
                if step%config['checkpoint_steps']==0 or time.monotonic()-last_save>=config['checkpoint_seconds'] or stop.requested or limit:
                    save(epoch,position+1)
                    if stop.requested:return 75
                    if limit:return 0
            extra.update(phase='development',dev_cursor=0,dev_sum=0.);save(epoch,train_batches)
        model.eval()
        with torch.inference_mode():
            for index in range(extra['dev_cursor'],dev_batches):
                batch=make_batch(config,'development',index);clip,frames=model(batch['audio'],return_frames=True)
                loss=objective(batch,clip,frames,config);extra['dev_sum']+=float(loss)*len(clip);extra['dev_cursor']=index+1
                if (index+1)%config['checkpoint_steps']==0 or time.monotonic()-last_save>=config['checkpoint_seconds'] or stop.requested:
                    save(epoch,train_batches)
                    if stop.requested:return 75
        dev_loss=extra['dev_sum']/config['development_scenes']
        row={'epoch':epoch+1,'development_loss':dev_loss,'training_loss':extra['train_sum']/max(extra['train_count'],1)}
        extra['history'].append(row)
        if dev_loss<extra['best_loss']:
            extra['best_loss']=dev_loss
            save_torch(out/'best.pt',{'model':model.state_dict(),'config':config,'settings':settings,'epoch':epoch+1,'development_loss':dev_loss,'data_fingerprint':dh,'code_fingerprint':code_hash()})
        extra.update(phase='train',dev_cursor=0,dev_sum=0.,train_sum=0.,train_count=0,finished=epoch+1==config['epochs'])
        cursor=0;save(epoch+1,0);ledger.append('synthetic_temporal_epoch',seed=seed,**row)
    completed_stage(receipt,stage='synthetic_temporal_training',config_hash=ch,data_fingerprint=dh,
      artifacts=[out/'best.pt',out/'training.json'],details={'seed':seed,'history':extra['history'],'elapsed_seconds':extra['elapsed_seconds']})
    ledger.append('synthetic_temporal_training_complete',seed=seed)
    return 0

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds',type=int,nargs='+',default=[17,29,43]);parser.add_argument('--max-steps',type=int)
    args=parser.parse_args()
    if not args.seeds or len(set(args.seeds))!=len(args.seeds) or any(s not in DEFAULTS['seeds'] for s in args.seeds):parser.error('Seeds must be a unique subset of 17 29 43')
    if args.max_steps is not None and args.max_steps<1:parser.error('--max-steps must be positive')
    safe_gpu();config=copy.deepcopy(DEFAULTS)
    config_path=PROJECT/'configs/synthetic_temporal.json'
    with RunLock('runs/training.lock'),StopRequested() as stop:
        if config_path.exists() and json.loads(config_path.read_text())!=config:raise RuntimeError('Synthetic protocol changed; do not overwrite frozen settings')
        atomic_json(config_path,config)
        ledger=EventLedger();ledger.append('synthetic_temporal_training_started',seeds=args.seeds,config_hash=fingerprint(config),data_fingerprint=scene_fingerprint(config))
        for seed in args.seeds:
            status=train_one(seed,config,args,stop,ledger)
            if status or args.max_steps is not None:return status
    return 0
if __name__=='__main__':raise SystemExit(main())
