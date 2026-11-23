#!/usr/bin/env python3
"""Restartable class-free background flow; pilot or full training on Ada."""
import argparse,json,time
from pathlib import Path
import torch
from uac.models import BackgroundFlow,interval_mask
from uac.synthetic import generate_batch
from uac.training import seed_all,save_torch
from uac.runtime import atomic_json,CheckpointStore,EventLedger,RunLock,StopRequested,source_fingerprint,fingerprint,completed_stage

parser=argparse.ArgumentParser();parser.add_argument('--pilot',action='store_true');parser.add_argument('--steps',type=int);args=parser.parse_args()
config=json.loads(Path('configs/research.json').read_text())
steps=args.steps or (64 if args.pilot else config['generator_steps'])
name='flow_pilot' if args.pilot else 'background_flow'
seed=17;seed_all(seed)
if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise SystemExit('One pinned CUDA GPU required')
model=BackgroundFlow(config['generator_base_channels']).cuda()
optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
settings={'seed':seed,'steps':steps,'batch_size':config['generator_batch_size'],'base':config['generator_base_channels'],'samples':48000,'method':'clamped_class_free_flow_matching','source':'synthetic_training_family','sampling_steps':8,'full_mask_fraction':0.3}
store=CheckpointStore('checkpoints/'+name+'/resume',config=settings,data_fingerprint=fingerprint({'synthetic':'train','seed_namespace':1000000}),code_fingerprint=source_fingerprint(['src/uac/models.py','src/uac/synthetic.py','scripts/train_flow.py']))
ledger=EventLedger('runs/events.jsonl')
with RunLock('runs/training.lock'),StopRequested() as stop:
    restored=store.load_training(model,optimizer)
    start=restored['global_step'] if restored else 0
    extra=restored['extra'] if restored else {'loss_history':[],'elapsed_seconds':0.}
    if restored:ledger.append('flow_resumed',name=name,step=start)
    begin_clock=time.monotonic();last_save=begin_clock
    for step in range(start,steps):
        batch=generate_batch(settings['batch_size'],num_samples=settings['samples'],seed=1000000+step,family='train',source_counts=1)
        interval=batch['intervals'][:,0]
        left=(interval[:,0]*16000).long();right=(interval[:,1]*16000).long()
        mask=interval_mask(len(left),settings['samples'],left,right,taper_samples=160)
        # Whole-record verification needs no-context generation in training.
        full=torch.rand((len(left),),device='cuda',generator=torch.Generator(device='cuda').manual_seed(3000000+step))<settings['full_mask_fraction']
        mask=torch.where(full[:,None],torch.ones_like(mask),mask)
        rng=torch.Generator(device='cuda').manual_seed(2000000+step)
        model.train();optimizer.zero_grad(set_to_none=True)
        loss=model.loss(batch['background'],batch['audio'],mask,generator=rng)
        if not bool(torch.isfinite(loss)):raise RuntimeError('Nonfinite flow loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2.);optimizer.step()
        if (step+1)%25==0:extra['loss_history'].append({'step':step+1,'loss':float(loss.detach())})
        if (step+1)%100==0 or time.monotonic()-last_save>=120 or step+1==steps or stop.requested:
            extra['elapsed_seconds']+=time.monotonic()-begin_clock;begin_clock=time.monotonic()
            store.save_training(model,optimizer,epoch=0,batch_cursor=step+1,global_step=step+1,extra=extra)
            ledger.append('flow_checkpoint',name=name,step=step+1,loss=float(loss.detach()),elapsed_seconds=extra['elapsed_seconds'])
            atomic_json('checkpoints/'+name+'/training.json',extra)
            print(json.dumps({'stage':name,'step':step+1,'loss':float(loss.detach()),'seconds':extra['elapsed_seconds']}),flush=True)
            last_save=time.monotonic()
            if stop.requested:raise SystemExit(75)
    model.eval()
    probe=generate_batch(8,num_samples=48000,seed=9000001,family='train',source_counts=1)
    interval=probe['intervals'][:,0];mask=interval_mask(8,48000,(interval[:,0]*16000).long(),(interval[:,1]*16000).long())
    with torch.inference_mode():
        replaced=model.replace(probe['audio'],mask,steps=8,generator=torch.Generator(device='cuda').manual_seed(90))
        projection=lambda x: ((x*probe['stems'][:,0]*mask).sum(-1)/(probe['stems'][:,0].square()*mask).sum(-1).clamp_min(1e-8)).abs()
        metrics={'outside_max_error':float(((replaced-probe['audio'])*(mask==0)).abs().max()),'finite':bool(torch.isfinite(replaced).all()),'median_target_projection_before':float(projection(probe['audio']).median()),'median_target_projection_after':float(projection(replaced).median()),'parameters':sum(p.numel() for p in model.parameters()),'peak_memory_mib':torch.cuda.max_memory_allocated()/1024**2,'steps':steps,'pilot_only':args.pilot}
    save_torch('checkpoints/'+name+'/final.pt',{'model':model.state_dict(),'settings':settings,'validation':metrics})
    atomic_json('manifests/'+name+'_validation.json',metrics)
    completed_stage('runs/'+name+'_complete.json',stage=name,config_hash=fingerprint(settings),artifacts=['checkpoints/'+name+'/final.pt','manifests/'+name+'_validation.json'],details=metrics)
    ledger.append('flow_training_complete',name=name,**metrics)
    print(json.dumps(metrics),flush=True)
