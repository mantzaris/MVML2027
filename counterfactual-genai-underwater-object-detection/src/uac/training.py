"""GPU training helpers with epoch/batch checkpoints and deterministic shuffles."""
from __future__ import annotations
import io, json, random, time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import average_precision_score
from uac.runtime import (atomic_bytes,atomic_json,CheckpointStore,EventLedger,StopRequested,source_fingerprint,fingerprint)


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True


def save_torch(path,payload):
    if torch.cuda.is_available(): torch.cuda.synchronize()
    buf=io.BytesIO();torch.save(payload,buf);atomic_bytes(path,buf.getvalue())


def train_supervised(model,train_x,train_y,dev_x,dev_y,*,name,config,data_hash,epochs=100,batch_size=64,lr=1e-3,patience=10,seed=17,frame_targets=None):
    """Resume at next optimizer boundary, using seed-indexed epoch permutation."""
    if any(x.device.type!='cuda' for x in (train_x,train_y,dev_x,dev_y)):
        raise RuntimeError('All training and validation tensors must be on Ada CUDA')
    out=Path('checkpoints')/name
    state_config={**config,'seed':seed,'epochs':epochs,'batch_size':batch_size,'lr':lr,'patience':patience,'n_train':len(train_y)}
    store=CheckpointStore(out/'resume',config=state_config,data_fingerprint=data_hash,code_fingerprint=source_fingerprint(['src/uac/models.py','src/uac/training.py']))
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-3)
    restored=store.load_training(model,optimizer)
    start_epoch=restored['epoch'] if restored else 0
    cursor=restored['batch_cursor'] if restored else 0
    step=restored['global_step'] if restored else 0
    extra=restored['extra'] if restored else {'best_ap':-1.,'bad_epochs':0,'history':[],'elapsed_seconds':0.}
    ledger=EventLedger('runs/events.jsonl')
    if restored: ledger.append('training_resumed',name=name,epoch=start_epoch,batch_cursor=cursor,global_step=step)
    if extra.get('finished'):
        payload=torch.load(out/'best.pt',map_location='cuda',weights_only=False);model.load_state_dict(payload['model']);return extra
    start_clock=time.monotonic();last_save=start_clock
    with StopRequested() as stop:
        for epoch in range(start_epoch,epochs):
            generator=torch.Generator(device='cuda').manual_seed(seed*100003+epoch)
            order=torch.randperm(len(train_y),device='cuda',generator=generator)
            model.train()
            for begin in range(cursor,len(order),batch_size):
                idx=order[begin:begin+batch_size]
                optimizer.zero_grad(set_to_none=True)
                if frame_targets is None:
                    logits=model(train_x[idx]);loss=F.binary_cross_entropy_with_logits(logits,train_y[idx])
                else:
                    logits,frames=model(train_x[idx],return_frames=True)
                    target=F.interpolate(frame_targets[idx,None].float(),size=frames.shape[-1],mode='nearest').squeeze(1)
                    loss=F.binary_cross_entropy_with_logits(logits,train_y[idx])+.5*F.binary_cross_entropy_with_logits(frames,target)
                if not bool(torch.isfinite(loss)): raise RuntimeError('Nonfinite supervised training loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step();step+=1
                if time.monotonic()-last_save>=120 or stop.requested:
                    extra['elapsed_seconds']+=time.monotonic()-start_clock;start_clock=time.monotonic()
                    store.save_training(model,optimizer,epoch=epoch,batch_cursor=begin+len(idx),global_step=step,extra=extra)
                    ledger.append('training_checkpoint',name=name,epoch=epoch,next_batch=begin+len(idx),global_step=step)
                    last_save=time.monotonic()
                    if stop.requested: raise SystemExit(75)
            cursor=0
            model.eval()
            with torch.inference_mode():
                predictions=torch.cat([model(dev_x[b:b+batch_size]) for b in range(0,len(dev_y),batch_size)])
                dev_loss=float(F.binary_cross_entropy_with_logits(predictions,dev_y));ap=float(average_precision_score(dev_y.cpu().numpy(),predictions.cpu().numpy()))
            row={'epoch':epoch+1,'dev_ap':ap,'dev_loss':dev_loss,'last_train_loss':float(loss)};extra['history'].append(row)
            if ap>extra['best_ap']+1e-5:
                extra['best_ap']=ap;extra['bad_epochs']=0
                save_torch(out/'best.pt',{'model':model.state_dict(),'config':state_config,'epoch':epoch+1,'dev_ap':ap,'data_hash':data_hash})
            else: extra['bad_epochs']+=1
            extra['elapsed_seconds']+=time.monotonic()-start_clock;start_clock=time.monotonic()
            extra['finished']=bool(extra['bad_epochs']>=patience or epoch+1==epochs)
            store.save_training(model,optimizer,epoch=epoch+1,batch_cursor=0,global_step=step,extra=extra)
            atomic_json(out/'training.json',extra)
            ledger.append('training_epoch',name=name,**row)
            if extra['finished']:break
    model.load_state_dict(torch.load(out/'best.pt',map_location='cuda',weights_only=False)['model'])
    return extra
