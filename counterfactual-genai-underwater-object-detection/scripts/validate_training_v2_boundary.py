#!/usr/bin/env python3
"""Prove exact CUDA recovery when checkpoint cursor equals the epoch length.

Small optimizer fixtures only; no real or held-out examples. Checkpoints use the
actual durable store and version2 supervised trainer. Both persisted and absent
loss diagnostics are tested; model/optimizer state and the next step must agree.
"""
from __future__ import annotations
import copy
import json
from pathlib import Path
import sys
import time

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'src'))
import torch
from uac.runtime import RunLock,CheckpointStore,fingerprint,source_fingerprint,atomic_json,sha256_file
from uac.training_v2 import seed_all,train_supervised
from extract_real_features import _safe_gpu

class TinyClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__();self.linear=torch.nn.Linear(4,1)
    def forward(self,x):
        if x.device.type!='cuda':raise RuntimeError('CUDA fixture only')
        return self.linear(x).squeeze(-1)

def equal_state(a,b):
    if torch.is_tensor(a):return torch.is_tensor(b) and torch.equal(a.cpu(),b.cpu())
    if isinstance(a,dict):return isinstance(b,dict) and set(a)==set(b) and all(equal_state(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(equal_state(x,y) for x,y in zip(a,b))
    return a==b

def step(model,optimizer,x,y):
    model.train();optimizer.zero_grad(set_to_none=True)
    loss=torch.nn.functional.binary_cross_entropy_with_logits(model(x),y)
    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
    return float(loss.detach())

def run_fixture(include_loss):
    seed=811;seed_all(seed)
    x=torch.arange(32,device='cuda',dtype=torch.float32).reshape(8,4)/16-1
    y=torch.tensor([0.,1.,0.,1.,1.,0.,1.,0.],device='cuda')
    reference=TinyClassifier().cuda()
    optimizer=torch.optim.AdamW(reference.parameters(),lr=1e-3,weight_decay=1e-3)
    order=torch.randperm(8,device='cuda',generator=torch.Generator(device='cuda').manual_seed(seed*100003))
    for begin in (0,4):last_loss=step(reference,optimizer,x[order[begin:begin+4]],y[order[begin:begin+4]])
    expected_model=copy.deepcopy(reference.state_dict());expected_optimizer=copy.deepcopy(optimizer.state_dict())
    name=f'validation/training_v2_boundary_{time.time_ns()}_{int(include_loss)}'
    config={'fixture':'cursor_equal_to_epoch_length','loss_diagnostic_in_checkpoint':include_loss}
    settings={**config,'seed':seed,'epochs':1,'batch_size':4,'lr':1e-3,'patience':1,'n_train':8}
    data_hash=fingerprint({'fixture':'fixed8x4','labels':[0,1,0,1,1,0,1,0]})
    store=CheckpointStore(Path('checkpoints')/name/'resume',config=settings,data_fingerprint=data_hash,
                          code_fingerprint=source_fingerprint(['src/uac/models.py','src/uac/training_v2.py']))
    extra={'best_ap':-1.,'bad_epochs':0,'history':[],'elapsed_seconds':0.}
    if include_loss:extra['last_train_loss']=last_loss
    store.save_training(reference,optimizer,epoch=0,batch_cursor=8,global_step=2,extra=extra)
    recovered=TinyClassifier().cuda()
    result=train_supervised(recovered,x,y,x,y,name=name,config=config,data_hash=data_hash,
                            epochs=1,batch_size=4,lr=1e-3,patience=1,seed=seed)
    payload=store.load_state(map_location='cpu')
    model_equal=equal_state(expected_model,recovered.state_dict())
    optimizer_equal=equal_state(expected_optimizer,payload['state']['optimizer'])
    count_exact=payload['global_step']==2 and payload['cursor']=={'epoch':1,'batch_cursor':0}
    logged=result['history'][0]['last_train_loss']
    loss_correct=(logged==last_loss) if include_loss else logged is None
    next_optimizer=torch.optim.AdamW(recovered.parameters(),lr=1e-3,weight_decay=1e-3)
    next_optimizer.load_state_dict(payload['state']['optimizer'])
    next_order=torch.randperm(8,device='cuda',generator=torch.Generator(device='cuda').manual_seed(seed*100003+1))[:4]
    step(reference,optimizer,x[next_order],y[next_order])
    step(recovered,next_optimizer,x[next_order],y[next_order])
    next_exact=equal_state(reference.state_dict(),recovered.state_dict()) and equal_state(optimizer.state_dict(),next_optimizer.state_dict())
    checks={'resume_did_not_repeat_optimizer_update':count_exact,'model_equal_after_boundary':model_equal,
            'adamw_state_equal_after_boundary':optimizer_equal,'loss_diagnostic_valid_or_explicit_null':loss_correct,
            'next_optimizer_step_bitwise_equal':next_exact}
    if not all(checks.values()):raise AssertionError(checks)
    return {'loss_was_present':include_loss,'checkpoint_cursor_before':8,'epoch_length':8,
            'completed_global_step':payload['global_step'],'checks':checks,'checkpoint_directory':str(Path('checkpoints')/name)}

def main():
    _safe_gpu()
    original_hash=sha256_file('src/uac/training.py')
    with RunLock('runs/training.lock'):
        records=[run_fixture(True),run_fixture(False)]
        report={'status':'passed','scope':'two small CUDA optimizer fixtures, no research data',
                'bug':'Original helper read local loss after an empty loop when saved cursor equals epoch length',
                'resolution':'Version2 persists last loss in checkpoint extra; unknown historical diagnostic remains null',
                'original_training_sha256':original_hash,'original_training_unchanged':sha256_file('src/uac/training.py')==original_hash,
                'v2_training_sha256':sha256_file('src/uac/training_v2.py'),'records':records}
        atomic_json('manifests/training_v2_boundary_validation.json',report)
        print(json.dumps(report),flush=True)
    return 0
if __name__=='__main__':raise SystemExit(main())
