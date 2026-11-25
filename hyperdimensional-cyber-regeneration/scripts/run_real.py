#!/usr/bin/env python3
"""Chronological captured-flow application; all ML and research tensors on Ada."""
import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from real_models import KEYS,make_model,infer,loss_terms
from real_metrics import GaussianNB,metrics,select_threshold
from real_data import (TRAIN,VALIDATION,TEST,CONDITIONS,load_capture,fit_preprocessing,windows,condition_batch,
                       concatenate,subset,training_mixture,input_hash,sha,require_cuda)


def write_csv(path,rows):
    if not rows:return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)


def predictions(model,b):return model.probabilities(b) if model.key=='NB' else infer(model,b)


def capture_weights(batch):
    _,inverse,counts=torch.unique(batch['scenario'],return_inverse=True,return_counts=True)
    return len(inverse)/(len(counts)*counts[inverse].float())


def prepare_data(pilot=False):
    captures=[load_capture(s) for s in TRAIN]
    preprocessing=fit_preprocessing(captures)
    train=concatenate([windows(c,s,preprocessing,limit=64 if pilot else None) for s,c in zip(TRAIN,captures)])
    del captures
    validation={s:windows(load_capture(s),s,preprocessing,limit=16 if pilot else None) for s in ([2] if pilot else VALIDATION)}
    return training_mixture(train,preprocessing),validation,preprocessing


def validation_cases(validation,preprocessing):
    return [condition_batch(b,c,preprocessing) for b in validation.values() for c in ['clean','missing20','spoof20']]


@torch.inference_mode()
def collect(model,batch,batch_size):
    return torch.cat([predictions(model,subset(batch,ids)) for ids in torch.arange(len(batch['x']),device='cuda').split(batch_size)])


def train_model(key,train,validation,args,seed,curves,directory):
    torch.manual_seed(seed);model=make_model(key).cuda();weights=capture_weights(train)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    torch.cuda.synchronize();started=time.monotonic()
    pretrain=args.pretrain_epochs if key in ['DAE','GMAE','HDC'] else 0
    for epoch in range(pretrain):
        model.train();values=[]
        for ids in torch.randperm(len(train['x']),device='cuda').split(args.batch):
            b=subset(train,ids)
            with torch.autocast('cuda',dtype=torch.bfloat16):per_window=model.pretraining_loss(b)
            loss=(per_window*weights[ids]).mean();optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step();values.append(loss.detach())
        curves.append(dict(seed=seed,method=key,epoch=epoch,split='pretrain',loss=float(torch.stack(values).mean())))
    torch.cuda.synchronize();pretrain_seconds=time.monotonic()-started;supervised_start=time.monotonic()
    best=float('inf');wait=0;state=None;best_epoch=-1
    for epoch in range(args.epochs):
        model.train();values=[]
        for ids in torch.randperm(len(train['x']),device='cuda').split(args.batch):
            b=subset(train,ids)
            with torch.autocast('cuda',dtype=torch.bfloat16):q=infer(model,b)
            loss=(loss_terms(q,b)['total']*weights[ids]).mean()
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();values.append(loss.detach())
        model.eval()
        with torch.inference_mode():
            scores=[]
            for case in validation:
                totals=[]
                for ids in torch.arange(len(case['x']),device='cuda').split(args.batch):
                    b=subset(case,ids);totals.append(loss_terms(infer(model,b),b)['total'])
                scores.append(torch.cat(totals).mean())
            score=float(torch.stack(scores).mean())
        curves.append(dict(seed=seed,method=key,epoch=epoch,split='train',loss=float(torch.stack(values).mean())))
        curves.append(dict(seed=seed,method=key,epoch=epoch,split='validation',loss=score))
        if score<best-1e-4:
            best=score;best_epoch=epoch;wait=0;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:wait+=1
        if epoch%5==0:print(json.dumps(dict(seed=seed,method=key,epoch=epoch,validation=score)),flush=True)
        if wait>=args.patience:break
    torch.cuda.synchronize();supervised_seconds=time.monotonic()-supervised_start
    model.load_state_dict(state);model.eval();directory.mkdir(parents=True,exist_ok=True)
    path=directory/f'{seed}_{key}.pt';torch.save(state,path)
    record=dict(seed=seed,method=key,best_epoch=best_epoch,epochs_run=epoch+1,pretrain_epochs=pretrain,
        validation_loss=best,parameters=sum(p.numel() for p in model.parameters()),pretrain_seconds=pretrain_seconds,
        supervised_seconds=supervised_seconds,train_seconds=time.monotonic()-started,checkpoint_sha256=sha(path))
    return model,record


@torch.inference_mode()
def evaluate(model,b,seed,condition,scenario,threshold,record,args,raw_directory):
    q=collect(model,b,args.batch);valid=b['valid'];labels=b['y'][valid];prob=q[valid]
    scores=metrics(labels,prob,threshold)
    window=b['window'][:,None].expand_as(valid)[valid];block=b['block'][:,None].expand_as(valid)[valid]
    raw=dict(labels=labels.cpu().numpy(),prob=prob.cpu().numpy(),window=window.cpu().numpy(),block=block.cpu().numpy(),row_id=b['row_id'][valid].cpu().numpy())
    directory=raw_directory/str(seed);directory.mkdir(parents=True,exist_ok=True)
    path=directory/f'{model.key}_{condition}_{scenario}.npz';np.savez_compressed(path,**raw)
    one=subset(b,slice(0,1))
    for _ in range(5):predictions(model,one)
    torch.cuda.synchronize();baseline=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();times=[]
    for _ in range(30):
        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        start.record();predictions(model,one);end.record();end.synchronize();times.append(start.elapsed_time(end))
    peak=(torch.cuda.max_memory_allocated()-baseline)/2**20
    row=dict(seed=seed,method=model.key,condition=condition,scenario=scenario,n=int(valid.sum()),positives=int(labels.sum()),
        **scores,threshold=threshold,latency_ms=float(np.median(times)),p95=float(np.quantile(times,.95)),peak_vram_mb=peak,
        train_seconds=record['train_seconds'],parameters=record['parameters'],input_bytes=int(one['x'].numel()*4+one['adj'].numel()*4+one['valid'].numel()))
    cal=[]
    for j in range(10):
        selected=(prob>=j/10)&(prob<(j+1)/10 if j<9 else prob<=1)
        cal.append(dict(seed=seed,method=model.key,condition=condition,scenario=scenario,bin=j,count=int(selected.sum()),
            confidence=float(prob[selected].mean()) if selected.any() else (j+.5)/10,
            accuracy=float(labels[selected].mean()) if selected.any() else float('nan')))
    return row,cal,path


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--pilot',action='store_true');args=parser.parse_args()
    config=json.loads((ROOT/'configs/study_real.json').read_text())
    for key in ['seeds','epochs','pretrain_epochs','patience','batch','lr']:setattr(args,key,config[key])
    if args.pilot:args.seeds=[11];args.epochs=2;args.pretrain_epochs=1
    require_cuda();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True;torch.use_deterministic_algorithms(True)
    data_manifest_path=ROOT/'results/real/ctu13_data_manifest.json'
    data_manifest=json.loads(data_manifest_path.read_text())
    if not data_manifest.get('complete'):raise RuntimeError('Wait for complete CTU-13 parsing and split checks')
    if not data_manifest['temporal_split_check']['passed']:raise AssertionError('Chronological completion check failed')
    for record in data_manifest['files']:
        if sha(ROOT/record['prepared_path'])!=record['prepared_sha256']:raise AssertionError('Prepared data changed')
        if record['scenario'] in TEST and min(record['rows_attack'],record['rows_normal'])<=0:raise AssertionError('Both test classes required')
    if sha(ROOT/'data/ctu13_sources.json')!=data_manifest['source_manifest_sha256']:raise AssertionError('Source catalogue changed')
    if sha(ROOT/'scripts/prepare_ctu13.py')!=data_manifest['preparation_script_sha256']:raise AssertionError('Parser changed')
    output=ROOT/'work/real_pilot' if args.pilot else ROOT/'results/real';output.mkdir(parents=True,exist_ok=True)
    raw_directory=output/'runs' if args.pilot else ROOT/'results/runs/real'
    checkpoint_directory=output/'checkpoints' if args.pilot else ROOT/'results/checkpoints/real'
    started=time.monotonic();train,validation,preprocessing=prepare_data(args.pilot)
    val_cases=validation_cases(validation,preprocessing);methods=list(KEYS)+['NB']
    source_paths=['src/real_data.py','src/real_models.py','src/real_metrics.py','scripts/run_real.py','scripts/prepare_ctu13.py','configs/study_real.json','data/ctu13_sources.json']
    manifest=dict(settings=vars(args),frozen_protocol=config,methods=methods,hardware=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_hashes={p:sha(ROOT/p) for p in source_paths},
        data_manifest_sha256=sha(data_manifest_path),checkpoints=[],thresholds=[],case_hashes={},raw_hashes={},
        train_hash=input_hash(train),validation_hashes={str(s):input_hash(b) for s,b in validation.items()},
        preprocessing={k:v.cpu().tolist() for k,v in preprocessing.items()},train_windows=len(train['x']),train_flows=int(train['valid'].sum()))
    curves=[];rows=[];calibration=[]
    for seed in args.seeds:
        models={};records={}
        for key in KEYS:
            model,record=train_model(key,train,val_cases,args,seed,curves,checkpoint_directory)
            models[key]=model;records[key]=record;manifest['checkpoints'].append(record)
        start=time.monotonic();model=GaussianNB().cuda().fit(train,capture_weights(train)).eval();torch.cuda.synchronize()
        path=checkpoint_directory/f'{seed}_NB.pt';torch.save(model.state_dict(),path)
        record=dict(seed=seed,method='NB',best_epoch=None,epochs_run=0,pretrain_epochs=0,validation_loss=None,parameters=sum(v.numel() for v in model.buffers()),fit_method='closed_form_Gaussian_statistics',
            pretrain_seconds=0.,supervised_seconds=0.,train_seconds=time.monotonic()-start,checkpoint_sha256=sha(path))
        models['NB']=model;records['NB']=record;manifest['checkpoints'].append(record)
        thresholds={}
        for key,model in models.items():
            truth=[];prob=[]
            for b in validation.values():
                q=collect(model,b,args.batch);truth.append(b['y'][b['valid']]);prob.append(q[b['valid']])
            thresholds[key]=select_threshold(torch.cat(truth),torch.cat(prob))
            manifest['thresholds'].append(dict(seed=seed,method=key,threshold=thresholds[key],selection='clean validation F1'))
        for scenario in ([3] if args.pilot else TEST):
            clean=windows(load_capture(scenario),scenario,preprocessing,limit=16 if args.pilot else None)
            for condition in (['clean','missing20'] if args.pilot else CONDITIONS):
                b=condition_batch(clean,condition,preprocessing);tag=f'{condition}_{scenario}'
                if seed==args.seeds[0]:manifest['case_hashes'][tag]=input_hash(b)
                for key,model in models.items():
                    row,cal,path=evaluate(model,b,seed,condition,scenario,thresholds[key],records[key],args,raw_directory)
                    rows.append(row);calibration.extend(cal);manifest['raw_hashes'][str(path.relative_to(ROOT))]=sha(path)
                del b
            del clean
        for name,records_ in [('summary',rows),('calibration',calibration),('curves',curves)]:write_csv(output/f'{name}.csv',records_)
        print(json.dumps(dict(seed_complete=seed,elapsed_seconds=time.monotonic()-started)),flush=True)
        del models;torch.cuda.empty_cache()
    manifest['wall_seconds']=time.monotonic()-started
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

if __name__=='__main__':main()
