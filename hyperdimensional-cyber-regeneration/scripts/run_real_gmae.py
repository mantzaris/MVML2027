#!/usr/bin/env python3
"""Exploratory GraphMAE follow-up; finish training before any new test inference."""
import argparse
import copy
import csv
import json
import subprocess
import time
from pathlib import Path
import torch
import run_real as base
from real_gmae_variants import KEYS, make_model
from real_models import infer, loss_terms
from real_metrics import select_threshold
from real_data import ROOT, TEST, CONDITIONS, windows, load_capture, condition_batch, subset, input_hash, sha, require_cuda


def load_csv(path):
    with path.open() as f:return list(csv.DictReader(f))


def base_evidence():
    path=ROOT/'results/real/manifest.json';manifest=json.loads(path.read_text())
    assert not manifest['settings']['pilot'] and len(manifest['raw_hashes'])==675
    for relative,digest in manifest['source_hashes'].items():
        assert sha(ROOT/relative)==digest,relative
    for relative,digest in manifest['raw_hashes'].items():
        assert sha(ROOT/relative)==digest,relative
    for record in manifest['checkpoints']:
        path_=ROOT/'results/checkpoints/real'/f"{record['seed']}_{record['method']}.pt"
        assert sha(path_)==record['checkpoint_sha256'],path_
    data_path=ROOT/'results/real/ctu13_data_manifest.json';data=json.loads(data_path.read_text())
    assert sha(data_path)==manifest['data_manifest_sha256']
    assert data['complete'] and data['temporal_split_check']['passed']
    for record in data['files']:
        assert sha(ROOT/record['prepared_path'])==record['prepared_sha256'],record['prepared_path']
    return path,manifest


def train_model(key,train,validation,args,seed,curves,directory):
    torch.manual_seed(seed);model=make_model(key).cuda();weights=base.capture_weights(train)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    torch.cuda.synchronize();started=time.monotonic()
    pretrain=0 if key=='EDGE_DENSE' else args.pretrain_epochs
    for epoch in range(pretrain):
        model.train();values=[]
        for ids in torch.randperm(len(train['x']),device='cuda').split(args.batch):
            b=subset(train,ids)
            with torch.autocast('cuda',dtype=torch.bfloat16):per_window=model.pretraining_loss(b)
            loss=(per_window*weights[ids]).mean();optimizer.zero_grad(set_to_none=True)
            loss.backward();optimizer.step();values.append(loss.detach())
        curves.append(dict(seed=seed,method=key,epoch=epoch,split='pretrain',loss=float(torch.stack(values).mean())))
    torch.cuda.synchronize();pretrain_seconds=time.monotonic()-started;supervised_start=time.monotonic()
    best=float('inf');wait=0;state=None;best_epoch=-1
    for epoch in range(args.epochs):
        model.train();values=[];regularizers=[];base_values=[]
        for ids in torch.randperm(len(train['x']),device='cuda').split(args.batch):
            b=subset(train,ids)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                q=infer(model,b);regularizer=model.supervised_regularizer(b,q)
            supervised=loss_terms(q,b)['total']
            loss=((supervised+regularizer)*weights[ids]).mean()
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5)
            optimizer.step();values.append(loss.detach())
            regularizers.append((regularizer.detach()*weights[ids]).mean());base_values.append((supervised.detach()*weights[ids]).mean())
        model.eval()
        with torch.inference_mode():
            scores=[]
            for case in validation:
                totals=[]
                for ids in torch.arange(len(case['x']),device='cuda').split(args.batch):
                    b=subset(case,ids);totals.append(loss_terms(infer(model,b),b)['total'])
                scores.append(torch.cat(totals).mean())
            score=float(torch.stack(scores).mean())
        for split,value in [('train',torch.stack(values).mean()),('train_base',torch.stack(base_values).mean()),('train_regularizer',torch.stack(regularizers).mean())]:
            curves.append(dict(seed=seed,method=key,epoch=epoch,split=split,loss=float(value)))
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
        supervised_seconds=supervised_seconds,train_seconds=time.monotonic()-started,
        checkpoint_path=str(path.relative_to(ROOT)),checkpoint_sha256=sha(path))
    return model,record


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--pilot',action='store_true');args=parser.parse_args()
    config=json.loads((ROOT/'configs/study_real_gmae.json').read_text())
    for key in ['seeds','epochs','pretrain_epochs','patience','batch','lr']:setattr(args,key,config[key])
    if args.pilot:args.seeds=[11];args.epochs=2;args.pretrain_epochs=1
    require_cuda();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True;torch.use_deterministic_algorithms(True)
    base_path,old=base_evidence()
    output=ROOT/'work/real_gmae_pilot' if args.pilot else ROOT/'results/real_gmae';output.mkdir(parents=True,exist_ok=True)
    raw_directory=output/'runs' if args.pilot else ROOT/'results/runs/real_gmae'
    checkpoint_directory=output/'checkpoints' if args.pilot else ROOT/'results/checkpoints/real_gmae'
    if not args.pilot and (output/'manifest.json').exists():raise RuntimeError('Completed study already exists; do not overwrite frozen results')
    started=time.monotonic();train,validation,preprocessing=base.prepare_data(args.pilot)
    val_cases=base.validation_cases(validation,preprocessing)
    if not args.pilot:
        assert input_hash(train)==old['train_hash'],'Training observations changed'
        assert {str(s):input_hash(b) for s,b in validation.items()}==old['validation_hashes'],'Validation observations changed'
        assert {k:v.cpu().tolist() for k,v in preprocessing.items()}==old['preprocessing'],'Preprocessing changed'
    imported=[] if args.pilot else old['methods'];methods=imported+list(KEYS)
    paths=['src/real_gmae_variants.py','scripts/run_real_gmae.py','configs/study_real_gmae.json','src/real_data.py','src/real_models.py','src/real_metrics.py','scripts/run_real.py','scripts/prepare_ctu13.py','data/ctu13_sources.json']
    manifest=dict(settings=vars(args),frozen_protocol=config,methods=methods,new_methods=list(KEYS),imported_methods=imported,
        method_origins={m:'real' if m in imported else 'real_gmae' for m in methods},hardware=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_hashes={p:sha(ROOT/p) for p in paths},
        data_manifest_sha256=old['data_manifest_sha256'],base_manifest_sha256=sha(base_path),base_summary_sha256=sha(ROOT/'results/real/summary.csv'),
        base_curves_sha256=sha(ROOT/'results/real/curves.csv'),base_calibration_sha256=sha(ROOT/'results/real/calibration.csv'),
        checkpoints=[],thresholds=[],case_hashes={},raw_hashes={},train_hash=input_hash(train),validation_hashes={str(s):input_hash(b) for s,b in validation.items()},
        preprocessing={k:v.cpu().tolist() for k,v in preprocessing.items()},train_windows=len(train['x']),train_flows=int(train['valid'].sum()),
        inference_scope='New runs FP32 evaluation; BF16-autocast training, TF32 allowed, deterministic algorithms; same original protocol.',
        interpretation='Exploratory follow-up using already examined CTU-13 test captures; source freeze does not restore untouched confirmation.')
    rows=[];curves=[];calibration=[]
    if not args.pilot:
        rows=load_csv(ROOT/'results/real/summary.csv');curves=load_csv(ROOT/'results/real/curves.csv');calibration=load_csv(ROOT/'results/real/calibration.csv')
        assert len(rows)==675
        manifest['raw_hashes']=copy.deepcopy(old['raw_hashes']);manifest['case_hashes']=copy.deepcopy(old['case_hashes']);manifest['thresholds']=copy.deepcopy(old['thresholds'])
        manifest['checkpoints']=[dict(r,checkpoint_path=f"results/checkpoints/real/{r['seed']}_{r['method']}.pt") for r in old['checkpoints']]
    training_records={};thresholds={}
    # All new model settings, checkpoints and thresholds are selected before any new held-out inference.
    for seed in args.seeds:
        for key in KEYS:
            model,record=train_model(key,train,val_cases,args,seed,curves,checkpoint_directory)
            training_records[(seed,key)]=record;manifest['checkpoints'].append(record)
            truth=[];prob=[]
            for b in validation.values():
                q=base.collect(model,b,args.batch);truth.append(b['y'][b['valid']]);prob.append(q[b['valid']])
            threshold=select_threshold(torch.cat(truth),torch.cat(prob));thresholds[(seed,key)]=threshold
            manifest['thresholds'].append(dict(seed=seed,method=key,threshold=threshold,selection='clean validation F1'))
            del model
        base.write_csv(output/'curves.csv',curves)
        print(json.dumps(dict(training_seed_complete=seed,elapsed_seconds=time.monotonic()-started)),flush=True)
    manifest['all_training_complete_seconds']=time.monotonic()-started
    for seed in args.seeds:
        models={}
        for key in KEYS:
            record=training_records[(seed,key)];path=ROOT/record['checkpoint_path'];assert sha(path)==record['checkpoint_sha256']
            model=make_model(key).cuda();model.load_state_dict(torch.load(path,map_location='cuda',weights_only=True));models[key]=model.eval()
        for scenario in ([3] if args.pilot else TEST):
            clean=windows(load_capture(scenario),scenario,preprocessing,limit=16 if args.pilot else None)
            for condition in (['clean','missing20'] if args.pilot else CONDITIONS):
                b=condition_batch(clean,condition,preprocessing);tag=f'{condition}_{scenario}';digest=input_hash(b)
                if args.pilot:manifest['case_hashes'][tag]=digest
                else:assert digest==old['case_hashes'][tag],tag
                for key,model in models.items():
                    row,cal,path=base.evaluate(model,b,seed,condition,scenario,thresholds[(seed,key)],training_records[(seed,key)],args,raw_directory)
                    rows.append(row);calibration.extend(cal);manifest['raw_hashes'][str(path.relative_to(ROOT))]=sha(path)
                del b
            del clean
        for name,records in [('summary',rows),('calibration',calibration),('curves',curves)]:base.write_csv(output/f'{name}.csv',records)
        print(json.dumps(dict(evaluation_seed_complete=seed,elapsed_seconds=time.monotonic()-started)),flush=True)
        del models;torch.cuda.empty_cache()
    manifest['wall_seconds']=time.monotonic()-started
    for relative,digest in manifest['source_hashes'].items():assert sha(ROOT/relative)==digest,relative
    if not args.pilot:
        assert len(rows)==975 and len(manifest['checkpoints'])==65 and len(manifest['raw_hashes'])==975
        assert sha(base_path)==manifest['base_manifest_sha256'] and sha(ROOT/'results/real/summary.csv')==manifest['base_summary_sha256']
        assert sha(ROOT/'results/real/curves.csv')==manifest['base_curves_sha256'] and sha(ROOT/'results/real/calibration.csv')==manifest['base_calibration_sha256']
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
