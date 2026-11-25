#!/usr/bin/env python3
"""Fresh-test follow-up study. V1 sources/results/checkpoints remain immutable."""
import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from regen import generate, Model, utilities
from regen_v2 import RegeneratorV2, infer, loss_terms
from run_study import subset, predict as old_predict, conditions as old_conditions

NEW_KEYS=['JH','JD','IH','ID','JH_ERM']
OLD_KEYS=['B1','B2','P','no_provenance','no_semantic','dense_gate']
OLD_METHODS=['B0','B1','B2','B3','B4','B5','P','random_gate','no_provenance','no_semantic','dense_gate']
METHODS=NEW_KEYS+['JH_product']+OLD_METHODS
OUT=ROOT/'results/v2'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path,rows):
    if not rows:return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)


def mixture(n,seed,family):
    # Keep the exact old hidden world/candidate catalogue; alter observations only.
    data=generate(n,seed,'cuda',family=family,corruption=0.)
    index=torch.arange(n,device='cuda')
    groups=index%4;severity=(index//4)%3
    for group,attack in enumerate(['targeted','random','coherent'],1):
        for level,c in enumerate([.1,.2,.4]):
            other=generate(n,seed,'cuda',family=family,corruption=c,attack=attack)
            ids=(groups==group)&(severity==level)
            for key in data:
                data[key][ids]=other[key][ids]
            del other
    data['observation_group']=groups
    return data


def risk(loss,group,beta=2.,mean_only=False):
    means=torch.stack([loss[group==g].mean() for g in range(4) if (group==g).any()])
    return means.mean() if mean_only else (torch.logsumexp(beta*means,0)-math.log(len(means)))/beta


def predictions(method,models,batch,fraction=.1,threshold=0.):
    if method in NEW_KEYS+['JH_product']:
        return infer(models['JH' if method=='JH_product' else method],batch,
                     product_override=method=='JH_product')
    p=old_predict(method,models,batch,fraction,threshold)
    prob=((1-batch['req'])+batch['req']*p.unsqueeze(1)).prod(-1)
    return p,prob


def candidate_utilities(prob,batch):
    return batch['values']*prob-batch['costs']


def train_model(key,train,val,args,seed,curves,checkpoint_dir):
    torch.manual_seed(seed)
    model=RegeneratorV2(key).cuda()
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    n=len(train['y']);start=time.monotonic()
    for epoch in range(args.pretrain):
        model.train();losses=[]
        for ids in torch.randperm(n,device='cuda').split(args.batch):
            b=subset(train,ids);x=b['x'];mask=torch.rand(x.shape[:2],device='cuda')<.3
            with torch.autocast('cuda',dtype=torch.bfloat16):
                reconstructed=model.feature_reconstruction(x,b['adj'],mask)
            loss=(1-F.cosine_similarity(reconstructed.float()[mask],x.float()[mask],dim=-1)).square().mean()
            opt.zero_grad(set_to_none=True);loss.backward();opt.step();losses.append(loss.detach())
        curves.append(dict(seed=seed,method=key,epoch=epoch,split='pretrain',loss=torch.stack(losses).mean().item()))
    best=float('inf');wait=0;best_state=None;best_epoch=-1
    for epoch in range(args.epochs):
        model.train();losses=[]
        for ids in torch.randperm(n,device='cuda').split(args.batch):
            b=subset(train,ids)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                p,prob=infer(model,b)
            # Reductions/probabilities/losses are FP32 even under training autocast.
            total=loss_terms(p.float(),prob.float(),b,tau=.1)['total']
            loss=risk(total,b['observation_group'],args.risk_beta,key=='JH_ERM')
            opt.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();losses.append(loss.detach())
        model.eval()
        with torch.inference_mode():
            totals=[]
            for ids in torch.arange(len(val['y']),device='cuda').split(args.batch):
                b=subset(val,ids);p,prob=infer(model,b)
                totals.append(loss_terms(p,prob,b,tau=.1)['total'])
            score=risk(torch.cat(totals),val['observation_group'],mean_only=True).item()
        curves.append(dict(seed=seed,method=key,epoch=epoch,split='train',loss=torch.stack(losses).mean().item()))
        curves.append(dict(seed=seed,method=key,epoch=epoch,split='validation',loss=score))
        if score<best-1e-4:
            best=score;wait=0;best_epoch=epoch;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:wait+=1
        if epoch%5==0:print(json.dumps(dict(seed=seed,method=key,epoch=epoch,validation=score)),flush=True)
        if wait>=args.patience:break
    model.load_state_dict(best_state);model.eval()
    checkpoint_dir.mkdir(parents=True,exist_ok=True)
    path=checkpoint_dir/f'{seed}_{key}.pt';torch.save(best_state,path)
    return model,time.monotonic()-start,dict(seed=seed,method=key,epoch=best_epoch,score=best,
        checkpoint_sha256=sha(path),parameters=sum(p.numel() for p in model.parameters()))


def settings():
    result=[tuple(c)+(.2,) for c in old_conditions() if c[0]!='repair']
    for attack in ['random','coherent']:
        for c in [.1,.4]:result.append(('attack_severity',c,attack,64,2048,.1,'test',.2))
    for attack in ['targeted','random','coherent']:
        result.append(('missing',.2,attack,64,2048,.1,'test',.5))
    return result


def evaluate(method,models,data,args,seed,setting,duration,threshold,raw_dir):
    name,c,attack,facts,budget,fraction,family,missing=setting
    chunks=[]
    with torch.inference_mode():
        for ids in torch.arange(len(data['y']),device='cuda').split(args.batch):
            b=subset(data,ids);p,prob=predictions(method,models,b,fraction,threshold)
            actual=utilities(b['y'],b);predicted=candidate_utilities(prob,b)
            top=predicted.topk(3,-1).indices
            truth=(torch.bmm(b['req'],(1-b['y']).unsqueeze(-1)).squeeze(-1)<.5)
            eligible=truth.any(-1)
            changed=(p-b['prior']).abs()>1e-4
            false=((p-b['y']).abs()>(b['prior']-b['y']).abs()+1e-4)&changed
            confident=(p<.1)|(p>.9);wrong=(p>.5)!=(b['y']>.5)
            stable=(b['previous_y']==b['y'])&((b['prior']>.5)==(b['y']>.5))
            root_counts=torch.zeros_like(b['req']).scatter_add_(2,b['parent'][:,None,:].expand_as(b['req']),b['req'])
            duplicate=(root_counts-1).clamp_min(0).sum(-1)
            arrays=dict(recall=truth.gather(1,top).any(-1).float(),eligible=eligible,
                regret=actual.max(-1).values-actual.gather(1,top[:,:1]).squeeze(-1),
                brier=(p-b['y']).square().mean(-1),plan_brier=(prob-truth.float()).square().mean(-1),
                group=b['group'],false_repair=false.float().sum(-1)/changed.sum(-1).clamp_min(1),
                false_certainty=(confident&wrong).float().sum(-1)/confident.sum(-1).clamp_min(1),
                coverage=confident.float().mean(-1),preservation=(changed&stable).float().sum(-1)/stable.sum(-1).clamp_min(1),
                logical_violation=(p>p.gather(1,b['parent'])+1e-6).float().mean(-1),
                changed_fraction=changed.float().mean(-1),planprob=prob,plantruth=truth.float(),duplicate_roots=duplicate)
            chunks.append({k:v.cpu().numpy() for k,v in arrays.items()})
        b=subset(data,slice(0,1))
        for _ in range(5):
            p,prob=predictions(method,models,b,fraction,threshold);candidate_utilities(prob,b).topk(3,-1)
        torch.cuda.synchronize();baseline=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();times=[]
        for _ in range(30):
            start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            start.record();p,prob=predictions(method,models,b,fraction,threshold)
            candidate_utilities(prob,b).topk(3,-1);end.record();end.synchronize();times.append(start.elapsed_time(end))
        peak=(torch.cuda.max_memory_allocated()-baseline)/2**20
    arrays={k:np.concatenate([part[k] for part in chunks]) for k in chunks[0]}
    tag=f'{method}_{name}_{c}_{attack}_{facts}_{budget}_{fraction}'
    path=raw_dir/str(seed);path.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path/f'{tag}.npz',**arrays)
    eligible=arrays['eligible'].astype(bool)
    row=dict(seed=seed,method=method,condition=name,corruption=c,attack=attack,facts=facts,budget=budget,
        repair_fraction=fraction,missing=missing,
        training_origin='v2_mixture' if method in NEW_KEYS+['JH_product'] else 'v1_checkpoint_reused',
        repair_policy='budgeted' if method in ['P','B5','random_gate','no_provenance','no_semantic','dense_gate'] else 'full_or_fixed',
        recall3=float(arrays['recall'][eligible].mean()),regret=float(arrays['regret'].mean()),
        brier=float(arrays['brier'].mean()),plan_brier=float(arrays['plan_brier'].mean()),
        false_repair=float(arrays['false_repair'].mean()),false_certainty=float(arrays['false_certainty'].mean()),
        coverage=float(arrays['coverage'].mean()),preservation=float(arrays['preservation'].mean()),
        logical_violation=float(arrays['logical_violation'].mean()),changed_fraction=float(arrays['changed_fraction'].mean()),
        latency_ms=float(np.median(times)),latency_p95_ms=float(np.quantile(times,.95)),peak_vram_mb=peak,
        train_seconds=duration,bytes=float(data['bytes'].float().mean()),eligible=int(eligible.sum()),
        changed_report_fraction=float(data['changed_fraction'].mean()),intervention_fraction=float(data['intervention_fraction'].mean()))
    calibration=[];dependency=[]
    if name=='severity' and c==.2:
        prob=arrays['planprob'].flatten();truth=arrays['plantruth'].flatten()
        for j in range(10):
            mask=(prob>=j/10)&(prob<(j+1)/10 if j<9 else prob<=1)
            calibration.append(dict(seed=seed,method=method,condition=name,attack=attack,corruption=c,bin=j,
                confidence=float(prob[mask].mean()) if mask.any() else (j+.5)/10,
                accuracy=float(truth[mask].mean()) if mask.any() else float('nan'),count=int(mask.sum())))
        duplicate=arrays['duplicate_roots'].flatten()
        for j in range(5):
            mask=duplicate==j
            dependency.append(dict(seed=seed,method=method,duplicate_roots=j,count=int(mask.sum()),
                plan_brier=float(((prob[mask]-truth[mask])**2).mean()) if mask.any() else float('nan'),
                mean_probability=float(prob[mask].mean()) if mask.any() else float('nan'),
                mean_truth=float(truth[mask].mean()) if mask.any() else float('nan')))
    return row,calibration,dependency


def main():
    config=json.loads((ROOT/'configs/study_v2.json').read_text())
    ap=argparse.ArgumentParser();ap.add_argument('--pilot',action='store_true');args=ap.parse_args()
    for key in ['train','val','test','train_seed','validation_seed','test_seed','seeds','batch','epochs','pretrain','patience','lr','risk_beta']:
        setattr(args,key,config[key])
    if args.pilot:
        args.train=1000;args.val=256;args.test=128;args.epochs=2;args.pretrain=1;args.seeds=[11];args.test_seed=config['pilot_test_seed']
    if not torch.cuda.is_available() or '5000 Ada' not in torch.cuda.get_device_name():raise RuntimeError('Ada CUDA required')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True;torch.use_deterministic_algorithms(True)
    torch.manual_seed(2027)
    output=ROOT/'work/v2_pilot' if args.pilot else OUT
    raw_dir=output/'runs' if args.pilot else ROOT/'results/runs/v2'
    checkpoint_dir=output/'checkpoints' if args.pilot else ROOT/'results/checkpoints/v2'
    output.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();train=mixture(args.train,args.train_seed,'train');val=mixture(args.val,args.validation_seed,'validation')
    if not args.pilot:
        old_manifest=json.loads((ROOT/'results/study_manifest.json').read_text())
        for name,b in [('train',train),('validation',val)]:
            fingerprint=hashlib.sha256(b['y'].cpu().numpy().tobytes()+b['req'].cpu().numpy().tobytes()).hexdigest()
            if fingerprint!=old_manifest['dataset_hashes'][name]:raise AssertionError('Mixture changed hidden worlds: '+name)
    source_paths=['src/regen.py','src/regen_v2.py','scripts/run_study.py','scripts/run_v2.py','configs/study_v2.json']
    source_hashes={p:sha(ROOT/p) for p in source_paths}
    manifest=dict(settings=vars(args),hardware=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_hashes=source_hashes,
        data_hashes={key:hashlib.sha256(b['x'].cpu().numpy().tobytes()+b['y'].cpu().numpy().tobytes()+b['req'].cpu().numpy().tobytes()).hexdigest()
                     for key,b in [('train',train),('validation',val)]},frozen_protocol=config,checkpoints=[],test_hashes={},legacy_checkpoint_hashes={},
        legacy_thresholds_sha256=sha(ROOT/'results/selected_thresholds.json'))
    durations_old={}
    for r in csv.DictReader((ROOT/'results/summary.csv').open()):durations_old[(int(r['seed']),r['method'])]=float(r['train_seconds'])
    threshold_rows=json.loads((ROOT/'results/selected_thresholds.json').read_text())['rows']
    threshold_map={(int(r['seed']),r['method']):float(r['threshold']) for r in threshold_rows}
    curves=[];rows=[];calibration=[];dependency=[]
    keys=['JH','ID'] if args.pilot else NEW_KEYS
    methods=['JH','ID','B0','B2','P'] if args.pilot else METHODS
    for seed in args.seeds:
        models={};durations={}
        for key in keys:
            models[key],durations[key],record=train_model(key,train,val,args,seed,curves,checkpoint_dir)
            manifest['checkpoints'].append(record)
        for key in OLD_KEYS:
            model=Model(kind='dense' if key in ['B1','B2','dense_gate'] else 'hdc').cuda()
            path=ROOT/'results/checkpoints'/f'{seed}_{key}.pt'
            manifest['legacy_checkpoint_hashes'][str(path.relative_to(ROOT))]=sha(path)
            model.load_state_dict(torch.load(path,map_location='cuda',weights_only=True));models[key]=model.eval()
        for setting in settings()[:1] if args.pilot else settings():
            name,c,attack,facts,budget,fraction,family,missing=setting
            data=generate(args.test,args.test_seed,'cuda',family=family,facts=facts,corruption=c,attack=attack,budget=budget,missing=missing)
            tag=f'{name}_{c}_{attack}_{facts}_{budget}_{fraction}'
            if seed==args.seeds[0]:
                manifest['test_hashes'][tag]=hashlib.sha256(data['x'].cpu().numpy().tobytes()+data['y'].cpu().numpy().tobytes()+data['req'].cpu().numpy().tobytes()).hexdigest()
            for method in methods:
                key={'B4':'P','B5':'P','random_gate':'P','JH_product':'JH'}.get(method,method)
                duration=durations[key] if key in durations else durations_old.get((seed,method),0.)
                row,cal,dep=evaluate(method,models,data,args,seed,setting,duration,threshold_map.get((seed,key),0.),raw_dir)
                rows.append(row);calibration.extend(cal);dependency.extend(dep)
            del data
        for name,records in [('summary',rows),('curves',curves),('calibration',calibration),('dependency_bins',dependency)]:write_csv(output/f'{name}.csv',records)
        print(json.dumps(dict(seed_complete=seed,elapsed_seconds=time.monotonic()-started)),flush=True)
        del models;torch.cuda.empty_cache()
    manifest['wall_seconds']=time.monotonic()-started
    manifest['methods']=methods;manifest['conditions']=[list(s) for s in settings()]
    manifest['scope']=['Fresh test seed42027; v1test32027 motivated the design and is notconfirmatory.',
        'Controlled static prerequisite snapshots, not full operational cyber episodes.',
        'Full regeneration changes repair policy; primary comparison uses full models on identical mixedtraining data.',
        'Joint probabilities exact only under assumed conditional star-forest posterior, not arbitrary true posterior.',
        'CUDA times warmed inference plus candidate scoring; incremental peak allocation excludesresidentstate.',
        'Five seeds share worlds; bootstrap crosses seeds and paired topology clusters.',
        'Original methods retain oldcheckpointtraining and validation thresholds; mixeddata controls isolate trainingchanges.',
        'All methods use the same sparse report codec; HDC remains internal retrievalfeatures.']
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

if __name__=='__main__':main()
