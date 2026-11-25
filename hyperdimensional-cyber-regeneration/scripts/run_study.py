#!/usr/bin/env python3
"""CUDA synthetic planning study; raw runs stay ignored, compact summaries are versioned."""
import argparse, csv, hashlib, json, os, random, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from regen import generate, Model, utilities, teacher_delta, topk_gate, apply_gate, semantic_loss, hdc_cleanup, without_provenance

def subset(data, idx):
    n=data["y"].shape[0]
    return {k:(v[idx] if torch.is_tensor(v) and v.ndim and v.shape[0]==n else v) for k,v in data.items()}

def features(batch, variant):
    x=batch["x"]
    if variant=="no_provenance":
        x=without_provenance(x)
    return x

PROVENANCE_COLUMNS=[9,10,11]

def expected_regret(p,b):
    actual=utilities(b["y"],b)
    return ((actual.max(-1,keepdim=True).values-actual)*torch.softmax(utilities(p,b)/.1,-1)).sum(-1).mean()

def predict(method, models, b, fraction=.1, threshold=.0):
    prior=b["prior"]
    if method=="B0":
        p=torch.minimum(prior,prior.gather(1,b["parent"]))
    elif method=="B3": p=hdc_cleanup(b)
    else:
        key={"B4":"P","B5":"P","random_gate":"P","dense_gate":"dense_gate"}.get(method,method)
        rec,logits=models[key](features(b,key),b["adj"])
        rec=rec.float().sigmoid()
        k=max(1,round(prior.shape[1]*fraction))
        if method in ["B1","B2","B4"]: p=rec
        else:
            q=logits.float().sigmoid()
            if method=="B5":
                q=-(prior*prior.clamp_min(1e-6).log()+(1-prior)*(1-prior).clamp_min(1e-6).log())
            elif method=="random_gate":
                rng=torch.Generator(device=prior.device).manual_seed(891)
                q=torch.rand(prior.shape,device=prior.device,generator=rng)
            g=topk_gate(q,k)
            if method not in ["B5","random_gate"]: g=g*(q>threshold)
            p=apply_gate(prior,rec,g)
    return p.clamp(1e-5,1-1e-5)

def train_model(key, train, val, args, seed, curves):
    torch.manual_seed(seed)
    model=Model(kind="dense" if key in ["B1","B2","dense_gate"] else "hdc").cuda()
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    n=len(train["y"]); batch=args.batch
    start=time.monotonic()
    for epoch in range(args.pretrain):
        model.train(); losses=[]
        for ids in torch.randperm(n,device="cuda").split(batch):
            b=subset(train,ids); x=features(b,key)
            mask=torch.rand(x.shape[:2],device="cuda")<.3
            with torch.autocast("cuda",dtype=torch.bfloat16):
                reconstruction=model.feature_reconstruction(x,b["adj"],mask)
            # GraphMAE-style scaled cosine feature reconstruction on masked nodes.
            loss=(1-F.cosine_similarity(reconstruction.float()[mask],x.float()[mask],dim=-1)).pow(2).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); losses.append(loss.detach())
        curves.append(dict(seed=seed,method=key,epoch=epoch,split="pretrain",loss=torch.stack(losses).mean().item()))
    best=float("inf"); wait=0; best_state=None
    for epoch in range(args.epochs):
        model.train(); losses=[]
        for ids in torch.randperm(n,device="cuda").split(batch):
            b=subset(train,ids)
            with torch.autocast("cuda",dtype=torch.bfloat16):
                rec_logits,gate_logits=model(features(b,key),b["adj"])
            rec=rec_logits.float().sigmoid(); q=gate_logits.float().sigmoid()
            hard=topk_gate(q,max(1,round(rec.shape[1]*.1)))
            gate=hard+q-q.detach()
            p=rec if key in ["B1","B2"] else apply_gate(b["prior"],rec,gate)
            loss=F.binary_cross_entropy_with_logits(rec_logits.float(),b["y"])
            if key!="B1":
                loss=loss+expected_regret(p,b)+F.binary_cross_entropy_with_logits(gate_logits.float(),b["teacher"])
                if key!="no_semantic": loss=loss+.1*semantic_loss(p,b)
                keep=((b["previous_y"]==b["y"]) & ((b["prior"]-b["y"]).abs()<.15)).float()
                loss=loss+.1*((p-b["prior"]).square()*keep).sum()/keep.sum().clamp_min(1)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5); opt.step()
            losses.append(loss.detach())
        model.eval()
        with torch.no_grad():
            scores=[]
            for ids in torch.arange(len(val["y"]),device="cuda").split(batch):
                b=subset(val,ids)
                p=predict(key,{key:model},b)
                scores.append(F.binary_cross_entropy(p,b["y"])+expected_regret(p,b))
            score=torch.stack(scores).mean().item()
        curves.append(dict(seed=seed,method=key,epoch=epoch,split="train",loss=torch.stack(losses).mean().item()))
        curves.append(dict(seed=seed,method=key,epoch=epoch,split="validation",loss=score))
        if score<best-1e-4:
            best=score; wait=0; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else: wait+=1
        if epoch%5==0: print(json.dumps({"seed":seed,"method":key,"epoch":epoch,"validation":score}),flush=True)
        if wait>=args.patience: break
    model.load_state_dict(best_state); model.eval()
    path=ROOT/"results/checkpoints"/f"{seed}_{key}.pt"; path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(best_state,path)
    return model,time.monotonic()-start

def conditions():
    out=[]
    for c in [0.,.1,.2,.4]:out.append(("severity",c,"targeted",64,2048,.1,"test"))
    for a in ["random","coherent"]:out.append(("attack",.2,a,64,2048,.1,"test"))
    for b in [0,512,8192]:out.append(("bandwidth",.2,"targeted",64,b,.1,"test"))
    for f in [128,256]:out.append(("size",.2,"coherent",f,2048,.1,"ood"))
    for f in [.05,.25,1.]:out.append(("repair",.2,"targeted",64,2048,f,"test"))
    return out

def write_csv(path,rows):
    if not rows:return
    keys=list(rows[0]); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="") as out:
        w=csv.DictWriter(out,fieldnames=keys);w.writeheader();w.writerows(rows)

def evaluate(method,models,data,args,seed,setting,training_seconds,threshold):
    name,c,a,nfacts,budget,fraction,family=setting
    chunks=[]
    with torch.inference_mode():
        for ids in torch.arange(len(data["y"]),device="cuda").split(args.batch):
            b=subset(data,ids); p=predict(method,models,b,fraction,threshold)
            u=utilities(p,b); actual=utilities(b["y"],b); top=u.topk(3,-1).indices
            # True feasibility is conjunction of required initial facts.
            feasible=(torch.bmm(b["req"],(1-b["y"]).unsqueeze(-1)).squeeze(-1)<.5)
            elig=feasible.any(-1)
            recall=feasible.gather(1,top).any(-1).float()
            regret=actual.max(-1).values-actual.gather(1,top[:,:1]).squeeze(-1)
            brier=(p-b["y"]).square().mean(-1)
            changed=(p-b["prior"]).abs()>1e-4
            false=((p-b["y"]).abs()>(b["prior"]-b["y"]).abs()+1e-4)&changed
            conf=(p<.1)|(p>.9); wrong=(p>.5)!=(b["y"]>.5)
            stable=(b["previous_y"]==b["y"])&((b["prior"]>.5)==(b["y"]>.5))
            planprob=torch.exp(torch.bmm(b["req"],p.log().unsqueeze(-1)).squeeze(-1))
            chunks.append({k:v.cpu().numpy() for k,v in dict(
                recall=recall,eligible=elig,regret=regret,brier=brier,group=b["group"],
                false_repair=false.float().sum(-1)/changed.sum(-1).clamp_min(1),
                false_certainty=(wrong&conf).float().sum(-1)/conf.sum(-1).clamp_min(1),
                coverage=conf.float().mean(-1),preservation=(changed&stable).float().sum(-1)/stable.sum(-1).clamp_min(1),
                planprob=planprob,plantruth=feasible.float()).items()})
        b=subset(data,slice(0,1))
        for _ in range(5): utilities(predict(method,models,b,fraction,threshold),b).topk(3,-1)
        torch.cuda.synchronize();base=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
        times=[]
        for _ in range(20):
            start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            start.record();utilities(predict(method,models,b,fraction,threshold),b).topk(3,-1);end.record()
            end.synchronize();times.append(start.elapsed_time(end))
        peak=(torch.cuda.max_memory_allocated()-base)/2**20
    arrays={k:np.concatenate([part[k] for part in chunks]) for k in chunks[0]}
    raw=ROOT/"results/runs"/str(seed);raw.mkdir(parents=True,exist_ok=True)
    tag=f"{method}_{name}_{c}_{a}_{nfacts}_{budget}_{fraction}"
    np.savez_compressed(raw/f"{tag}.npz",**arrays)
    eligible=arrays["eligible"].astype(bool)
    row=dict(seed=seed,method=method,condition=name,corruption=c,attack=a,facts=nfacts,budget=budget,repair_fraction=fraction,
        recall3=float(arrays["recall"][eligible].mean()) if eligible.any() else float("nan"),
        regret=float(arrays["regret"].mean()),brier=float(arrays["brier"].mean()),
        false_repair=float(arrays["false_repair"].mean()),false_certainty=float(arrays["false_certainty"].mean()),
        preservation=float(arrays["preservation"].mean()),coverage=float(arrays["coverage"].mean()),latency_ms=float(np.median(times)),
        peak_vram_mb=peak,train_seconds=training_seconds,bytes=float(data["bytes"].float().mean().item()),
        eligible=int(eligible.sum()),changed_report_fraction=float(data["changed_fraction"].mean().item()),intervention_fraction=float(data["intervention_fraction"].mean().item()))
    calibration=[]
    if name=="severity" and c==.2:
        probs=arrays["planprob"].flatten();truth=arrays["plantruth"].flatten()
        for j in range(10):
            mask=(probs>=j/10)&(probs<(j+1)/10 if j<9 else probs<=1)
            calibration.append(dict(seed=seed,method=method,bin=j,confidence=float(probs[mask].mean()) if mask.any() else (j+.5)/10,
              accuracy=float(truth[mask].mean()) if mask.any() else float("nan"),count=int(mask.sum())))
    return row,calibration

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--train",type=int,default=20000);ap.add_argument("--val",type=int,default=2000)
    ap.add_argument("--test",type=int,default=4000);ap.add_argument("--epochs",type=int,default=40)
    ap.add_argument("--pretrain",type=int,default=5);ap.add_argument("--patience",type=int,default=8)
    ap.add_argument("--batch",type=int,default=256);ap.add_argument("--lr",type=float,default=.001)
    ap.add_argument("--seeds",type=int,nargs="+",default=[11,23,37,51,71]);ap.add_argument("--pilot",action="store_true")
    args=ap.parse_args()
    assert torch.cuda.is_available() and "5000 Ada" in torch.cuda.get_device_name(0)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=True
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(2027)
    train=generate(args.train,12027,"cuda",family="train")
    val=generate(args.val,22027,"cuda",family="validation")
    train["teacher"]=torch.cat([topk_gate(teacher_delta(subset(train,ids),tau=.1),max(1,round(64*.1)),positive_only=True)
        for ids in torch.arange(args.train,device="cuda").split(128)])
    curves=[];rows=[];calibration=[];started=time.monotonic()
    source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/"src/regen.py",ROOT/"scripts/run_study.py"]}
    methods=["B0","B1","B2","B3","B4","B5","P","random_gate","no_provenance","no_semantic","dense_gate"]
    keys=["B1","B2","P","no_provenance","no_semantic","dense_gate"]
    if args.pilot: keys=["B1","P"];methods=["B0","B1","P"]
    hashes={k:hashlib.sha256(b["y"].cpu().numpy().tobytes()+b["req"].cpu().numpy().tobytes()).hexdigest() for k,b in [("train",train),("validation",val)]}
    for seed in args.seeds:
        models={};durations={}
        for key in keys:
            models[key],durations[key]=train_model(key,train,val,args,seed,curves)
        thresholds={}
        with torch.inference_mode():
            for key in keys:
                if key=="B1":continue
                scores=[]
                for threshold in [0.,.1,.3,.5]:
                    losses=[]
                    for ids in torch.arange(args.val,device="cuda").split(args.batch):
                        b=subset(val,ids);m=key
                        losses.append(expected_regret(predict(m,models,b,threshold=threshold),b))
                    scores.append(torch.stack(losses).mean().item())
                thresholds[key]=[0.,.1,.3,.5][int(np.argmin(scores))]
        for setting in conditions()[:1] if args.pilot else conditions():
            name,c,a,f,budget,frac,family=setting
            data=generate(args.test,32027,"cuda",facts=f,family=family,corruption=c,attack=a,budget=budget)
            for method in methods:
                key={"B4":"P","B5":"P","random_gate":"P","dense_gate":"dense_gate"}.get(method,method)
                row,cal=evaluate(method,models,data,args,seed,setting,durations.get(key,0),thresholds.get(key,0))
                rows.append(row);calibration.extend(cal)
            del data
        output=ROOT/("work/pilot" if args.pilot else "results")
        write_csv(output/"summary.csv",rows);write_csv(output/"calibration.csv",calibration);write_csv(output/"curves.csv",curves)
        print(json.dumps({"seed_complete":seed,"elapsed_seconds":time.monotonic()-started}),flush=True)
        del models;torch.cuda.empty_cache()
    manifest=dict(settings=vars(args),hardware=torch.cuda.get_device_name(0),torch=torch.__version__,cuda=torch.version.cuda,
        source_hashes=source_hashes,source_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),dataset_hashes=hashes,
        gpu_wall_seconds=time.monotonic()-started,
        deviations=["Independent planning snapshots with prior-time observations; no temporal recovery-delay estimate.",
          "64 training facts; 128/256-fact extrapolation. Facts are not host counts.",
          "40 max epochs, patience 8, batch 256; 5 masked pretraining epochs.",
          "No exhaustive hyperparameter search; validation selects gate threshold and early stopping.",
          "Full decoder executes before selection; gate sparsity does not establish compute savings.",
          "Latency: warmed CUDA batch-one model plus candidate scoring; excludes serialization/transfers.",
          "Peak VRAM is incremental allocation above resident-model/data baseline.",
          "Figures use seed-level confidence intervals; primary paired contrast separately uses hierarchical bootstrap.",
          "All methods use the same sparse report codec; HDC is internal memory, not a wire-compression claim.",
          "No independent link-loss/burst, no-evidence, or pair-repair sweep in this run.",
          "Synthetic controlled test; no external CybORG validation."])
    (output/"study_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")

if __name__=="__main__":main()
