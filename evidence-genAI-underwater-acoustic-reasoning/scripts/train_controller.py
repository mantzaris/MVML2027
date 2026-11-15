"""Fit policy on policy-fit only and freeze all choices before test unlock."""
from __future__ import annotations
import json,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from underwater_reasoning.controller import Network,Ensemble,state,load_study,deterministic_score,method_mask
from underwater_reasoning.io import atomic_json,canonical_hash,file_hash
from underwater_reasoning.runtime import seed_all,atomic_torch
ROOT=Path(__file__).resolve().parents[1];torch.set_num_threads(4)
METHODS=['B1_raw','B2_prior','B3_fixed','B4_retrieval','B4_retrieval_dsp','B5_all','B6_planner','B7_class_only','B8_deterministic','B9_entropy','P','A1_one_shot','A2_no_penalty','A3_no_stop']

def dataset(items,target):
    xs=[];ys=[];available=[];groups=[]
    for item in items:
        losses={m:item['scores'][m][target] for m in range(16)}
        for mask in range(16):
            xs.append(state(item['asset'],mask));ys.append([losses[mask]-losses[mask|(1<<a)] if not mask&(1<<a) else 0. for a in range(4)]+[losses[mask]])
            available.append([not bool(mask&(1<<a)) for a in range(4)]+[True]);groups.append(item['row']['source_group'])
    return np.array(xs,dtype='float32'),np.array(ys,dtype='float32'),np.array(available,dtype='float32'),np.array(groups)

def fit(train,dev,seed,target):
    x,y,mask,groups=dataset(train,target);dx,dy,dm,_=dataset(dev,target);mean=x.mean(0);scale=x.std(0);scale[scale<1e-6]=1
    x=(x-mean)/scale;dx=(dx-mean)/scale;weights=[];histories=[]
    gx=np.unique(groups)
    for member in range(3):
        seed_all(seed+member*1009);rng=np.random.default_rng(seed+member*1009);sample=rng.choice(gx,len(gx),replace=True)
        index=np.concatenate([np.flatnonzero(groups==g) for g in sample]);tx=torch.tensor(x[index]);ty=torch.tensor(y[index]);tm=torch.tensor(mask[index])
        net=Network(x.shape[1]);opt=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=1e-4);best=float('inf');bad=0;history=[];best_weights=None
        for epoch in range(100):
            net.train();permutation=torch.randperm(len(tx))
            for batch in permutation.split(128):
                opt.zero_grad();pred=net(tx[batch]);loss=(torch.nn.functional.huber_loss(pred,ty[batch],reduction='none')*tm[batch]).sum()/tm[batch].sum();loss.backward();opt.step()
            net.eval()
            with torch.inference_mode():validation=float((torch.nn.functional.huber_loss(net(torch.tensor(dx)),torch.tensor(dy),reduction='none')*torch.tensor(dm)).sum()/dm.sum())
            history.append(validation)
            if validation<best-1e-7:best=validation;bad=0;best_weights={k:v.detach().clone() for k,v in net.state_dict().items()}
            else:bad+=1
            if bad>=10:break
        weights.append(best_weights);histories.append({'member':member,'epochs':len(history),'validation_huber':history,'bootstrap_unique_groups':len(set(sample))})
    return {'seed':seed,'target':target,'mean':mean.tolist(),'scale':scale.tolist(),'weights':weights,'histories':histories,'training_sources':len(gx),'train_rows':len(x)}

def empirical(items,masks,target='loss'):
    return float(np.mean([item['scores'][int(mask)][target] for item,mask in zip(items,masks)]))

def main():
    started=time.monotonic();manifest,data=load_study(ROOT,['policy_fit','development','calibration']);train,dev,cal=[data[p] for p in ('policy_fit','development','calibration')]
    for part in ('development','calibration'):
        for item in data[part]:item['planner']=json.loads((ROOT/f'results/raw/planner/{part}/{item["row"]["id"]}.json').read_text())
    # Feasibility gate: all matrices exist; parse validity and heterogeneous useful tools.
    valid=np.mean([item['scores'][m]['valid'] for item in dev for m in range(16)])
    if valid<.98:raise ValueError(f'Development parse/schema gate failed: {valid:.3f}; corrective pass required')
    for seed in (17,29,43):
        for target in ('loss','class_loss'):
            path=ROOT/f'checkpoints/controller_{target}_{seed}.pt'
            if not path.exists():atomic_torch(path,fit(train,dev,seed,target))
    ensemble=Ensemble(torch.load(ROOT/'checkpoints/controller_loss_17.pt',weights_only=False));class_ensemble=Ensemble(torch.load(ROOT/'checkpoints/controller_class_loss_17.pt',weights_only=False))
    pairs=[m for m in range(16) if m.bit_count()==2]
    fixed=min(pairs,key=lambda m:empirical(dev,[m]*len(dev)));retrieval=min([9,10,12],key=lambda m:empirical(dev,[m]*len(dev)))
    deterministic=min(pairs,key=lambda m:np.mean([deterministic_score(i,m,manifest['labels'])['loss'] for i in dev]))
    candidates=[]
    for beta in (0,.5,1):
        for penalty in (0,.02,.05):
            masks=[ensemble.select(i['asset'],beta,penalty)[0] for i in dev]
            candidates.append({'beta':beta,'lambda':penalty,'dev_loss':empirical(dev,masks),'mean_tools':float(np.mean([m.bit_count() for m in masks]))})
    chosen=min(candidates,key=lambda c:(c['dev_loss'],c['mean_tools'],c['beta'],c['lambda']))
    class_candidates=[]
    for beta in (0,.5,1):
        for penalty in (0,.02,.05):
            masks=[class_ensemble.select(i['asset'],beta,penalty)[0] for i in dev]
            class_candidates.append({'beta':beta,'lambda':penalty,'dev_loss':empirical(dev,masks,'class_loss'),'mean_tools':float(np.mean([m.bit_count() for m in masks]))})
    class_chosen=min(class_candidates,key=lambda c:(c['dev_loss'],c['mean_tools'],c['beta'],c['lambda']))
    entropy=[i['asset']['initial_state'][5] for i in dev];edges=np.quantile(entropy,[1/3,2/3]).tolist();entropy_masks=[]
    for b in range(3):
        members=[i for i,e in zip(dev,entropy) if int(np.searchsorted(edges,e,side='right'))==b]
        entropy_masks.append(min(pairs,key=lambda m:empirical(members,[m]*len(members))) if members else fixed)
    settings={'beta':chosen['beta'],'lambda':chosen['lambda'],'class_beta':class_chosen['beta'],'class_lambda':class_chosen['lambda'],
        'fixed_mask':fixed,'retrieval_dsp_mask':retrieval,'deterministic_mask':deterministic,'entropy_edges':edges,'entropy_masks':entropy_masks}
    fixed_loss=empirical(dev,[fixed]*len(dev));entropy_loss=empirical(dev,[method_mask('B9_entropy',i,settings,ensemble,class_ensemble)[0] for i in dev])
    settings['locked_comparator']='B3_fixed' if fixed_loss<=entropy_loss else 'B9_entropy'
    thresholds={};summary={}
    for method in METHODS:
        masks=[method_mask(method,i,settings,ensemble,class_ensemble)[0] for i in cal]
        risks=[ensemble.predict(i['asset'],max(m,0))[2] for i,m in zip(cal,masks)]
        # Empirical upper order statistic: ties are retained, actual coverage reported.
        threshold=float(np.sort(risks)[max(0,int(np.ceil(.7*len(risks)))-1)]);thresholds[method]=threshold
        dev_masks=[method_mask(method,i,settings,ensemble,class_ensemble)[0] for i in dev]
        scores=[deterministic_score(i,m,manifest['labels']) if method=='B8_deterministic' else i['scores'][m] for i,m in zip(dev,dev_masks)]
        summary[method]={'dev_loss':float(np.mean([s['loss'] for s in scores])),'dev_supported_coverage':float(np.mean([s['supported']/4 for s in scores])),
            'mean_tools':float(np.mean([max(m,0).bit_count() for m in dev_masks])),'calibration_coverage':float(np.mean(np.array(risks)<=threshold))}
    settings['thresholds']=thresholds;atomic_json(ROOT/'configs/locked_controller.json',settings)
    complementarity={'dev_parse_schema_validity':float(valid),'best_fixed_pair':fixed,'oracle_pair_loss':float(np.mean([min(i['scores'][m]['loss'] for m in pairs) for i in dev])),
        'fixed_pair_loss':fixed_loss,'per_clip_best_pair_counts':{str(m):sum(min(pairs,key=lambda n:i['scores'][n]['loss'])==m for i in dev) for m in pairs}}
    atomic_json(ROOT/'results/manifests/controller_training.json',{'settings':settings,'candidate_search':candidates,'class_candidate_search':class_candidates,'development':summary,
        'complementarity':complementarity,'seeds':[17,29,43],'seconds':time.monotonic()-started,'test_outcomes_examined':False,
        'checkpoint_hashes':{p.name:file_hash(p) for p in (ROOT/'checkpoints').glob('controller*.pt')}})
    atomic_json(ROOT/'protocol/test_unlock_log.json',{'unlocked':False,'status':'ready_for_locked_test_after_pilot_review','prepared_at_utc':datetime.now(timezone.utc).isoformat(),
        'controller_config_sha256':file_hash(ROOT/'configs/locked_controller.json'),'terminal_prompt_sha256':file_hash(ROOT/'prompts/terminal_v1.txt'),
        'scoring_sha256':file_hash(ROOT/'src/underwater_reasoning/scoring.py'),'confirmatory_contrasts':['P vs '+settings['locked_comparator'],'P vs B7_class_only','P vs A1_one_shot'],
        'interpretation':'Automated proxy endpoints with no human participants; source clusters and limited development groups constrain inference.'})
    print(json.dumps({'settings':settings,'complementarity':complementarity,'development':summary},indent=2),flush=True)

if __name__=='__main__':main()
