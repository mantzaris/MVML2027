"""Source-bootstrap ensemble for observed-state claim-risk acquisition."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from .acquisition import observed_state,choose_action
from .scoring import score,deterministic_report

class Network(torch.nn.Module):
    def __init__(self,width):
        super().__init__();self.layers=torch.nn.Sequential(torch.nn.Linear(width,64),torch.nn.ReLU(),torch.nn.Dropout(.1),
            torch.nn.Linear(64,32),torch.nn.ReLU(),torch.nn.Dropout(.1),torch.nn.Linear(32,5))
    def forward(self,x):return self.layers(x)

def state(asset,mask):
    selected=[a for a in range(4) if mask&(1<<a)]
    return observed_state(asset['initial_state'],selected,asset['tool_vectors'],16).astype('float32')

class Ensemble:
    def __init__(self,payload):
        self.mean=np.asarray(payload['mean'],dtype='float32');self.scale=np.asarray(payload['scale'],dtype='float32');self.members=[]
        for weights in payload['weights']:
            net=Network(len(self.mean));net.load_state_dict(weights);net.eval();self.members.append(net)
    def predict(self,asset,mask):
        x=torch.from_numpy((state(asset,mask)-self.mean)/self.scale)[None]
        with torch.inference_mode():v=np.stack([net(x).numpy()[0] for net in self.members])
        return v[:,:4].mean(0),v[:,:4].std(0),float(v[:,4].mean())
    def select(self,asset,beta=.5,cost_penalty=.02,budget=2,one_shot=False,force=False):
        mask=0;trace=[]
        if one_shot:
            means,deviations,_=self.predict(asset,0);values=means-beta*deviations-cost_penalty
            for action in np.argsort(-values,kind='stable')[:budget]:
                if force or values[action]>0:mask|=1<<int(action);trace.append(int(action))
            return mask,trace
        for _ in range(budget):
            means,deviations,_=self.predict(asset,mask);selected=[a for a in range(4) if mask&(1<<a)]
            action=choose_action(means,deviations,selected,budget=budget,beta=beta,cost_penalty=cost_penalty,force=force)
            if action is None:break
            mask|=1<<action;trace.append(action)
        return mask,trace

def load_study(root,parts,tag='main',require_terminal=True):
    root=Path(root);manifest=json.loads((root/'data/private/experiment_manifest.json').read_text());lookup={r['id']:r for r in manifest['records']};out={}
    for part in parts:
        out[part]=[]
        for uid in manifest['assessments'][part]:
            row=lookup[uid];asset=json.loads((root/f'data/derived/evidence/{uid}.json').read_text());ref=json.loads((root/f'data/derived/references/{uid}.json').read_text())['reference']
            reviews=[]
            for role in (0,1):
                path=root/f'results/raw/review/main/{part}/{uid}_{role}.json'
                record=json.loads(path.read_text()) if path.exists() else None
                reviews.append(record['result']['parsed'] if record and not record['result']['errors'] else None)
            item={'row':row,'asset':asset,'reference':ref,'reviews':reviews,'reports':{},'scores':{}}
            if require_terminal:
                for mask in list(range(16))+[-1]:
                    path=root/f'results/raw/terminal/{tag}/{part}/{uid}_{mask}.json';record=json.loads(path.read_text())
                    item['reports'][mask]=record
                    item['scores'][mask]=score(record['result']['parsed'],row['label'],ref,asset,mask,reviews,record['result']['errors'])
            out[part].append(item)
    return manifest,out

def deterministic_score(item,mask,labels):
    report=deterministic_report(item['asset'],mask,labels)
    return score(report,item['row']['label'],item['reference'],item['asset'],mask,item['reviews'])

def method_mask(name,item,settings,ensemble,class_ensemble,budget=2):
    asset=item['asset'];beta=settings['beta'];penalty=settings['lambda']
    if name=='B1_raw':return -1,[]
    if name=='B2_prior':return 0,[]
    if name=='B3_fixed':mask=settings['fixed_mask']
    elif name=='B4_retrieval':mask=8
    elif name=='B4_retrieval_dsp':mask=settings['retrieval_dsp_mask']
    elif name=='B5_all':mask=15
    elif name=='B8_deterministic':mask=settings['deterministic_mask']
    elif name=='B9_entropy':
        index=int(np.searchsorted(settings['entropy_edges'],asset['initial_state'][5],side='right'));mask=settings['entropy_masks'][index]
    elif name=='B6_planner':return item['planner']['mask'],item['planner']['actions']
    elif name=='B7_class_only':return class_ensemble.select(asset,settings['class_beta'],settings['class_lambda'],budget)
    elif name=='P':return ensemble.select(asset,beta,penalty,budget)
    elif name=='A1_one_shot':return ensemble.select(asset,beta,penalty,budget,one_shot=True)
    elif name=='A2_no_penalty':return ensemble.select(asset,0,penalty,budget)
    elif name=='A3_no_stop':return ensemble.select(asset,beta,penalty,budget,force=True)
    else:raise ValueError(name)
    return mask,[i for i in range(4) if mask&(1<<i)]
