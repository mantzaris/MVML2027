"""Frozen-checkpoint waveform stress derivatives, with new references and cards."""
from __future__ import annotations
import json,time
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy import signal
from scipy.special import softmax
import torch
from torch.utils.data import DataLoader
from underwater_reasoning import dsp
from underwater_reasoning.references import estimate
from underwater_reasoning.io import atomic_json,canonical_hash,file_hash
from underwater_reasoning.runtime import budget_check
from prepare_evidence import compact,vectors
from train_classifier import features,AudioImages,new_model,infer
ROOT=Path(__file__).resolve().parents[1]

def transform(x,fs,name,uid):
    rms=float(np.sqrt(np.mean(x*x)));rng=np.random.default_rng(int(canonical_hash([17,uid,name])[:8],16))
    if name.startswith('noise_'):
        snr=int(name.split('_')[1]);noise=rng.normal(size=len(x));noise*=rms/max(float(np.sqrt(np.mean(noise*noise))),1e-20)*10**(-snr/20);return (x+noise).astype('float32')
    if name=='lowpass_2000':return signal.sosfiltfilt(signal.butter(4,2000,fs=fs,output='sos'),x).astype('float32')
    if name=='gain_025':return (x*.25).astype('float32')
    if name=='clipping':return np.clip(x/max(rms,1e-20)*.4,-1,1).astype('float32')
    raise ValueError(name)

def main():
    if not json.loads((ROOT/'protocol/test_unlock_log.json').read_text()).get('unlocked'):raise ValueError('Test locked')
    started=time.monotonic();manifest=json.loads((ROOT/'data/private/experiment_manifest.json').read_text());lookup={r['id']:r for r in manifest['records']};labels=manifest['labels'];chosen=[]
    for label in labels:
        candidates=[lookup[uid] for uid in manifest['assessments']['test'] if lookup[uid]['label']==label]
        chosen+=sorted(candidates,key=lambda r:canonical_hash([17,'robustness',r['id']]))[:12]
    variants=['noise_20','noise_10','noise_0','lowpass_2000','gain_025','clipping'];rows=[]
    for row in chosen:
        x,fs=sf.read(ROOT/row['audio_path'],dtype='float32')
        for variant in variants:
            uid=row['id']+'_'+variant;path='data/raw/robustness/'+uid+'.wav';out=ROOT/path;out.parent.mkdir(parents=True,exist_ok=True)
            if not out.exists():sf.write(out,transform(x,fs,variant,row['id']),fs,subtype='FLOAT')
            rows.append({**row,'id':uid,'original_id':row['id'],'variant':variant,'audio_path':path,'partition':'robustness'})
    atomic_json(ROOT/'data/private/robustness_manifest.json',{'records':rows,'labels':labels,'source_clips':len(chosen),'variants':variants})
    feature_path=features(rows,cache_name='robustness');training=json.loads((ROOT/'results/manifests/classifier_training.json').read_text());anchor=training['seeds'][0]
    model=new_model();model.load_state_dict(torch.load(ROOT/anchor['checkpoint'],weights_only=False)['model']);loader=DataLoader(AudioImages(feature_path,rows,list(range(len(rows)))),batch_size=64,num_workers=4,pin_memory=True)
    logits,y,idx,emb=infer(model,loader,embedding=True);del model;torch.cuda.empty_cache()
    pred=np.load(ROOT/'data/derived/classifier_seed17.npz');positions={uid:i for i,uid in enumerate(pred['ids'])};coordinates=(emb-pred['pca_mean'])@pred['pca_components'].T
    train_ids=[r['id'] for r in manifest['records'] if r['partition']=='acoustic_fit'];bank=pred['embeddings'][[positions[uid] for uid in train_ids]]
    metadata=[{**lookup[uid],'anonymous_id':'T'+canonical_hash(uid)[:8]} for uid in train_ids];classifier_hash=file_hash(ROOT/'data/derived/classifier_seed17.npz')
    np.savez_compressed(ROOT/'data/derived/robustness_classifier.npz',ids=np.array([r['id'] for r in rows]),logits=logits,temperature=pred['temperature'])
    for i,row in enumerate(rows):
        budget_check();uid=row['id'];x,fs=sf.read(ROOT/row['audio_path'],dtype='float32');wavehash=dsp.record_hash(x,fs)
        atomic_json(ROOT/f'data/derived/references/{uid}.json',{'id':uid,'waveform_sha256':wavehash,'reference':estimate(x,fs)})
        cards={};times={}
        for family,func in zip('ABC',(dsp.spectrum,dsp.modulation,dsp.temporal)):
            t=time.perf_counter();cards[family]=func(x,fs);times[family]=time.perf_counter()-t
        t=time.perf_counter();neighbors=dsp.retrieve(emb[i],bank,metadata,row['source_group']);times['D']=time.perf_counter()-t
        compacted={f:compact(cards[f]) for f in 'ABC'};compacted['D']={'id':'D0','class_analogies_only':True,'neighbors':[{'id':n['exemplar_id'],'label':n['training_label'],'distance':round(n['cosine_distance'],4)} for n in neighbors]}
        probs=softmax(logits[i]/float(pred['temperature']));order=np.sort(probs)
        initial=[*probs.tolist(),float(-np.dot(probs,np.log(np.maximum(probs,1e-12)))),float(order[-1]-order[-2]),*coordinates[i].tolist(),float(np.mean(x*x)),float(np.mean(x==0)),float(np.mean(abs(x)>=1))]
        atomic_json(ROOT/f'data/derived/evidence/{uid}.json',{'id':uid,'waveform_sha256':wavehash,'cards':cards,'compact':compacted,'neighbors':neighbors,
            'prior':{'id':'P0','class_probabilities':{label:round(float(p),5) for label,p in zip(labels,probs)}},'initial_state':initial,
            'tool_vectors':vectors(cards,neighbors,labels),'tool_seconds':times,'classifier_sha256':classifier_hash})
        if (i+1)%30==0:print('robustness prepared',i+1,len(rows),flush=True)
    atomic_json(ROOT/'results/manifests/robustness_preparation.json',{'source_clips':len(chosen),'source_groups':len({r['source_group'] for r in chosen}),'transformed_clips':len(rows),'variants':variants,'seconds':time.monotonic()-started,'new_references':True,'retrained_models':False})

if __name__=='__main__':main()
