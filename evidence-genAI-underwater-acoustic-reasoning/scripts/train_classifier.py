"""Swin V2 Tiny fitting with development-only selection and atomic resume.

The script never computes test classification metrics. Test logits/embeddings
are saved privately for the later locked analysis, with source identities hidden
from every model prompt. Raw recordings remain native 32 kHz.
"""
from __future__ import annotations
import argparse,json,os,time
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','8')
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset,DataLoader
from torchvision.models import swin_v2_t,Swin_V2_T_Weights
from sklearn.metrics import f1_score,accuracy_score
from sklearn.decomposition import PCA
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from underwater_reasoning.io import atomic_json,file_hash,canonical_hash
from underwater_reasoning.runtime import budget_check,seed_all,rng_state,restore_rng,atomic_torch

ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(8)
torch.set_float32_matmul_precision('high')

def features(rows):
    path=ROOT/'data/derived/mel_images.npy';ids=ROOT/'data/derived/mel_ids.json'
    if path.exists():
        if json.loads(ids.read_text())!=[r['id'] for r in rows]:raise ValueError('Feature ID mismatch')
        return path
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.part.npy');bank=np.lib.format.open_memmap(tmp,mode='w+',dtype='float16',shape=(len(rows),256,256))
    mel=torchaudio.transforms.MelSpectrogram(sample_rate=20000,n_fft=1024,win_length=1024,hop_length=256,
        f_min=20,f_max=8000,n_mels=128,power=2,normalized=False,mel_scale='htk')
    for i,row in enumerate(rows):
        budget_check();x,fs=sf.read(ROOT/row['audio_path'],dtype='float32')
        if fs!=32000:raise ValueError('Unexpected native sample rate')
        y=torch.from_numpy(resample_poly(x,5,8).astype('float32'))
        with torch.no_grad():
            m=mel(y);db=10*torch.log10(m.clamp_min(1e-12));db=(db-db.max()).clamp(-90,0)
            img=F.interpolate(((db+90)/90)[None,None],size=(256,256),mode='bilinear',align_corners=False)[0,0]
        bank[i]=img.numpy()
        if (i+1)%500==0:print('features',i+1,len(rows),flush=True);bank.flush()
    bank.flush();del bank;tmp.replace(path);atomic_json(ids,[r['id'] for r in rows]);return path

class AudioImages(Dataset):
    def __init__(self,path,rows,indices):self.bank=np.load(path,mmap_mode='r');self.rows=rows;self.indices=indices
    def __len__(self):return len(self.indices)
    def __getitem__(self,j):
        i=self.indices[j];x=torch.tensor(np.array(self.bank[i],copy=True),dtype=torch.float32)[None].expand(3,-1,-1)
        x=(x-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]
        return x,self.rows[i]['label_index'],i

def new_model():
    model=swin_v2_t(weights=Swin_V2_T_Weights.IMAGENET1K_V1);model.head=torch.nn.Linear(model.head.in_features,5)
    return model.cuda()

def infer(model,loader,embedding=False):
    model.eval();logits=[];labels=[];indices=[];vectors=[]
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for x,y,idx in loader:
            x=x.cuda(non_blocking=True)
            if embedding:
                h=model.flatten(model.avgpool(model.permute(model.norm(model.features(x)))));out=model.head(h);vectors.append(h.float().cpu().numpy())
            else:out=model(x)
            logits.append(out.float().cpu().numpy());labels.extend(y.tolist());indices.extend(idx.tolist())
    return np.concatenate(logits),np.asarray(labels),np.asarray(indices),np.concatenate(vectors) if vectors else None

def fit(name,seed,lr,path,rows,max_epochs):
    complete=ROOT/f'results/raw/classifier/{name}.json';best=ROOT/f'checkpoints/{name}_best.pt';last=ROOT/f'checkpoints/{name}_last.pt'
    if complete.exists():return json.loads(complete.read_text())
    seed_all(seed);train=DataLoader(AudioImages(path,rows,[i for i,r in enumerate(rows) if r['partition']=='acoustic_fit']),batch_size=32,shuffle=True,num_workers=4,pin_memory=True)
    dev=DataLoader(AudioImages(path,rows,[i for i,r in enumerate(rows) if r['partition']=='development']),batch_size=64,num_workers=4,pin_memory=True)
    model=new_model();backbone=[p for n,p in model.named_parameters() if not n.startswith('head.')]
    opt=torch.optim.AdamW([{'params':backbone,'lr':lr},{'params':model.head.parameters(),'lr':1e-3}],weight_decay=.01)
    history=[];best_f=-1;bad=0;start=0;restore_check=None
    if last.exists():
        cp=torch.load(last,weights_only=False,map_location='cpu');model.load_state_dict(cp['model']);opt.load_state_dict(cp['optimizer']);restore_rng(cp['rng'])
        start=cp['epoch']+1;history=cp['history'];best_f=cp['best_f'];bad=cp['bad'];restore_check=cp.get('restore_check')
    t=time.monotonic()
    for epoch in range(start,max_epochs):
        budget_check(reserve_seconds=600);model.train();loss_sum=0;count=0
        for x,y,_ in train:
            budget_check();x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True);opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):loss=F.cross_entropy(model(x),y)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step();loss_sum+=loss.item()*len(y);count+=len(y)
        logits,y,_,_=infer(model,dev);score=float(f1_score(y,logits.argmax(1),average='macro',labels=range(5),zero_division=0))
        history.append({'epoch':epoch+1,'train_loss':loss_sum/count,'dev_macro_f1':score,'dev_accuracy':float(accuracy_score(y,logits.argmax(1))),'elapsed_seconds':time.monotonic()-t})
        if score>best_f+1e-8:best_f=score;bad=0;atomic_torch(best,{'model':model.state_dict(),'seed':seed,'lr':lr,'epoch':epoch+1,'dev_macro_f1':score})
        else:bad+=1
        cp={'model':model.state_dict(),'optimizer':opt.state_dict(),'rng':rng_state(),'epoch':epoch,'history':history,'best_f':best_f,'bad':bad,'restore_check':restore_check}
        atomic_torch(last,cp)
        if restore_check is None:
            # Real model/optimizer/RNG serialization round trip at an epoch boundary.
            expected=torch.rand(16,device='cuda');saved=torch.load(last,weights_only=False,map_location='cpu')
            model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);restore_rng(saved['rng']);actual=torch.rand(16,device='cuda')
            restored,_,_,_=infer(model,dev)
            restore_check={'rng_exact':bool(torch.equal(expected,actual)),'dev_logits_exact':bool(np.array_equal(logits,restored)),
                'optimizer_states_restored':len(opt.state),'resume_epoch':epoch+2}
            if not restore_check['rng_exact'] or not restore_check['dev_logits_exact']:raise ValueError('Checkpoint recovery failed')
            restore_rng(saved['rng'])
        print(name,json.dumps(history[-1]),flush=True)
        atomic_json(ROOT/f'results/raw/classifier/{name}_progress.json',{'history':history,'restore_check':restore_check})
        if bad>=5:break
    result={'name':name,'seed':seed,'backbone_lr':lr,'head_lr':.001,'weight_decay':.01,'batch_size':32,'max_epochs':max_epochs,'patience':5,
        'history':history,'best_dev_macro_f1':best_f,'checkpoint':str(best.relative_to(ROOT)),'checkpoint_sha256':file_hash(best),
        'restore_check':restore_check,'peak_vram_bytes':torch.cuda.max_memory_allocated(),'elapsed_seconds':time.monotonic()-t}
    atomic_json(complete,result);del model,opt;torch.cuda.empty_cache();return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--max-epochs',type=int,default=30);a=p.parse_args()
    audit=json.loads((ROOT/'results/manifests/audio_download_audit.json').read_text())
    if audit['cross_partition_hash_collisions']:raise ValueError('Unresolved waveform leakage')
    manifest=json.loads((ROOT/'data/private/experiment_manifest.json').read_text());rows=manifest['records'];path=features(rows)
    trials=[fit(f'search17_lr{lr:g}',17,lr,path,rows,a.max_epochs) for lr in (1e-6,1e-5,3e-5)]
    anchor=max(trials,key=lambda r:r['best_dev_macro_f1']);lr=anchor['backbone_lr']
    repeats=[anchor]+[fit(f'seed{seed}_lr{lr:g}',seed,lr,path,rows,a.max_epochs) for seed in (29,43)]
    all_loader=DataLoader(AudioImages(path,rows,list(range(len(rows)))),batch_size=64,num_workers=4,pin_memory=True)
    for result in repeats:
        seed_all(result['seed']);model=new_model();model.load_state_dict(torch.load(ROOT/result['checkpoint'],weights_only=False)['model'])
        logits,y,idx,emb=infer(model,all_loader,embedding=True)
        assert np.array_equal(idx,np.arange(len(rows)))
        dev=np.array([r['partition']=='development' for r in rows]);train=np.array([r['partition']=='acoustic_fit' for r in rows])
        def nll(logtemp):
            z=logits[dev]/np.exp(logtemp);return float(np.mean(logsumexp(z,axis=1)-z[np.arange(sum(dev)),y[dev]]))
        solution=minimize_scalar(nll,bounds=(-3,3),method='bounded');temperature=float(np.exp(solution.x))
        pca=PCA(n_components=32,random_state=17).fit(emb[train]);coords=pca.transform(emb)
        out=ROOT/f'data/derived/classifier_seed{result["seed"]}.npz'
        np.savez_compressed(out,ids=np.array([r['id'] for r in rows]),logits=logits,embeddings=emb,pca=coords,temperature=temperature,
            pca_components=pca.components_,pca_mean=pca.mean_)
        result['temperature']=temperature;result['inference_sha256']=file_hash(out);result['pca_explained_variance']=float(pca.explained_variance_ratio_.sum())
        del model;torch.cuda.empty_cache()
    atomic_json(ROOT/'results/manifests/classifier_training.json',{'trials':trials,'selected_backbone_lr':lr,'seeds':repeats,
        'selection_partition':'development','test_metrics_examined':False,'feature_config':{'sample_rate':20000,'mel_bins':128,'fft':1024,'hop':256,'fmin':20,'fmax':8000,'dynamic_range_db':90,'image':[256,256]},
        'torch':torch.__version__,'device':torch.cuda.get_device_name(),'completed_at_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()})
    print('CLASSIFIER COMPLETE',flush=True)

if __name__=='__main__':main()
