"""CUDA preprocessing and fixed windows for captured CTU-13 flow tables.

CPU is used only to decode the saved numeric arrays. Scaling, corruption,
window construction and observed-endpoint graph operations remain on CUDA.
"""
from pathlib import Path
import hashlib
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
TRAIN=[1,2,3,4,5,6,7,8,13]
VALIDATION=[9]
TEST=[10,11,12]
CONDITIONS=['clean','missing20','missing50','spoof20','spoof40']
WINDOW=32


def require_cuda():
    if not torch.cuda.is_available() or '5000 Ada' not in torch.cuda.get_device_name():
        raise RuntimeError('The attached RTX 5000 Ada GPU is required; no CPU fallback')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_capture(scenario):
    require_cuda()
    path=ROOT/'data/external/ctu13/parsed'/f'scenario_{scenario:02d}.npz'
    with np.load(path,allow_pickle=False) as archive:
        data={key:torch.from_numpy(archive[key].view(np.int64) if archive[key].dtype==np.uint64 else archive[key]).to('cuda')
              for key in ['features','observed','label','timestamp','row_id','src','dst']}
    data['features']=data['features'].float();data['observed']=data['observed'].bool();data['label']=data['label'].float()
    if not torch.isfinite(data['features']).all():raise ValueError('Parser left nonfinite input features')
    if not (data['timestamp'][1:]>=data['timestamp'][:-1]).all():raise ValueError('Capture must be sorted chronologically')
    return data


def fit_preprocessing(captures):
    # Only the first six continuous/log features are standardized. Ports are
    # already scaled to [0,1]; protocol/state flags remain binary.
    count=torch.zeros(14,device='cuda');total=count.clone();square=count.clone()
    for b in captures:
        mask=b['observed'].float();x=b['features']
        count+=mask.sum(0);total+=(mask*x).sum(0);square+=(mask*x.square()).sum(0)
    mean=total/count.clamp_min(1);std=(square/count.clamp_min(1)-mean.square()).clamp_min(1e-6).sqrt()
    mean[6:]=0;std[6:]=1
    benign_total=torch.zeros(14,device='cuda');benign_count=benign_total.clone()
    for b in captures:
        x=((b['features']-mean)/std).clamp(-5,5)
        mask=b['observed'] & (b['label']<.5)[:,None]
        benign_total+=(x*mask).sum(0);benign_count+=mask.sum(0)
    centroid=benign_total/benign_count.clamp_min(1)
    return dict(mean=mean,std=std,benign_centroid=centroid,observed_counts=count)


def windows(capture,scenario,preprocessing,limit=None):
    n=len(capture['label'])
    if limit is not None:n=min(n,limit*WINDOW)
    size=(n+WINDOW-1)//WINDOW;pad=size*WINDOW-n
    def padded(value,fill=0):
        value=value[:n]
        if pad:
            value=torch.cat((value,torch.full((pad,)+value.shape[1:],fill,dtype=value.dtype,device='cuda')))
        return value.reshape((size,WINDOW)+value.shape[1:])
    observed=padded(capture['observed'])
    scaled=((capture['features'][:n]-preprocessing['mean'])/preprocessing['std']).clamp(-5,5)
    features=padded(torch.where(capture['observed'][:n],scaled,torch.zeros_like(scaled)))
    valid=padded(torch.ones(n,device='cuda',dtype=torch.bool))
    src,dst=padded(capture['src']),padded(capture['dst'])
    # Actual observed endpoint sharing; edges imply no logical label constraint.
    adj=(((src[:,:,None]==src[:,None,:]) & (src[:,:,None]!=0)) |
         ((dst[:,:,None]==dst[:,None,:]) & (dst[:,:,None]!=0)) |
         ((src[:,:,None]==dst[:,None,:]) & (src[:,:,None]!=0)) |
         ((dst[:,:,None]==src[:,None,:]) & (dst[:,:,None]!=0)))
    adj=adj & valid[:,:,None] & valid[:,None,:]
    return dict(x=torch.cat((features,observed.float()),-1),adj=adj.float(),valid=valid,
        y=padded(capture['label']),row_id=padded(capture['row_id'],-1),timestamp=padded(capture['timestamp']),
        scenario=torch.full((size,),scenario,device='cuda',dtype=torch.long),
        window=torch.arange(size,device='cuda'),block=torch.arange(size,device='cuda')//64)


def corrupt(batch,condition,centroid,seed):
    if condition not in CONDITIONS:raise ValueError(condition)
    if condition=='clean':return batch
    generator=torch.Generator(device='cuda').manual_seed(seed)
    x=batch['x'].clone();observed=x[:,:,14:].bool()
    rate=.2 if condition.endswith('20') else (.5 if condition.endswith('50') else .4)
    if condition.startswith('missing'):
        observed=observed & (torch.rand(observed.shape,generator=generator,device='cuda')>=rate)
        x[:,:,:14]=torch.where(observed,x[:,:,:14],torch.zeros_like(x[:,:,:14]))
        x[:,:,14:]=observed.float()
    else:
        selected=(torch.rand(x[:,:,:8].shape,generator=generator,device='cuda')<rate)&observed[:,:,:8]
        x[:,:,:8]=torch.where(selected,centroid[:8],x[:,:,:8])
    return dict(batch,x=x)


def condition_batch(batch,condition,preprocessing):
    scenario=int(batch['scenario'][0])
    return corrupt(batch,condition,preprocessing['benign_centroid'],83027+scenario*100+CONDITIONS.index(condition))


def concatenate(batches):
    return {key:torch.cat([b[key] for b in batches],0) for key in batches[0]}


def subset(batch,indices):
    n=len(batch['x'])
    return {key:value[indices] if torch.is_tensor(value) and value.ndim and value.shape[0]==n else value for key,value in batch.items()}


def training_mixture(clean,preprocessing):
    data=dict(clean,x=clean['x'].clone())
    ids=torch.arange(len(clean['x']),device='cuda')
    for j,condition in enumerate(['clean','missing20','spoof20']):
        variant=corrupt(clean,condition,preprocessing['benign_centroid'],82027+j)
        data['x'][ids%3==j]=variant['x'][ids%3==j]
    return data


def input_hash(batch):
    digest=hashlib.sha256()
    for key in ['x','adj','valid','y','row_id','window','block']:
        digest.update(batch[key].cpu().numpy().tobytes())
    return digest.hexdigest()
