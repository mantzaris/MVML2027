"""Measured cross-representation intervention evidence and subset-only prediction."""
from __future__ import annotations
import json
from pathlib import Path
import torch
from uac.models import FeatureDetector, SpectralDetector, BackgroundFlow, matched_sham, interval_mask
from uac.dsp import extract_features, PRIMARY_GROUPS
from uac.runtime import sha256_file

PRIMARY_SLICES={'fourier':(0,32),'stft':(32,96),'wavelet':(96,152)}
SUBSET_NAMES={frozenset(['fourier']):'fourier',frozenset(['stft']):'stft',frozenset(['wavelet']):'wavelet',frozenset(['fourier','stft']):'fourier_stft',frozenset(['fourier','wavelet']):'fourier_wavelet',frozenset(['stft','wavelet']):'stft_wavelet',frozenset(PRIMARY_GROUPS):'main'}


def load_real_cache(split,device='cuda',verify=True):
    path=Path('data/features/real')/split/'index.json'
    index=json.loads(path.read_text())
    if index['status']!='complete':raise RuntimeError('Incomplete real feature cache: '+split)
    data=[]
    for shard in index['shards']:
        if verify and sha256_file(shard['payload'])!=shard['payload_sha256']:raise RuntimeError('Feature checksum mismatch')
        data.append(torch.load(shard['payload'],map_location='cpu',weights_only=False))
    return {'audio':torch.cat([d['audio'] for d in data]).to(device),'features':torch.cat([d['features'] for d in data]).to(device),
            'labels':torch.cat([d['labels'] for d in data]).to(device),'ids':[x for d in data for x in d['ids']],
            'rows':[x for d in data for x in d['rows']],'index':index}


class DetectorBank:
    def __init__(self,seed=17):
        index=json.loads(Path('manifests/detector_index.json').read_text())
        records=index['records']
        for record in records:
            if int(record['seed'])==seed and sha256_file(record['checkpoint'])!=record['checkpoint_sha256']:
                raise RuntimeError('Detector checkpoint checksum mismatch')
        self.records={r['method']:r for r in records if int(r['seed'])==seed}
        self.models={}
        for name,r in self.records.items():
            if r['kind']=='foundation':continue
            model=SpectralDetector() if r['kind']=='cnn' else FeatureDetector(r['input_dim'])
            payload=torch.load(r['checkpoint'],map_location='cuda',weights_only=False)
            model=model.cuda();model.load_state_dict(payload['model']);model.eval();self.models[name]=model
    @torch.inference_mode()
    def logits(self,name,features,audio=None):
        record=self.records[name]
        if record['kind']=='cnn':return self.models[name](audio)
        return self.models[name](features[:,record['feature_indices']])
    @torch.inference_mode()
    def subset_logit(self,features,views):
        return self.logits(SUBSET_NAMES[frozenset(views)],features)
    @torch.inference_mode()
    def view_logits(self,features):
        return torch.stack([self.logits(v,features).clamp(-8,8) for v in PRIMARY_GROUPS],-1)


def load_flow():
    payload=torch.load('checkpoints/background_flow/final.pt',map_location='cuda',weights_only=False)
    flow=BackgroundFlow(payload['settings']['base']).cuda();flow.load_state_dict(payload['model']);flow.eval();return flow


def band_mismatch(edit,nuisance):
    frequencies=torch.fft.rfftfreq(edit.shape[-1],d=1/16000,device='cuda')
    edges=[0,50,100,200,500,1000,2000,4000,8001]
    target_power=torch.fft.rfft(edit).abs().square();sham_power=torch.fft.rfft(nuisance).abs().square()
    target=torch.stack([target_power[:,(frequencies>=a)&(frequencies<b)].sum(-1) for a,b in zip(edges[:-1],edges[1:])],-1)
    sham=torch.stack([sham_power[:,(frequencies>=a)&(frequencies<b)].sum(-1) for a,b in zip(edges[:-1],edges[1:])],-1)
    target=target/target.sum(-1,keepdim=True).clamp_min(1e-20);sham=sham/sham.sum(-1,keepdim=True).clamp_min(1e-20)
    return (target-sham).abs().sum(-1)


@torch.inference_mode()
def generate_intervention_features(flow,audio,*,seed,k=4,steps=8,probe="central"):
    """Common generated samples for all matched policies; no labels consumed."""
    if audio.device.type!='cuda':raise RuntimeError('Interventions require CUDA')
    n=audio.shape[-1]
    mask=torch.ones_like(audio) if probe=="full" else interval_mask(len(audio),n,n//4,3*n//4,taper_samples=160)
    replacements=[];shams=[];energy_errors=[];ratios=[];band_errors=[];boundaries=[];outside_errors=[];draw_validity=[]
    generator=torch.Generator(device='cuda').manual_seed(seed)
    for draw in range(k):
        replaced=flow.replace(audio,mask,steps=steps,generator=generator)
        ratio_before=replaced.square().mean(-1).sqrt()/audio.square().mean(-1).sqrt().clamp_min(1e-8)
        finite_draw=torch.isfinite(replaced).all(-1)&torch.isfinite(ratio_before)&(ratio_before>.01)&(ratio_before<10)&(replaced.abs().amax(-1)<1000)
        draw_validity.append(finite_draw)
        # Preserve the failure flag; only substitute safe numeric input to the
        # feature extractor so an invalid draw returns the base score perrecord.
        replaced=torch.where(finite_draw[:,None],replaced,audio)
        noise=torch.randn(audio.shape,device='cuda',generator=generator)
        sham,diag=matched_sham(audio,replaced,mask,noise)
        replacements.append(extract_features(replaced,groups=PRIMARY_GROUPS)['features'])
        shams.append(extract_features(sham,groups=PRIMARY_GROUPS)['features'])
        energy_errors.append(diag['relative_energy_mismatch'])
        band_errors.append(band_mismatch(replaced-audio,sham-audio))
        change=replaced-audio
        rms=audio.square().mean(-1).sqrt().clamp_min(1e-8)
        exterior=(mask==0).float()
        outside=((change.square()*exterior).sum(-1)/exterior.sum(-1).clamp_min(1)).sqrt()/rms
        derivative=change[:,1:]-change[:,:-1]
        edge=((mask[:,1:]>0)&(mask[:,1:]<1)).float()
        boundary=((derivative.square()*edge).sum(-1)/edge.sum(-1).clamp_min(1)).sqrt()/rms
        outside_errors.append(outside);boundaries.append(boundary)
        ratios.append(replaced.square().mean(-1).sqrt()/audio.square().mean(-1).sqrt().clamp_min(1e-8))
    replaced_features=torch.stack(replacements,1);sham_features=torch.stack(shams,1)
    energy_error=torch.stack(energy_errors,1);rms_ratio=torch.stack(ratios,1)
    valid=torch.isfinite(replaced_features).all((-1,-2))&torch.isfinite(sham_features).all((-1,-2))
    band_error=torch.stack(band_errors,1);outside=torch.stack(outside_errors,1);boundary=torch.stack(boundaries,1)
    valid=valid&torch.stack(draw_validity,1).all(1)&(energy_error.amax(1)<.05)&(band_error.amax(1)<.2)&(outside.amax(1)<1e-5)&(rms_ratio.amin(1)>.01)&(rms_ratio.amax(1)<10)
    return {'replacement_features':replaced_features,'sham_features':sham_features,'energy_mismatch':energy_error,'rms_ratio':rms_ratio,'valid':valid,'band_error':band_error,'outside_error':outside,'boundary_penalty':boundary,'draw_validity':torch.stack(draw_validity,1)}


@torch.inference_mode()
def intervention_components(bank,cache,background_statistics):
    n,k,d=cache['replacement_features'].shape
    # Counterfactual contrast Eq6: (Lx-Lminus)-(Lx-Lsham)=Lsham-Lminus.
    minus=bank.view_logits(cache['replacement_features'].reshape(n*k,d)).reshape(n,k,3)
    sham=bank.view_logits(cache['sham_features'].reshape(n*k,d)).reshape(n,k,3)
    delta=sham-minus
    # The exterior may contain a true source; never compare the full replacement
    # waveform with a pure-background distribution. This penalty measures only
    # unintended exterior changes and normalized transition disturbance.
    penalty=(cache['outside_error']+cache['boundary_penalty']).mean(1)
    return {'delta':delta,'penalty':penalty,'valid':cache['valid'],'minus_logits':minus,'sham_logits':sham}


def verification_score(components,views,dispersion=.5,invalid_penalty=.1):
    positions=[PRIMARY_GROUPS.index(v) for v in views]
    evidence=components['delta'][...,positions].flatten(1)
    median=evidence.median(1).values
    mad=(evidence-median[:,None]).abs().median(1).values
    score=median-dispersion*mad-invalid_penalty*components['penalty']
    score=torch.where(components['valid'],score,torch.zeros_like(score))
    return score
