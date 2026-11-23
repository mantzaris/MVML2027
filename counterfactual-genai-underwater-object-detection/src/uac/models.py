"""Compact CUDA acoustic models and a conditional flow background generator."""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F


def require_cuda(x):
    if x.device.type != 'cuda':
        raise RuntimeError('Research tensor computation requires the pinned Ada CUDA device')


class FeatureDetector(nn.Module):
    def __init__(self, dim, width=128):
        super().__init__()
        self.register_buffer('mean', torch.zeros(dim))
        self.register_buffer('scale', torch.ones(dim))
        self.net = nn.Sequential(nn.Linear(dim,width),nn.LayerNorm(width),nn.GELU(),nn.Dropout(.1),nn.Linear(width,64),nn.GELU(),nn.Linear(64,1))
    def fit_scaling(self,x):
        require_cuda(x)
        with torch.no_grad():
            self.mean.copy_(x.mean(0)); self.scale.copy_(x.std(0).clamp_min(1e-4))
    def forward(self,x):
        require_cuda(x)
        return self.net(((x-self.mean)/self.scale).clamp(-12,12)).squeeze(-1)


class SpectralDetector(nn.Module):
    """GPU log-mel CNN with a temporal frame head and clip maximum-mean pooling."""
    def __init__(self, sample_rate=16000, channels=32):
        super().__init__()
        import torchaudio
        self.frontend=torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,n_fft=1024,win_length=1024,hop_length=256,f_min=20,f_max=7500,n_mels=96,power=2.0)
        blocks=[]; cin=1
        for cout in (channels,channels*2,channels*4):
            blocks.extend([nn.Conv2d(cin,cout,3,padding=1),nn.GroupNorm(8,cout),nn.GELU(),nn.Conv2d(cout,cout,3,padding=1),nn.GroupNorm(8,cout),nn.GELU(),nn.AvgPool2d((2,2))]);cin=cout
        self.encoder=nn.Sequential(*blocks)
        self.temporal=nn.Sequential(nn.Conv1d(cin,64,3,padding=1),nn.GELU(),nn.Dropout(.1),nn.Conv1d(64,1,1))
    def forward(self,x,return_frames=False):
        require_cuda(x)
        power=self.frontend(x.float()).clamp_min(1e-10)
        spec=power.log().clamp(-23,12)
        # Fixed physical log scale; no per-clip amplitude normalization.
        z=self.encoder(((spec+12)/8).unsqueeze(1)).mean(2)
        frame=self.temporal(z).squeeze(1)
        clip=.5*frame.mean(-1)+.5*frame.amax(-1)
        return (clip,frame) if return_frames else clip


class ResidualBlock(nn.Module):
    def __init__(self,cin,cout):
        super().__init__()
        self.net=nn.Sequential(nn.Conv1d(cin,cout,5,padding=2),nn.GroupNorm(8,cout),nn.SiLU(),nn.Conv1d(cout,cout,5,padding=2),nn.GroupNorm(8,cout),nn.SiLU())
        self.skip=nn.Conv1d(cin,cout,1) if cin!=cout else nn.Identity()
    def forward(self,x): return (self.net(x)+self.skip(x))/math.sqrt(2)


class BackgroundFlow(nn.Module):
    """Class-free masked waveform flow matching; known context always clamped."""
    def __init__(self,base=32):
        super().__init__()
        self.enc=nn.ModuleList([ResidualBlock(4,base),ResidualBlock(base,base*2),ResidualBlock(base*2,base*4),ResidualBlock(base*4,base*8)])
        self.mid=ResidualBlock(base*8,base*8)
        self.dec=nn.ModuleList([ResidualBlock(base*16,base*4),ResidualBlock(base*8,base*2),ResidualBlock(base*4,base),ResidualBlock(base*2,base)])
        self.out=nn.Conv1d(base,1,1)
    def forward(self,xt,t,context,mask):
        require_cuda(xt)
        h=torch.stack([xt,context*(1-mask),mask,t[:,None].expand_as(xt)],1)
        skips=[]
        for block in self.enc:
            h=block(h);skips.append(h);h=F.avg_pool1d(h,4)
        h=self.mid(h)
        for block,skip in zip(self.dec,reversed(skips)):
            h=F.interpolate(h,size=skip.shape[-1],mode='linear',align_corners=False)
            h=block(torch.cat([h,skip],1))
        return self.out(h).squeeze(1)
    def loss(self,background,observed,mask,generator=None):
        scale=observed.square().mean(-1,keepdim=True).sqrt().clamp_min(1e-4)
        b=background/scale;u=observed/scale
        t=torch.rand(b.shape[0],device=b.device,generator=generator)
        z=torch.randn(b.shape,device=b.device,generator=generator)
        xt=mask*((1-t[:,None])*z+t[:,None]*b)+(1-mask)*u
        pred=self(xt,t,u,mask)
        loss=((pred-(b-z)).square()*mask).sum(-1)/mask.sum(-1).clamp_min(1)
        return loss.mean()
    @torch.inference_mode()
    def replace(self,observed,mask,steps=8,generator=None):
        require_cuda(observed)
        scale=observed.square().mean(-1,keepdim=True).sqrt().clamp_min(1e-4)
        u=observed/scale
        z=torch.randn(u.shape,device=u.device,generator=generator)
        y=mask*z+(1-mask)*u
        for step in range(steps):
            t=torch.full((u.shape[0],),step/steps,device=u.device)
            y=mask*(y+self(y,t,u,mask)/steps)+(1-mask)*u
        return y*scale


def interval_mask(batch,n,start,end,device='cuda',taper_samples=160):
    """Same-support taper used by target replacement and matched nuisance edit."""
    t=torch.arange(n,device=device)[None,:]
    if not torch.is_tensor(start): start=torch.full((batch,),int(start),device=device)
    if not torch.is_tensor(end): end=torch.full((batch,),int(end),device=device)
    inside=(t>=start[:,None])&(t<end[:,None])
    edge=torch.minimum(t-start[:,None],end[:,None]-1-t).clamp_min(0).float()/max(taper_samples,1)
    return inside.float()*(.5-.5*torch.cos(torch.pi*edge.clamp_max(1)))


def matched_sham(original,replacement,mask,background_noise):
    """Same-support random-phase nuisance, spectrum shaped to actual edit energy."""
    require_cuda(original)
    edit=replacement-original
    spectrum=torch.fft.rfft(background_noise)
    unit_phase=spectrum/spectrum.abs().clamp_min(1e-12)
    nuisance=torch.fft.irfft(unit_phase*torch.fft.rfft(edit).abs(),n=original.shape[-1])*mask
    scale=edit.square().sum(-1,keepdim=True).sqrt()/nuisance.square().sum(-1,keepdim=True).sqrt().clamp_min(1e-8)
    sham=original+nuisance*scale
    delta=sham-original
    mismatch=(delta.square().sum(-1).sqrt()-edit.square().sum(-1).sqrt()).abs()/edit.square().sum(-1).sqrt().clamp_min(1e-8)
    return sham,{'relative_energy_mismatch':mismatch}
