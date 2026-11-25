"""CUDA core for a controlled cyber-prerequisite regeneration study.

This is a synthetic, static decision snapshot with noisy prior-time observations.
Facts are not hosts. Candidates are fixed conjunctions of required facts, with
deterministic terminal utility, not a full cyber range or sequential planner.
Topology clusters are independently seeded across splits. Test truth and attack
masks are only generator/evaluator outputs; Model.forward accepts x and adj only.

Feature channels: 0:3 current report values; 3:6 observed indicators;
6:9 report ages; 9 prior-time noisy fused observation; 10 provisional current
belief; 11 normalized graph degree; 12 known candidate criticality;
13 observed current/prior-time disagreement.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F

FEATURES = 14
_CODEBOOKS = {}

def _cuda(device):
    device = torch.device(device)
    if device.type != "cuda":
        raise RuntimeError("Research tensor computation requires CUDA; CPU fallback is disabled.")
    return device

def _generator(seed, device):
    return torch.Generator(device=device).manual_seed(int(seed))

def _codebook(facts, dim, device):
    device = _cuda(device)
    key = (facts, dim, str(device))
    if key not in _CODEBOOKS:
        # Same codebook prefix across sizes: allocate from the maximum fact domain.
        gen = _generator(90217, device)
        full = torch.randint(0, 2, (512, dim), generator=gen, device=device).float().mul_(2).sub_(1)
        _CODEBOOKS[key] = full[:facts].contiguous()
    return _CODEBOOKS[key]

def generate(n, seed, device, facts=64, family="train", corruption=.2,
             attack="targeted", missing=.2, budget=2048):
    """Generate paired worlds independent of corruption/visibility settings.

    Each topology is shared by up to 16 independent hidden-state snapshots,
    enabling cluster bootstrap. Current remote reports use a 12-byte record
    (uint16 fact, uint16 source, float32 value, float32 age) and 16-byte header.
    Budget applies to each of two remote observers; observer zero is local.
    bytes contains actual payload length of that explicit serialization schema.
    Transport is sparse symbolic report exchange common to all model methods;
    HDC is internal memory, not a claim of an HDC wire codec comparison.
    """
    device = _cuda(device)
    if not 8 <= facts <= 512 or n < 1:
        raise ValueError("Require 8<=facts<=512 and n>=1")
    if not 0 <= corruption <= 1 or not 0 <= missing <= 1 or budget < 0:
        raise ValueError("Invalid corruption, missingness or budget")
    g = _generator(seed, device)
    clusters = (n + 15)//16
    roots = max(2, facts//(4 if family in ("ood", "extrapolation", "unseen") else 8))
    parents = torch.randint(0, roots, (clusters, facts), generator=g, device=device)
    parents[:, :roots] = torch.arange(roots, device=device)
    parent = parents.repeat_interleave(16,0)[:n]
    idx = torch.arange(facts, device=device)
    adj = torch.zeros(n, facts, facts, device=device)
    adj.scatter_(2, parent.unsqueeze(-1), 1)
    adj = torch.maximum(adj, adj.transpose(1,2))
    adj[:, idx, idx] = 1
    # Independent latent readiness and genuine transitions; projection enforces
    # supplied child => parent logical factors in both physical states.
    latent_previous = (torch.rand(n,facts,generator=g,device=device) < .83).float()
    latent_previous[:,:roots] = (torch.rand(n,roots,generator=g,device=device)<.9).float()
    latent_current = torch.where(torch.rand(n,facts,generator=g,device=device)<.06,
                                1-latent_previous,latent_previous)
    previous_y = latent_previous * latent_previous.gather(1,parent)
    y = latent_current * latent_current.gather(1,parent)
    # Candidate generation uses topology/action catalogue, never hidden state.
    scores = torch.rand(n,32,facts,generator=g,device=device)
    chosen = scores.topk(min(5,facts),dim=-1).indices
    length = torch.randint(2,6,(n,32),generator=g,device=device)
    req = torch.zeros(n,32,facts,device=device)
    req.scatter_(2,chosen,(torch.arange(5,device=device)[None,None,:]<length[:,:,None]).float())
    values = .7 + .5*torch.rand(n,32,generator=g,device=device)
    costs = .02 + .18*torch.rand(n,32,generator=g,device=device)
    criticality = (req*values.unsqueeze(-1)).sum(1)
    criticality = criticality/criticality.amax(1,keepdim=True).clamp_min(1e-6)
    # All observation draws precede any condition-dependent branch.
    natural_flip = torch.rand(n,facts,3,generator=g,device=device)<.04
    reports = torch.where(natural_flip,1-y.unsqueeze(-1),y.unsqueeze(-1)).clone()
    visibility = torch.rand(n,facts,3,generator=g,device=device)
    ages = torch.rand(n,facts,3,generator=g,device=device)
    history_flip = torch.rand(n,facts,generator=g,device=device)<.10
    history = torch.where(history_flip,1-previous_y,previous_y)*.84+.08
    rank_random = torch.rand(n,facts,3,generator=g,device=device)
    source_pick = torch.randint(0,3,(n,1,1),generator=g,device=device)
    coherent_group = torch.randint(0,roots,(n,1),generator=g,device=device)
    # Different conditions preserve world, topology and raw sensor draws.
    observed = visibility >= missing
    if attack == "random":
        attack_score = rank_random
    elif attack == "targeted":
        source_preference = (torch.arange(3,device=device)[None,None,:]==source_pick).float()
        attack_score = criticality[:,:,None] + .3*source_preference + .02*rank_random
    elif attack in ("coherent","coherent_correlated"):
        same_group = (parent==coherent_group).float()
        attack_score = same_group[:,:,None] + .2*criticality[:,:,None]+.02*rank_random
    elif attack in ("none","benign"):
        attack_score = rank_random
        corruption = 0
    else:
        raise ValueError("Unknown attack: "+attack)
    eligible = observed.flatten(1)
    attack_score = attack_score.flatten(1).masked_fill(~eligible,-float("inf"))
    ranks = attack_score.argsort(dim=1,descending=True,stable=True).argsort(dim=1)
    count = (eligible.sum(1).float()*corruption).floor().long()
    attacked = (ranks<count[:,None]) & eligible
    attacked = attacked.reshape(n,facts,3)
    before_attack = reports.clone()
    if attack in ("coherent","coherent_correlated"):
        # All-ready story is locally rule consistent where the whole motif is hit.
        reports = torch.where(attacked,torch.ones_like(reports),reports)
    else:
        reports = torch.where(attacked,1-reports,reports)
    # Common observable-priority remote serializer: freshest reports first.
    raw_observed = observed.clone()
    counts = torch.zeros(n,3,device=device,dtype=torch.long)
    for source in (1,2):
        cap = max(0,min(facts,(int(budget)-16)//12))
        priority = (-ages[:,:,source]).masked_fill(~observed[:,:,source],-float("inf"))
        order = priority.argsort(dim=1,descending=True,stable=True).argsort(dim=1)
        observed[:,:,source] &= order<cap
        counts[:,source] = observed[:,:,source].sum(1)
    byte_counts = ((counts[:,1:]*12)+16*(counts[:,1:]>0)).sum(1)
    delivered = observed.sum((1,2)).clamp_min(1)
    changed_fraction = ((reports!=before_attack)&observed).sum((1,2))/delivered
    intervention_fraction = (attacked&observed).sum((1,2))/delivered
    false_report_fraction = ((reports!=y.unsqueeze(-1))&observed).sum((1,2))/delivered
    seen = observed.float()
    obs_reports = torch.where(observed,reports,torch.full_like(reports,.5))
    obs_ages = torch.where(observed,ages,torch.ones_like(ages))
    weight = seen*(1.5-.5*obs_ages)
    # Weakly informative prior-time evidence; genuine changes remain possible.
    provisional = ((weight*obs_reports).sum(-1)+.45*history)/(weight.sum(-1)+.45)
    provisional = provisional.clamp(.02,.98)
    degree = adj.sum(-1)
    degree = degree/degree.amax(1,keepdim=True)
    x = torch.cat([obs_reports,seen,obs_ages,history.unsqueeze(-1),
                   provisional.unsqueeze(-1),degree.unsqueeze(-1),
                   criticality.unsqueeze(-1),(provisional-history).abs().unsqueeze(-1)],-1)
    return dict(x=x, prior=provisional, y=y, previous_y=previous_y, adj=adj,
                req=req, values=values, costs=costs,
                group=torch.arange(n,device=device)//16+int(seed)*1000000,
                parent=parent, reports=obs_reports, observed=seen,
                mask=attacked, raw_observed=raw_observed, bytes=byte_counts,
                transitions=(y!=previous_y), criticality=criticality,
                changed_fraction=changed_fraction, intervention_fraction=intervention_fraction,
                false_report_fraction=false_report_fraction)

def utilities(b, batch):
    """Exact product semantics on 0/1 truth, relaxed conjunction on probabilities."""
    success = ((1-batch["req"])+batch["req"]*b.unsqueeze(1)).prod(-1)
    return batch["values"]*success-batch["costs"]

def feasible(b, batch):
    return ((1-batch["req"])+batch["req"]*b.unsqueeze(1)).prod(-1)

def regret(b,batch,tau=.15):
    truth = utilities(batch["y"],batch)
    gap = truth.amax(-1,keepdim=True)-truth
    return (utilities(b,batch).div(tau).softmax(-1)*gap).sum(-1)

@torch.no_grad()
def teacher_delta(batch,tau=.15,chunk=256):
    """Equation (3), analytical single-coordinate substitution, no model gradients."""
    n,facts = batch["prior"].shape
    output=[]
    for start in range(0,n,chunk):
        stop=min(n,start+chunk)
        b=batch["prior"][start:stop].clamp(1e-6,1-1e-6)
        y=batch["y"][start:stop]
        req=batch["req"][start:stop]
        value=batch["values"][start:stop]
        cost=batch["costs"][start:stop]
        success=((1-req)+req*b.unsqueeze(1)).prod(-1)
        u=value*success-cost
        true_u=value*((1-req)+req*y.unsqueeze(1)).prod(-1)-cost
        gaps=true_u.amax(-1,keepdim=True)-true_u
        base=(u.div(tau).softmax(-1)*gaps).sum(-1)
        # [batch,fact,candidate]: ratio affects only candidates requiring f.
        ratio=(y/b).unsqueeze(-1)
        cf_success=success.unsqueeze(1)*(1-req.transpose(1,2)+req.transpose(1,2)*ratio)
        cf_u=value.unsqueeze(1)*cf_success-cost.unsqueeze(1)
        cf_regret=(cf_u.div(tau).softmax(-1)*gaps.unsqueeze(1)).sum(-1)
        output.append((base.unsqueeze(-1)-cf_regret).clamp_min(0))
    return torch.cat(output)

def topk_gate(scores,k,positive_only=False):
    k=max(0,min(int(k),scores.shape[-1]))
    gate=torch.zeros_like(scores)
    if k:
        # Stable sort ensures deterministic ties without perturbing real scores.
        indices=scores.argsort(dim=-1,descending=True,stable=True)[...,:k]
        values=torch.ones_like(indices,dtype=scores.dtype)
        if positive_only:
            values=(scores.gather(-1,indices)>0).to(scores.dtype)
        gate.scatter_(-1,indices,values)
    return gate

def apply_gate(prior,reconstructed,gate):
    return (1-gate)*prior+gate*reconstructed

def semantic_loss(b,batch):
    parents=b.gather(1,batch["parent"])
    mask=(batch["parent"]!=torch.arange(b.shape[1],device=b.device)[None,:])
    satisfaction=1-b*(1-parents)
    return (-satisfaction.clamp_min(1e-6).log()*mask).sum()/mask.sum().clamp_min(1)

def symbolic_update(batch):
    b=batch["prior"]
    return torch.minimum(b,b.gather(1,batch["parent"]))

def without_provenance(x):
    """Remove source distinctions and ages while retaining observable means."""
    result=x.clone()
    counts=x[:,:,3:6].sum(-1,keepdim=True)
    mean=(x[:,:,:3]*x[:,:,3:6]).sum(-1,keepdim=True)/counts.clamp_min(1)
    mean=torch.where(counts>0,mean,torch.full_like(mean,.5))
    result[:,:,:3]=mean.expand_as(result[:,:,:3])
    result[:,:,3:6]=(counts/3).expand_as(result[:,:,3:6])
    result[:,:,6:9]=0
    return result

def hdc_features(x,dim=4096):
    code=_codebook(x.shape[1],dim,x.device)
    signed=(2*x[:,:,:3]-1)*x[:,:,3:6]
    with torch.autocast(device_type="cuda", enabled=False):
        memory=torch.einsum("nfs,fd->nsd",signed.float(),code)
        decoded=torch.einsum("nsd,fd->nfs",memory,code)/dim
    return decoded.clamp(-2,2)

@torch.no_grad()
def hdc_cleanup(batch,dim=4096):
    retrieval=(hdc_features(batch["x"],dim)+1)*.5
    weights=batch["observed"]*(1.5-.5*batch["x"][:,:,6:9])
    return ((retrieval*weights).sum(-1)+.45*batch["x"][:,:,9]).div(weights.sum(-1)+.45).clamp(.02,.98)

class Model(nn.Module):
    """Shared graph model; HDC supplies three fixed associative retrieval features."""
    def __init__(self,kind="hdc",hidden=128,dim=4096,max_facts=512):
        super().__init__()
        if kind not in ("hdc","dense"):
            raise ValueError(kind)
        self.kind,self.dim,self.max_facts=kind,dim,max_facts
        self.mask_token=nn.Parameter(torch.zeros(FEATURES))
        self.input=nn.Linear(FEATURES+(3 if kind=="hdc" else 0),hidden)
        self.self_layers=nn.ModuleList([nn.Linear(hidden,hidden) for _ in range(3)])
        self.neighbor_layers=nn.ModuleList([nn.Linear(hidden,hidden,bias=False) for _ in range(3)])
        self.norms=nn.ModuleList([nn.LayerNorm(hidden) for _ in range(3)])
        self.belief=nn.Linear(hidden,1)
        self.gate=nn.Linear(hidden,1)
        self.feature_decoder=nn.Sequential(nn.Linear(hidden,hidden),nn.GELU(),nn.Linear(hidden,FEATURES))
        nn.init.zeros_(self.belief.weight)
        nn.init.zeros_(self.belief.bias)

    def encode(self,x,adj):
        _cuda(x.device)
        if x.shape[1]>self.max_facts:
            raise ValueError("Fact capacity exceeded")
        inp=torch.cat([x,hdc_features(x,self.dim).to(x.dtype)],-1) if self.kind=="hdc" else x
        h=F.gelu(self.input(inp))
        degree=adj.sum(-1,keepdim=True).clamp_min(1)
        normalized=adj/degree
        for sl,nl,norm in zip(self.self_layers,self.neighbor_layers,self.norms):
            neighbor=torch.bmm(normalized.to(h.dtype),h)
            h=norm(h+F.gelu(sl(h)+nl(neighbor)))
        return h

    def forward(self,x,adj):
        h=self.encode(x,adj)
        residual=torch.logit(x[:,:,10].float().clamp(.02,.98))
        return self.belief(h).squeeze(-1).float()+residual,self.gate(h).squeeze(-1).float()

    def feature_reconstruction(self,x,adj,mask):
        masked=torch.where(mask.unsqueeze(-1),self.mask_token[None,None,:],x)
        h=self.encode(masked,adj)
        # GraphMAE-style latent remasking before decoding and neighbor propagation.
        h=torch.where(mask.unsqueeze(-1),torch.zeros_like(h),h)
        h=h+torch.bmm((adj/adj.sum(-1,keepdim=True).clamp_min(1)).to(h.dtype),h)
        return self.feature_decoder(h).float()

def masked_feature_loss(pred,target,mask):
    """Scaled cosine error on held-out node features, exponent two."""
    similarity=F.cosine_similarity(pred.float(),target.float(),dim=-1,eps=1e-8)
    values=(1-similarity).square()
    return (values*mask).sum()/mask.sum().clamp_min(1)

def slice_batch(batch,index):
    return {name:value[index] for name,value in batch.items()}


def serialize_reports(batch,index=0):
    """Materialize exactly the two remote payloads whose lengths bytes reports.

    Serialization is host I/O; GPU tensor operations remain on CUDA. The source
    observer identity and record count are carried in the explicit header.
    There is no payload when a source transmits zero records.
    """
    import struct
    values=batch["reports"][index].detach().cpu()
    observed=batch["observed"][index].detach().cpu()
    ages=batch["x"][index,:,6:9].detach().cpu()
    payloads={}
    for source in (1,2):
        indices=observed[:,source].nonzero().flatten().tolist()
        if not indices:
            payloads[source]=b""
            continue
        header=struct.pack("<4sIII",b"HCR1",source,len(indices),0)
        records=b"".join(struct.pack("<HHff",fact,source,float(values[fact,source]),
                                    float(ages[fact,source])) for fact in indices)
        payloads[source]=header+records
    return payloads
