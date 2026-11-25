"""Tie-aware CUDA flow-classification metrics and a conventional reference."""
import torch
from torch import nn
from real_models import observable_input


def ranked(y,q):
    order=q.argsort(descending=True,stable=True);p=q[order];truth=y[order].double()
    ends=torch.cat(((p[1:]!=p[:-1]).nonzero().flatten(),torch.tensor([len(p)-1],device='cuda')))
    tp=truth.cumsum(0)[ends];counts=(ends+1).double()
    return tp,counts,p[ends]


def metrics(y,q,threshold=.5):
    y=y.double();q=q.float();tp,counts,_=ranked(y,q);positive=y.sum();negative=len(y)-positive
    precision=tp/counts
    ap=((tp-torch.cat((torch.zeros(1,device='cuda'),tp[:-1])))*precision).sum()/positive.clamp_min(1)
    tpr=torch.cat((torch.zeros(1,device='cuda'),tp/positive.clamp_min(1)))
    fpr_curve=torch.cat((torch.zeros(1,device='cuda'),(counts-tp)/negative.clamp_min(1)))
    auc=torch.trapezoid(tpr,fpr_curve)
    decision=q>=threshold;actual=y>.5
    hit=(decision&actual).sum().double();false=(decision&~actual).sum().double();miss=(~decision&actual).sum().double()
    recall=hit/positive.clamp_min(1);fpr=false/negative.clamp_min(1)
    return dict(ap=float(ap) if positive>0 else float('nan'),auc=float(auc) if min(positive,negative)>0 else float('nan'),
        brier=float((q.double()-y).square().mean()),balanced_accuracy=float((recall+1-fpr)/2),
        f1=float(2*hit/(2*hit+false+miss).clamp_min(1)),recall=float(recall),fpr=float(fpr))


def select_threshold(y,q):
    tp,counts,thresholds=ranked(y,q)
    f1=2*tp/(counts+y.sum()).clamp_min(1)
    # Scores are descending, so first argmax is the highest threshold in a tie.
    return float(thresholds[f1.argmax()])


class GaussianNB(nn.Module):
    """Conditional-independent Gaussian evidence update; missing values omitted.

    Every normalized measurement, including binary indicators, uses a Gaussian
    approximation with variance floor 1e-3. The mask is not a class feature.
    """
    def __init__(self):
        super().__init__();self.key='NB'
        self.register_buffer('mean',torch.zeros(2,14));self.register_buffer('variance',torch.ones(2,14))
        self.register_buffer('prior',torch.full((2,),.5))

    @torch.no_grad()
    def fit(self,batch,window_weight):
        x,valid=observable_input(batch);features=x[:,:,:14];observed=x[:,:,14:]
        weight=window_weight[:,None]/valid.sum(-1,keepdim=True).clamp_min(1)
        class_totals=[]
        for cls in range(2):
            rows=(batch['y']==cls)&valid;w=weight*rows
            total=(w[...,None]*observed).sum((0,1)).clamp_min(1e-8)
            mean=(w[...,None]*observed*features).sum((0,1))/total
            variance=(w[...,None]*observed*(features-mean).square()).sum((0,1))/total
            self.mean[cls]=mean;self.variance[cls]=variance.clamp_min(1e-3);class_totals.append(w.sum())
        totals=torch.stack(class_totals);self.prior.copy_((totals/totals.sum()).clamp_min(1e-8))
        return self

    def probabilities(self,batch):
        x,active=observable_input(batch);features=x[:,:,:14];observed=x[:,:,14:]
        likelihood=[]
        for cls in range(2):
            terms=-.5*(self.variance[cls].log()+(features-self.mean[cls]).square()/self.variance[cls])
            likelihood.append((terms*observed).sum(-1)+self.prior[cls].log())
        return torch.where(active,(likelihood[1]-likelihood[0]).sigmoid(),0.)
