"""CUDA models for the direct-candidate and inexpensive reconstruction study.

SCR composes established set aggregation and mixture-of-experts ideas with a
canonical prerequisite representation. It predicts candidate feasibility;
it does not reconstruct fact beliefs or claim a coherent joint posterior.
V1/V2 implementations are intentionally unchanged.
"""
import torch
from torch import nn
import torch.nn.functional as F

try:
    from .regen import utilities
    from .regen_v2 import (require_cuda, root_mask, independent_probability,
                           loss_terms as belief_loss_terms)
except ImportError:
    from regen import utilities
    from regen_v2 import (require_cuda, root_mask, independent_probability,
                          loss_terms as belief_loss_terms)

KEYS = ("SCR", "CAND_CTX", "CAND_LOCAL", "LOGIT", "DAE")
CANDIDATE_KEYS = frozenset(("SCR", "CAND_CTX", "CAND_LOCAL"))
EPS = 1e-6


def _population_std(x, dim):
    """Population standard deviation with finite derivative at zero variance."""
    return (x.float().var(dim=dim, unbiased=False) + EPS).sqrt()


def observable_features(x, parent):
    """Shared 56-dimensional features; only received observations and topology.

    Concatenate each fact, its parent, and the global raw-feature mean/std.
    Neither hidden state nor attack-group labels enter this representation.
    The shared transform supports arbitrary fact counts and renumberings.
    """
    require_cuda(x)
    require_cuda(parent)
    x = x.float()
    parent_x = x.gather(1, parent.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
    mean = x.mean(1, keepdim=True).expand_as(x)
    std = _population_std(x, 1).unsqueeze(1).expand_as(x)
    return torch.cat((x, parent_x, mean, std), -1)


def canonical_requirements(requirements, parent):
    """Return required nonroots and unique touched roots for a star forest.

    A required child implies its root. Thus adding its root explicitly does
    not change this canonical set. This is an observable symbolic operation,
    not a use of the latent world or of privileged feasibility labels.
    """
    require_cuda(requirements)
    require_cuda(parent)
    req = (requirements > 0).float()
    children = req * (~root_mask(parent)).unsqueeze(1)
    touched = torch.zeros_like(req).scatter_add(
        2, parent[:, None, :].expand_as(req), req)
    return children, (touched > 0).float()


def _set_moments(h, weights):
    count = weights.sum(-1, keepdim=True)
    mean = torch.bmm(weights, h) / count.clamp_min(1.)
    second = torch.bmm(weights, h.square()) / count.clamp_min(1.)
    std = ((second - mean.square()).clamp_min(0.) + EPS).sqrt()
    std = std * (count > 0)
    return mean, std, count


class CandidateScorer(nn.Module):
    """Canonical prerequisite set scorer with optional shared latent regime.

    Global context contains means/stds of observable learned fact embeddings.
    SCR mixes four feasibility experts using one global gate shared by every
    candidate in a snapshot. Experts have no assigned attack-class semantics.
    CAND_CTX has one expert. CAND_LOCAL zeros the same 64 context coordinates,
    retaining exactly the CAND_CTX parameter count and head dimensions.
    """
    def __init__(self, key="SCR"):
        super().__init__()
        if key not in CANDIDATE_KEYS:
            raise ValueError(f"Unknown candidate model: {key}")
        self.key = key
        self.experts = 4 if key == "SCR" else 1
        self.phi = nn.Sequential(nn.Linear(28, 32), nn.GELU(),
                                 nn.Linear(32, 32), nn.GELU())
        self.head = nn.Sequential(nn.Linear(162, 64), nn.GELU(),
                                  nn.Linear(64, self.experts))
        self.gate = (nn.Sequential(nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 4))
                     if key == "SCR" else None)

    def _encode(self, x, parent):
        require_cuda(x)
        require_cuda(parent)
        parent_x = x.gather(1, parent.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
        h = self.phi(torch.cat((x, parent_x), -1))
        # Pooling, variance and probabilities are FP32 under mixed precision.
        with torch.autocast(device_type="cuda", enabled=False):
            h = h.float()
            context = torch.cat((h.mean(1), _population_std(h, 1)), -1)
            if self.key == "CAND_LOCAL":
                context = torch.zeros_like(context)
        return h, context

    def _weights(self, context):
        if self.gate is None:
            return torch.ones((context.shape[0], 1), device=context.device)
        return self.gate(context).float().softmax(-1)

    def context_weights(self, batch):
        """Observed-set mixture weights; diagnostic only, no group-label input."""
        _, context = self._encode(batch["x"], batch["parent"])
        return self._weights(context)

    def forward(self, x, parent, requirements):
        h, context = self._encode(x, parent)
        with torch.autocast(device_type="cuda", enabled=False):
            children, roots = canonical_requirements(requirements, parent)
            child_mean, child_std, child_count = _set_moments(h, children)
            root_mean, _, root_count = _set_moments(h, roots)
            local = torch.cat((child_mean, child_std, root_mean,
                               child_count.log1p(), root_count.log1p()), -1)
            features = torch.cat((local, context[:, None, :].expand(
                -1, requirements.shape[1], -1)), -1)
        logits = self.head(features).float()
        weights = self._weights(context)
        with torch.autocast(device_type="cuda", enabled=False):
            probabilities = (logits.sigmoid() * weights[:, None, :]).sum(-1)
            # The empty conjunction is true by definition.
            return torch.where(requirements.sum(-1) == 0,
                               torch.ones_like(probabilities), probabilities)


class LogisticRegenerator(nn.Module):
    """Cheap shared logistic fact decoder, without a prior-logit shortcut."""
    def __init__(self):
        super().__init__()
        self.key = "LOGIT"
        self.readout = nn.Linear(56, 1)

    def forward(self, x, parent):
        return self.readout(observable_features(x, parent)).squeeze(-1).float()


class DenoisingAutoencoder(nn.Module):
    """Shared feature autoencoder, followed by a supervised readiness decoder.

    Pretraining reconstructs the original *observed* 56-dimensional feature
    vector from additional zero masking and Gaussian noise. It does not use
    clean state labels. Supervised training subsequently learns latent→truth.
    The feature decoder is retained for explicit parameter accounting.
    """
    def __init__(self):
        super().__init__()
        self.key = "DAE"
        self.encoder = nn.Sequential(nn.Linear(56, 32), nn.GELU(),
                                     nn.Linear(32, 8), nn.GELU())
        self.decoder = nn.Sequential(nn.Linear(8, 32), nn.GELU(), nn.Linear(32, 56))
        self.readout = nn.Linear(8, 1)

    def forward(self, x, parent):
        features = observable_features(x, parent)
        return self.readout(self.encoder(features)).squeeze(-1).float()

    def pretraining_loss(self, batch, mask_rate=.25, noise_std=.05):
        """Return per-snapshot feature-denoising MSE; no hidden labels read."""
        target = observable_features(batch["x"], batch["parent"])
        keep = torch.rand_like(target) >= mask_rate
        corrupted = target * keep + noise_std * torch.randn_like(target)
        reconstructed = self.decoder(self.encoder(corrupted)).float()
        return (reconstructed - target).square().mean((1, 2))


def make_model(key):
    if key in CANDIDATE_KEYS:
        return CandidateScorer(key)
    if key == "LOGIT":
        return LogisticRegenerator()
    if key == "DAE":
        return DenoisingAutoencoder()
    raise ValueError(f"Unknown third-study model: {key}")


def infer(model, batch):
    """Observable-only inference; candidate methods have no fact marginals."""
    if model.key in CANDIDATE_KEYS:
        return None, model(batch["x"], batch["parent"], batch["req"])
    marginal = model(batch["x"], batch["parent"]).sigmoid()
    with torch.autocast(device_type="cuda", enabled=False):
        probability = independent_probability(marginal.float(), batch["req"].float())
    return marginal, probability


def loss_terms(model, marginals, planprob, batch, tau=.1):
    """Per-snapshot supervised objectives; only this path reads truth labels.

    Candidate models use feasibility BCE+Brier+0.25 soft decision regret.
    LOGIT/DAE use the exact V2 fact-based objective, including consistency.
    These are deliberately different learning targets, documented as such.
    """
    if model.key not in CANDIDATE_KEYS:
        return belief_loss_terms(marginals, planprob, batch, tau=tau)
    require_cuda(planprob)
    q = planprob.float().clamp(EPS, 1. - EPS)
    truth = batch["y"].float()
    feasible = (torch.bmm(batch["req"].float(), (1-truth).unsqueeze(-1))
                .squeeze(-1) < .5).float()
    candidate_bce = F.binary_cross_entropy(q, feasible, reduction="none").mean(-1)
    plan_brier = (planprob.float() - feasible).square().mean(-1)
    true_utility = utilities(truth, batch)
    gap = true_utility.max(-1, keepdim=True).values - true_utility
    predicted = batch["values"] * planprob.float() - batch["costs"]
    regret = (torch.softmax(predicted / tau, -1) * gap).sum(-1)
    return dict(candidate_bce=candidate_bce, plan_brier=plan_brier,
                decision_regret=regret,
                total=candidate_bce + plan_brier + .25 * regret)
