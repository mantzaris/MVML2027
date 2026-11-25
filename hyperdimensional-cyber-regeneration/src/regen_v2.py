"""Dependency-aware full regeneration for the second CUDA study.

The star-forest posterior uses one Bernoulli parameter per fact. A root
parameter is its marginal readiness; a child's parameter is readiness
conditional on its root being ready. Candidate probability counts each
required root once. This is exact under this conditional posterior family,
not a claim that the true posterior factorizes after deceptive observations.

Original src/regen.py and its frozen experiment remain unchanged.
"""
import torch
import torch.nn.functional as F

try:
    from .regen import Model, utilities
except ImportError:
    from regen import Model, utilities

VARIANTS = {"JH", "JD", "IH", "ID", "JH_ERM"}
STRUCTURED = {"JH", "JD", "JH_ERM"}


def require_cuda(value):
    if value.device.type != "cuda":
        raise RuntimeError("Research tensors require CUDA; CPU fallback is disabled.")


def root_mask(parent):
    require_cuda(parent)
    return parent == torch.arange(parent.shape[-1], device=parent.device)[None, :]


def marginal_probabilities(factors, parent):
    """Map conditional factors to logically coherent marginal probabilities."""
    require_cuda(factors)
    return torch.where(root_mask(parent), factors,
                       factors * factors.gather(1, parent))


def independent_probability(marginals, requirements):
    require_cuda(marginals)
    return ((1 - requirements) + requirements * marginals.unsqueeze(1)).prod(-1)


def joint_probability(factors, requirements, parent):
    """Conjunction probability for the generator's one-level root-child forest.

    requirements[b,j,f] indicates that candidate j needs fact f. Each child
    contributes its conditional factor, and every touched root contributes
    exactly once, whether or not that root is explicitly in the requirement.
    Empty requirements have probability one. Hard 0/1 factors retain exact
    Boolean conjunction semantics. No hidden state is read.
    """
    require_cuda(factors)
    roots = root_mask(parent)
    incidence = F.one_hot(parent, num_classes=factors.shape[-1]).float()
    # Use FP32 even inside mixed-precision training; binary incidence is exact.
    with torch.autocast(device_type="cuda", enabled=False):
        touched = torch.bmm(requirements.float(), incidence) > 0
        needed = ((requirements > 0) & ~roots.unsqueeze(1)) | touched
        return torch.where(needed, factors.float().unsqueeze(1),
                           torch.ones_like(factors.float().unsqueeze(1))).prod(-1)


class RegeneratorV2(Model):
    """Matched encoders with independent or conditional belief semantics.

    The inherited unused gate head and pretraining decoder are retained for
    identical architecture accounting across variants. The gate is not used
    in second-study inference or supervised loss. All facts can be regenerated.
    """
    def __init__(self, variant="JH", hidden=128, dim=4096, max_facts=512):
        if variant not in VARIANTS:
            raise ValueError(f"Unknown second-study variant: {variant}")
        super().__init__(kind="hdc" if variant in {"JH", "IH", "JH_ERM"} else "dense",
                         hidden=hidden, dim=dim, max_facts=max_facts)
        self.variant = variant
        self.structured = variant in STRUCTURED

    def forward(self, x, adj, parent):
        h = self.encode(x, adj)
        prior = x[:, :, 10].float().clamp(.02, .98)
        if self.structured:
            # A coherent initialization approximates the shared fused prior.
            # If child prior exceeds parent prior, its conditional factor clips
            # at .98 and its marginal is reduced; this is not exact prior copy.
            conditional = (prior / prior.gather(1, parent)).clamp(.02, .98)
            prior = torch.where(root_mask(parent), prior, conditional)
        residual = torch.logit(prior)
        return self.belief(h).squeeze(-1).float() + residual


def infer(model, batch, product_override=False):
    """Observable-only inference returning marginal beliefs and plan scores.

    product_override is an inference ablation: structured marginals are kept,
    but their product replaces the dependency-corrected candidate probability.
    """
    factors = model(batch["x"], batch["adj"], batch["parent"]).float().sigmoid()
    if model.structured:
        marginals = marginal_probabilities(factors, batch["parent"])
        planprob = (independent_probability(marginals, batch["req"])
                    if product_override else
                    joint_probability(factors, batch["req"], batch["parent"]))
    else:
        marginals = factors
        planprob = independent_probability(marginals, batch["req"])
    return marginals, planprob


def predicted_utilities(planprob, batch):
    require_cuda(planprob)
    return batch["values"] * planprob - batch["costs"]


def consistency_penalty(marginals, parent):
    """Squared violation of child marginal <= parent marginal, per example.

    This is a consistency regularizer, not a probability of logical violation.
    It is exactly zero for the conditional model's coherent marginals.
    """
    require_cuda(marginals)
    nonroot = ~root_mask(parent)
    violation = F.relu(marginals - marginals.gather(1, parent)).square()
    return (violation * nonroot).sum(-1) / nonroot.sum(-1).clamp_min(1)


def loss_terms(marginals, planprob, batch, tau=.1):
    """Identical per-example objective for all five second-study variants.

    Hidden labels are read here for training/validation only. No structured
    NLL or perfect-repair oracle teacher is added to any method. Per-example
    outputs allow the runner to apply mean or group-robust aggregation.
    """
    require_cuda(marginals)
    p = marginals.float().clamp(1e-6, 1 - 1e-6)
    truth = batch["y"].float()
    fact_bce = F.binary_cross_entropy(p, truth, reduction="none").mean(-1)
    feasible = (torch.bmm(batch["req"], (1-truth).unsqueeze(-1)).squeeze(-1) < .5).float()
    plan_brier = (planprob.float() - feasible).square().mean(-1)
    true_u = utilities(truth, batch)
    gap = true_u.max(-1, keepdim=True).values - true_u
    decision_regret = (torch.softmax(predicted_utilities(planprob, batch) / tau, -1) * gap).sum(-1)
    semantic = consistency_penalty(marginals.float(), batch["parent"])
    total = fact_bce + plan_brier + .25 * decision_regret + .1 * semantic
    return dict(fact_bce=fact_bce, plan_brier=plan_brier,
                decision_regret=decision_regret, semantic=semantic, total=total)


def per_example_loss(marginals, planprob, batch, tau=.1):
    return loss_terms(marginals, planprob, batch, tau)["total"]
