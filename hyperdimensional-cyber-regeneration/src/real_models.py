"""CUDA adaptations for externally captured network-flow measurements.

These models are retrained on generic flow features. They do not interpret
intrusion labels as readiness, create prerequisite implications, or reuse the
synthetic benchmark's channel-10 prior shortcut. Endpoint-sharing adjacency
is an observed relationship, not a logical dependency graph.
"""
import torch
from torch import nn
import torch.nn.functional as F

KEYS = ("LOGIT", "DAE", "DENSE", "GMAE", "HDC", "SCR_FLOW", "CTX_FLOW", "LOCAL_FLOW")
FLOW_FEATURES = 14
INPUT_FEATURES = 28
EPS = 1e-6


def require_cuda(value):
    if value.device.type != "cuda":
        raise RuntimeError("Research tensors require CUDA; CPU fallback is disabled.")


def observable_input(batch):
    """Return sanitized observed features and the real-row mask.

    The first 14 columns hold normalized measurements; the last 14 indicate
    whether those measurements were received. Missing values and padded rows
    cannot affect a prediction, even if their stored placeholders are changed.
    """
    x, valid = batch["x"], batch["valid"]
    require_cuda(x)
    require_cuda(valid)
    if x.ndim != 3 or x.shape[-1] != INPUT_FEATURES or valid.shape != x.shape[:2]:
        raise ValueError("Expected x[N,F,28] and valid[N,F]")
    active = valid > 0
    observed = torch.where(active[..., None], x[..., FLOW_FEATURES:].float(), 0.).clamp(0., 1.)
    values = torch.where((observed > 0) & active[..., None], x[..., :FLOW_FEATURES].float(), 0.)
    return torch.cat((values, observed), -1), active


def normalized_adjacency(adj, active):
    """Mask padded endpoints and normalize observable adjacency including self."""
    require_cuda(adj)
    if adj.shape != (active.shape[0], active.shape[1], active.shape[1]):
        raise ValueError("Expected adj[N,F,F]")
    edges = torch.where(active[:, :, None] & active[:, None, :], adj.float(), 0.).clamp_min(0.)
    identity = torch.eye(active.shape[1], device=adj.device)[None] * active[:, :, None]
    edges = torch.maximum(edges, identity)
    return edges / edges.sum(-1, keepdim=True).clamp_min(1.)


def masked_moments(h, active):
    """FP32 population moments, excluding padding and safe at zero variance."""
    with torch.autocast(device_type="cuda", enabled=False):
        h = torch.where(active[..., None], h.float(), 0.)
        count = active.sum(1, keepdim=True).float()
        mean = h.sum(1) / count.clamp_min(1.)
        variance = (h.square().sum(1) / count.clamp_min(1.) - mean.square()).clamp_min(0.)
        std = (variance + EPS).sqrt() * (count > 0)
        return mean, std


def observed_reconstruction_loss(reconstructed, target, active):
    """Per-window MSE on originally observed measurements, not mask columns."""
    weights = (target[..., FLOW_FEATURES:] > 0).float() * active[..., None]
    error = (reconstructed.float() - target[..., :FLOW_FEATURES].float()).square()
    return (error * weights).sum((1, 2)) / weights.sum((1, 2)).clamp_min(1.)


class LogisticFlow(nn.Module):
    def __init__(self):
        super().__init__()
        self.key = "LOGIT"
        self.readout = nn.Linear(INPUT_FEATURES, 1)

    def forward(self, batch):
        x, _ = observable_input(batch)
        return self.readout(x).squeeze(-1).float()


class DenoisingFlow(nn.Module):
    """Observed-feature DAE with a supervised attack-probability head.

    Extra corruption masks 25% of feature coordinates, clears their indicator,
    and adds Gaussian noise (std .05) to retained observed values only. The
    target is the original input's observed features, never clean oracle data.
    The decoder is retained in parameter counts but unused by inference.
    """
    def __init__(self):
        super().__init__()
        self.key = "DAE"
        self.encoder = nn.Sequential(nn.Linear(INPUT_FEATURES, 32), nn.GELU(),
                                     nn.Linear(32, 8), nn.GELU())
        self.decoder = nn.Sequential(nn.Linear(8, 32), nn.GELU(), nn.Linear(32, FLOW_FEATURES))
        self.readout = nn.Linear(8, 1)

    def forward(self, batch):
        x, _ = observable_input(batch)
        return self.readout(self.encoder(x)).squeeze(-1).float()

    def pretraining_loss(self, batch, mask_rate=.25, noise_std=.05):
        target, active = observable_input(batch)
        observed = target[..., FLOW_FEATURES:]
        keep = (torch.rand_like(observed) >= mask_rate).float() * observed
        corrupted_values = (target[..., :FLOW_FEATURES] + noise_std * torch.randn_like(observed)) * keep
        corrupted = torch.cat((corrupted_values, keep), -1)
        reconstructed = self.decoder(self.encoder(corrupted))
        return observed_reconstruction_loss(reconstructed, target, active)


class GraphFlow(nn.Module):
    """Dense/GraphMAE/HDC controls over supplied observed endpoint adjacency.

    DENSE and GMAE have identical architectures; the runner only pretrains
    GMAE. HDC adds 28 fixed associative retrieval coordinates and receives the
    same masked pretraining as GMAE. No logical parent relation is assumed.
    HDC keys identify chronological window slots, so arbitrary slot reordering
    without transporting keys is not a claimed HDC invariance.
    """
    def __init__(self, key, hidden=128, dim=4096, max_flows=64):
        super().__init__()
        if key not in ("DENSE", "GMAE", "HDC"):
            raise ValueError(key)
        self.key, self.dim, self.max_flows = key, dim, max_flows
        self.mask_token = nn.Parameter(torch.zeros(INPUT_FEATURES))
        self.input = nn.Linear(INPUT_FEATURES * (2 if key == "HDC" else 1), hidden)
        self.self_layers = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(3)])
        self.neighbor_layers = nn.ModuleList([nn.Linear(hidden, hidden, bias=False) for _ in range(3)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(3)])
        self.readout = nn.Linear(hidden, 1)
        self.feature_decoder = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, FLOW_FEATURES))
        # Generated on CUDA from a local seed, not included in trained weights.
        self.register_buffer("role_keys", torch.empty(0), persistent=False)

    def _keys(self, count, device):
        if count > self.max_flows:
            raise ValueError(f"At most {self.max_flows} chronological flow slots are supported")
        if self.role_keys.numel() == 0 or self.role_keys.device != device:
            generator = torch.Generator(device=device).manual_seed(27027)
            self.role_keys = (torch.randint(0, 2, (self.max_flows, self.dim),
                                            device=device, generator=generator).float() * 2 - 1)
        return self.role_keys[:count]

    def associative_features(self, x, active):
        require_cuda(x)
        keys = self._keys(x.shape[1], x.device)
        with torch.autocast(device_type="cuda", enabled=False):
            evidence = torch.where(active[..., None], x.float(), 0.)
            memory = torch.einsum("nfc,fd->ncd", evidence, keys)
            retrieved = torch.einsum("ncd,fd->nfc", memory, keys) / self.dim
            return torch.where(active[..., None], retrieved, 0.)

    def encode(self, x, adj, active):
        require_cuda(x)
        x = torch.where(active[..., None], x, 0.)
        graph = normalized_adjacency(adj, active)
        inp = torch.cat((x, self.associative_features(x, active)), -1) if self.key == "HDC" else x
        h = F.gelu(self.input(inp))
        h = torch.where(active[..., None], h, 0.)
        for local, neighbor, norm in zip(self.self_layers, self.neighbor_layers, self.norms):
            pooled = torch.bmm(graph.to(h.dtype), h)
            h = norm(h + F.gelu(local(h) + neighbor(pooled)))
            h = torch.where(active[..., None], h, 0.)
        return h, graph

    def forward(self, batch):
        x, active = observable_input(batch)
        h, _ = self.encode(x, batch["adj"], active)
        return self.readout(h).squeeze(-1).float()

    def pretraining_loss(self, batch, mask_rate=.3):
        target, active = observable_input(batch)
        selected = (torch.rand(active.shape, device=target.device) < mask_rate) & active
        masked = torch.where(selected[..., None], self.mask_token[None, None, :], target)
        h, graph = self.encode(masked, batch["adj"], active)
        h = torch.where(selected[..., None], torch.zeros_like(h), h)
        h = h + torch.bmm(graph.to(h.dtype), h)
        reconstructed = self.feature_decoder(h)
        # GraphMAE-style loss evaluates only the additionally masked real nodes.
        return observed_reconstruction_loss(reconstructed, target, selected)


class ContextFlow(nn.Module):
    """Direct flow classifier with one shared context gate per fixed window."""
    def __init__(self, key):
        super().__init__()
        if key not in ("SCR_FLOW", "CTX_FLOW", "LOCAL_FLOW"):
            raise ValueError(key)
        self.key = key
        self.experts = 4 if key == "SCR_FLOW" else 1
        self.phi = nn.Sequential(nn.Linear(INPUT_FEATURES, 32), nn.GELU(), nn.Linear(32, 32), nn.GELU())
        self.head = nn.Sequential(nn.Linear(96, 64), nn.GELU(), nn.Linear(64, self.experts))
        self.gate = (nn.Sequential(nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 4))
                     if self.experts == 4 else None)

    def _encode(self, batch):
        x, active = observable_input(batch)
        h = self.phi(x)
        h = torch.where(active[..., None], h, 0.)
        mean, std = masked_moments(h, active)
        context = torch.cat((mean, std), -1)
        if self.key == "LOCAL_FLOW":
            context = torch.zeros_like(context)
        return h, context

    def _weights(self, context):
        if self.gate is None:
            return torch.ones((context.shape[0], 1), device=context.device)
        return self.gate(context).float().softmax(-1)

    def context_weights(self, batch):
        _, context = self._encode(batch)
        return self._weights(context)

    def probabilities(self, batch):
        h, context = self._encode(batch)
        features = torch.cat((h.float(), context[:, None, :].expand(-1, h.shape[1], -1)), -1)
        expert_probability = self.head(features).float().sigmoid()
        return (expert_probability * self._weights(context)[:, None, :]).sum(-1)


def make_model(key):
    if key == "LOGIT":
        return LogisticFlow()
    if key == "DAE":
        return DenoisingFlow()
    if key in ("DENSE", "GMAE", "HDC"):
        return GraphFlow(key)
    if key in ("SCR_FLOW", "CTX_FLOW", "LOCAL_FLOW"):
        return ContextFlow(key)
    raise ValueError(f"Unknown real-flow model: {key}")


def infer(model, batch):
    """Return attack probabilities, with padded positions defined as zero."""
    _, active = observable_input(batch)
    probability = model.probabilities(batch) if isinstance(model, ContextFlow) else model(batch).sigmoid()
    return torch.where(active, probability.float(), 0.)


def loss_terms(probability, batch):
    """Common per-window supervised BCE+Brier; labels read only in this path."""
    require_cuda(probability)
    require_cuda(batch["y"])
    require_cuda(batch["valid"])
    active = batch["valid"] > 0
    if probability.shape != active.shape or batch["y"].shape != active.shape:
        raise ValueError("Expected predictions, labels and valid mask with shape [N,F]")
    target = torch.where(active, batch["y"].float(), 0.)
    weights = active.float()
    denominator = weights.sum(-1).clamp_min(1.)
    p = probability.float().clamp(EPS, 1. - EPS)
    bce = (F.binary_cross_entropy(p, target, reduction="none") * weights).sum(-1) / denominator
    brier = ((probability.float() - target).square() * weights).sum(-1) / denominator
    return dict(bce=bce, brier=brier, total=bce + brier)
