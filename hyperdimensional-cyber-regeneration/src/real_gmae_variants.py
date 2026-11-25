"""Two exploratory GraphMAE adaptations for the frozen CTU-13 flow task.

EDGE changes only observable message weights. CONS changes only training:
two masked-view reconstruction/alignment and optional prediction consistency.
Neither reads labels in prediction, pretraining, gating, or augmentation. All
four keys retain GraphFlow's parameter count; extra forwards are not free FLOPs.
"""
import torch
import torch.nn.functional as F

try:
    from .real_models import (GraphFlow, FLOW_FEATURES, EPS, observable_input,
                              observed_reconstruction_loss, normalized_adjacency,
                              require_cuda)
except ImportError:
    from real_models import (GraphFlow, FLOW_FEATURES, EPS, observable_input,
                             observed_reconstruction_loss, normalized_adjacency,
                             require_cuda)

KEYS = ("EDGE_GMAE", "EDGE_DENSE", "CONS_GMAE", "CONS_PRE")
EDGE_FLOOR = .1
EDGE_TEMPERATURE = 2.
NODE_MASK_RATE = .3
LATENT_WEIGHT = .5
FEATURE_MASK_RATE = .2
PREDICTION_WEIGHT = .5


def valid_mean(values, active):
    """One masked mean per window, with an exact zero for all-padding rows."""
    values = torch.where(active, values.float(), 0.)
    return values.sum(-1) / active.sum(-1).float().clamp_min(1.)


def reliability_edges(x, adj, active, observed=None):
    """Observable overlap/compatibility edge weights, before row normalization.

    r_ij = sum_k m_ik m_jk / 14;
    d_ij = sum_k m_ik m_jk (x_ik-x_jk)^2 / max(1,sum_k m_ik m_jk);
    W_ij = A_ij [0.1 + 0.9 r_ij exp(-d_ij / 2)].

    Every valid diagonal is forced to one. An explicit availability override
    clears additionally masked nodes during pretraining: the learned mask token
    is not mistaken for an actual received measurement. No clean target is used.
    """
    require_cuda(x)
    require_cuda(adj)
    require_cuda(active)
    if x.shape[:2] != active.shape or x.shape[-1] != 2*FLOW_FEATURES:
        raise ValueError("Expected x[N,F,28] and active[N,F]")
    if adj.shape != (x.shape[0], x.shape[1], x.shape[1]):
        raise ValueError("Expected adj[N,F,F]")
    with torch.autocast(device_type="cuda", enabled=False):
        availability = x[..., FLOW_FEATURES:] if observed is None else observed
        require_cuda(availability)
        if availability.shape != x.shape[:-1]+(FLOW_FEATURES,):
            raise ValueError("Expected availability[N,F,14]")
        availability = torch.where(active[..., None], availability.float(), 0.).clamp(0., 1.)
        values = torch.where((availability > 0) & active[..., None], x[..., :FLOW_FEATURES].float(), 0.)
        overlap = availability[:, :, None, :] * availability[:, None, :, :]
        support = overlap.sum(-1)
        squared_difference = (values[:, :, None, :] - values[:, None, :, :]).square()
        distance = (overlap * squared_difference).sum(-1) / support.clamp_min(1.)
        compatible = (support / FLOW_FEATURES) * torch.exp(-distance / EDGE_TEMPERATURE)
        real_edges = active[:, :, None] & active[:, None, :]
        weights = torch.where(real_edges, adj.float(), 0.).clamp_min(0.)
        weights = weights * (EDGE_FLOOR + (1.-EDGE_FLOOR)*compatible)
        diagonal = torch.eye(x.shape[1], device=x.device, dtype=torch.bool)[None]
        weights = torch.where(diagonal & real_edges, torch.ones_like(weights), weights)
        return torch.where(real_edges, weights, 0.)


def feature_masked_view(batch, rate=FEATURE_MASK_RATE):
    """Hide independently selected observed coordinates, including indicators.

    Only x/adj/valid are returned. Labels and metadata cannot flow into a view.
    This starts from the currently observed input, never hidden clean features.
    """
    x, active = observable_input(batch)
    observed = x[..., FLOW_FEATURES:]
    keep = (torch.rand_like(observed) >= rate).float() * observed
    values = torch.where(keep > 0, x[..., :FLOW_FEATURES], 0.)
    return dict(x=torch.cat((values, keep), -1), adj=batch["adj"], valid=active)


def symmetric_latent_consistency(first, second, active):
    """Symmetric stop-gradient cosine discrepancy, averaged over valid nodes."""
    require_cuda(first)
    require_cuda(second)
    if first.shape != second.shape or first.shape[:2] != active.shape:
        raise ValueError("Expected two latent arrays[N,F,H] and active[N,F]")
    with torch.autocast(device_type="cuda", enabled=False):
        first = torch.where(active[..., None], first.float(), 0.)
        second = torch.where(active[..., None], second.float(), 0.)
        u, v = F.normalize(first, dim=-1, eps=EPS), F.normalize(second, dim=-1, eps=EPS)
        # Clamping handles a possible final FP32 dot-product ulp at +/-1.
        uv = (u*v.detach()).sum(-1).clamp(-1., 1.)
        vu = (v*u.detach()).sum(-1).clamp(-1., 1.)
        return valid_mean(.5*((1.-uv)+(1.-vu)), active)


class VariantFlow(GraphFlow):
    def __init__(self, key):
        super().__init__("GMAE")
        self.key = key

    def supervised_regularizer(self, batch, probability):
        # An explicit per-window zero lets the runner share one objective API.
        require_cuda(probability)
        return probability.new_zeros(probability.shape[0], dtype=torch.float32)


class EdgeFlow(VariantFlow):
    """GraphMAE architecture with a fixed observable reliability edge gate."""
    def __init__(self, key):
        if key not in ("EDGE_GMAE", "EDGE_DENSE"):
            raise ValueError(key)
        super().__init__(key)

    def encode(self, x, adj, active, observed=None):
        require_cuda(x)
        x = torch.where(active[..., None], x, 0.)
        graph = normalized_adjacency(reliability_edges(x, adj, active, observed), active)
        h = F.gelu(self.input(x))
        h = torch.where(active[..., None], h, 0.)
        for local, neighbor, norm in zip(self.self_layers, self.neighbor_layers, self.norms):
            pooled = torch.bmm(graph.to(h.dtype), h)
            h = norm(h + F.gelu(local(h) + neighbor(pooled)))
            h = torch.where(active[..., None], h, 0.)
        return h, graph

    def pretraining_loss(self, batch, mask_rate=NODE_MASK_RATE):
        target, active = observable_input(batch)
        selected = (torch.rand(active.shape, device=target.device) < mask_rate) & active
        masked = torch.where(selected[..., None], self.mask_token[None, None, :], target)
        availability = torch.where(selected[..., None], 0., target[..., FLOW_FEATURES:])
        h, graph = self.encode(masked, batch["adj"], active, observed=availability)
        h = torch.where(selected[..., None], torch.zeros_like(h), h)
        reconstructed = self.feature_decoder(h + torch.bmm(graph.to(h.dtype), h))
        return observed_reconstruction_loss(reconstructed, target, selected)


class ConsistencyFlow(VariantFlow):
    """GraphFlow inference with two-view masked reconstruction/alignment.

    Each pretraining step uses two independently node-masked encoder passes;
    their encodes supply both reconstruction and symmetric stopped-target latent
    alignment. CONS_GMAE adds a second prediction on a feature-masked training
    view, targeting the detached original prediction. CONS_PRE omits that term.
    """
    def __init__(self, key):
        if key not in ("CONS_GMAE", "CONS_PRE"):
            raise ValueError(key)
        super().__init__(key)

    def _masked_view(self, target, active, adj, mask_rate):
        selected = (torch.rand(active.shape, device=target.device) < mask_rate) & active
        masked = torch.where(selected[..., None], self.mask_token[None, None, :], target)
        latent, graph = self.encode(masked, adj, active)
        remasked = torch.where(selected[..., None], torch.zeros_like(latent), latent)
        reconstruction = self.feature_decoder(remasked + torch.bmm(graph.to(latent.dtype), remasked))
        loss = observed_reconstruction_loss(reconstruction, target, selected)
        return latent, loss

    def pretraining_terms(self, batch, mask_rate=NODE_MASK_RATE):
        target, active = observable_input(batch)
        first, rec_first = self._masked_view(target, active, batch["adj"], mask_rate)
        second, rec_second = self._masked_view(target, active, batch["adj"], mask_rate)
        reconstruction = .5*(rec_first+rec_second)
        consistency = symmetric_latent_consistency(first, second, active)
        return dict(reconstruction=reconstruction, consistency=consistency,
                    total=reconstruction + LATENT_WEIGHT*consistency)

    def pretraining_loss(self, batch, mask_rate=NODE_MASK_RATE):
        return self.pretraining_terms(batch, mask_rate)["total"]

    def supervised_regularizer(self, batch, probability):
        require_cuda(probability)
        if self.key == "CONS_PRE":
            return super().supervised_regularizer(batch, probability)
        augmented = feature_masked_view(batch)
        alternative = self(augmented).float().sigmoid()
        error = (alternative-probability.float().detach()).square()
        return PREDICTION_WEIGHT*valid_mean(error, augmented["valid"])


def make_model(key):
    if key in ("EDGE_GMAE", "EDGE_DENSE"):
        return EdgeFlow(key)
    if key in ("CONS_GMAE", "CONS_PRE"):
        return ConsistencyFlow(key)
    raise ValueError(f"Unknown GraphMAE variant: {key}")
