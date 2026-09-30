"""Downstream heads. `compute_metrics` returns the loss and metrics; CKPT_METRIC
selects checkpoints. Node stores holding `emb` skip the encoder."""

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import HeteroData

from remilp.models.encoder import MILPEncoder, mlp


def bernoulli_kl(target: torch.Tensor, logits: torch.Tensor, eps=1e-6) -> torch.Tensor:
    """KL(Bernoulli(target) || Bernoulli(sigmoid(logits))), elementwise."""
    t = target.float().clamp(eps, 1.0 - eps)
    ce = F.binary_cross_entropy_with_logits(logits.float(), t, reduction="none")
    entropy = -(t * t.log() + (1.0 - t) * (1.0 - t).log())
    return ce - entropy


class GapHead(nn.Module):

    CKPT_METRIC = "mae"
    cache_types = ("variables", "constraints")
    keep_fields: dict = {}

    def __init__(self, encoder: MILPEncoder):
        super().__init__()
        self.encoder = encoder
        d = encoder.instance_dim
        self.head = mlp(d, d, 1)

    def forward(self, data: HeteroData) -> torch.Tensor:
        if all("emb" in data[t] for t in self.cache_types):
            nodes = {t: data[t].emb for t in self.cache_types}
            instance = self.encoder.pool(nodes, data)["instance"]
        else:
            instance = self.encoder(data)["instance"]
        return self.head(instance).squeeze(-1)

    def compute_metrics(self, data: HeteroData) -> dict:
        pred = self(data)
        target = data.lp_ip_gap.view(-1)
        clipped = pred.clamp(0, 1)
        return {
            "loss": F.l1_loss(pred, target),
            "mae": F.l1_loss(clipped, target).detach(),
            "mse": F.mse_loss(clipped, target).detach(),
        }


class _NodeHead(nn.Module):
    """Per-node binary prediction against a soft target."""

    CKPT_METRIC = "kl"
    node_type: str
    soft_label: str
    hard_label: str
    mask_field: str | None = None

    def __init__(self, encoder: MILPEncoder):
        super().__init__()
        self.encoder = encoder
        h = encoder.hidden_dim
        self.head = mlp(h, h, 1)

    @property
    def cache_types(self) -> tuple:
        return (self.node_type,)

    @property
    def keep_fields(self) -> dict:
        if self.mask_field is None:
            return {}
        return {self.node_type: (self.mask_field,)}

    def forward(self, data: HeteroData) -> torch.Tensor:
        store = data[self.node_type]
        z = store.emb if "emb" in store else self.encoder(data)[self.node_type]
        return self.head(z).squeeze(-1)

    def compute_metrics(self, data: HeteroData) -> dict:
        logits = self(data)
        store = data[self.node_type]
        target = store[self.soft_label].view(-1)
        hard = store[self.hard_label].view(-1)
        if self.mask_field is not None:
            mask = store[self.mask_field].view(-1) > 0.5
            logits, target, hard = logits[mask], target[mask], hard[mask]
        return {
            "loss": F.binary_cross_entropy_with_logits(logits, target),
            "kl": bernoulli_kl(target, logits).mean().detach(),
            "accuracy": ((logits > 0) == (hard > 0.5)).float().mean().detach(),
        }


class SolutionHead(_NodeHead):
    """P(variable = 1), scored on binary variables only."""

    node_type = "variables"
    soft_label = "solution_probs"
    hard_label = "optimal_sol"
    mask_field = "is_binary"


class ActivityHead(_NodeHead):

    node_type = "constraints"
    soft_label = "active_probs"
    hard_label = "active_at_optimum"


HEADS = {"gap": GapHead, "solution": SolutionHead, "activity": ActivityHead}
