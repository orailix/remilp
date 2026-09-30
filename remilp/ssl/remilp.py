"""ReMILP: a hypernetwork predicts each substituted variable's embedding from its
reference embedding, contrasted with action and sample negatives."""

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.transforms import Compose

from remilp.config import ReMILPConfig
from remilp.data.transforms import (
    TRANSVECTION,
    AddRedundantConstraints,
    Homogenize,
    NormalizeObjective,
    NormalizeRows,
    TransformVariables,
    Unhomogenize,
)
from remilp.ssl.base import Objective

T_DIM = 4  # (log|u|, 1[u<0], lambda, mu)


class ReMILPViews:
    """(reference, transformed) views of one graph."""

    def __init__(self, cfg: ReMILPConfig):
        self.transvect = Compose(
            [
                Homogenize(),
                TransformVariables(
                    cfg.equiv_k_frac,
                    cfg.equiv_k_cap,
                    cfg.lambda_bound,
                    cfg.u_bound,
                ),
                Unhomogenize(),
            ]
        )
        self.nuisance = AddRedundantConstraints(cfg.aug_k_frac, cfg.aug_r)
        self.normalize = Compose([NormalizeRows(), NormalizeObjective()])

    def reference(self, g):
        return self.normalize(self.nuisance(g))

    def transformed(self, g):
        return self.normalize(self.nuisance(NormalizeRows()(self.transvect(g))))

    def __call__(self, g):
        return self.reference(g.clone()), self.transformed(g)


class ReMILP(Objective):
    def __init__(self, encoder, cfg: ReMILPConfig):
        super().__init__(encoder)
        if cfg.negative_types not in ("both", "sample"):
            raise ValueError(f"remilp.negative_types {cfg.negative_types!r}")
        self.cfg = cfg
        h = encoder.hidden_dim
        self.views = ReMILPViews(cfg)
        # Hypernetwork, initialised at the identity operator.
        self.hyper = nn.Linear(T_DIM + h, h * h)
        with torch.no_grad():
            self.hyper.weight.normal_(0.0, 0.02)
            self.hyper.bias.copy_(torch.eye(h).reshape(-1))

    def split_views(self, batch):
        if isinstance(batch, (list, tuple)):
            return batch
        transformed = Batch.from_data_list(
            [self.views.transformed(g) for g in batch.to_data_list()]
        )
        reference = Batch.from_data_list(
            [self.views.reference(g) for g in batch.to_data_list()]
        )
        return reference, transformed

    def forward(self, batch):
        reference, transformed = self.split_views(batch)
        x_trans = self.encoder(transformed)["variables"]
        x_ref = self.encoder(reference)["variables"]
        return self.loss(transformed, x_ref, x_trans)

    def loss(self, transformed, x_ref, x_trans):
        n_var = x_ref.shape[0]

        tv = transformed[TRANSVECTION]
        shift = transformed["variables"].transvection_shift.squeeze(-1)
        idx = scored_variables(
            tv, shift, n_var, transformed["variables"].batch, self.cfg.equiv_k_cap
        )
        if idx.numel() < 2:
            zero = x_trans.sum() * 0.0 + self.hyper.weight.sum() * 0.0
            return zero, {"n_scored": float(idx.numel())}

        transvected = bool(tv.edge_index.numel()) or bool((shift != 0).any())
        tfeat = transformation_features(tv, shift, n_var)[idx]
        partner = partner_index(tv, n_var)[idx]
        z_partner = x_ref.new_zeros(idx.numel(), x_ref.shape[1])
        has_partner = partner >= 0
        z_partner[has_partner] = x_ref[partner[has_partner]].detach()
        signature = torch.unique(
            torch.cat([tfeat, partner.unsqueeze(1).to(tfeat.dtype)], dim=1),
            dim=0,
            return_inverse=True,
        )[1]
        loss, logs = self.contrastive_loss(
            tfeat,
            x_ref[idx],
            x_trans[idx],
            z_partner,
            signature,
            x_trans,
            idx,
            identity=not transvected,
        )
        logs["n_scored"] = float(idx.numel())
        return loss, logs

    def operator(self, tfeat, z_partner) -> torch.Tensor:
        h = self.encoder.hidden_dim
        return self.hyper(torch.cat([tfeat, z_partner], dim=1)).view(-1, h, h)

    def contrastive_loss(
        self,
        tfeat,
        z_ref,
        z_target,
        z_partner,
        signature,
        negative_pool,
        anchor_index,
        identity=False,
    ):
        """One softmax per scored variable; with `identity` the prediction is the
        reference embedding and there are no action negatives."""
        k, device = z_ref.shape[0], z_ref.device
        if identity:
            pred = F.normalize(z_ref, dim=1)
        else:
            ops = self.operator(tfeat, z_partner)
            pred = F.normalize(torch.bmm(ops, z_ref.unsqueeze(-1)).squeeze(-1), dim=1)
        target = F.normalize(z_target, dim=1)
        scores = [(pred * target).sum(1, keepdim=True)]
        logs = {"collision_frac": 0.0}

        if self.cfg.negative_types == "both" and not identity:
            n_neg = min(self.cfg.negatives, max(1, k - 1))
            others = (
                torch.randint(1, k, (k, n_neg), device=device)
                + torch.arange(k, device=device).unsqueeze(1)
            ) % k
            # [k, k, h]: every operator applied to every embedding.
            pairs = torch.einsum("jab,ib->jia", ops, z_ref)
            rows = torch.arange(k, device=device).unsqueeze(1)
            action = pairs[others, rows]
            action_scores = (F.normalize(action, dim=2) * target.unsqueeze(1)).sum(2)
            # Negatives with the positive's own transformation are masked.
            collide = signature[others] == signature.unsqueeze(1)
            scores.append(action_scores.masked_fill(collide, -1e4))
            logs["collision_frac"] = collide.float().mean().item()

        n_all = negative_pool.shape[0]
        n_neg = min(self.cfg.negatives, max(1, n_all - 1))
        sampled = torch.randint(0, n_all, (k, n_neg), device=device)
        self_hit = sampled == anchor_index.unsqueeze(1)
        pool = F.normalize(negative_pool[sampled.reshape(-1)], dim=1)
        sample_scores = (pred.unsqueeze(1) * pool.view(k, n_neg, -1)).sum(2)
        scores.append(sample_scores.masked_fill(self_hit, -1e4))

        logits = torch.cat(scores, dim=1) / self.cfg.temperature
        labels = torch.zeros(k, dtype=torch.long, device=device)
        loss = F.cross_entropy(logits, labels)
        logs["pos_acc"] = (logits.argmax(1) == 0).float().mean().item()
        return loss, logs


def scored_variables(tv, shift, n_var, graph=None, cap=None) -> torch.Tensor:
    """Transformed variables; without any, up to `cap` random variables per graph."""
    device = shift.device
    target = torch.zeros(n_var, dtype=torch.bool, device=device)
    if tv.edge_index.numel():
        target[tv.edge_index[1]] = True
    target |= shift != 0
    if target.any():
        return target.nonzero(as_tuple=True)[0]
    if graph is None or cap is None:
        return torch.arange(n_var, device=device)
    order = torch.argsort(graph.double() + torch.rand(n_var, device=device))
    counts = torch.bincount(graph)
    starts = torch.cumsum(counts, 0) - counts
    rank = torch.arange(n_var, device=device) - starts[graph[order]]
    return order[rank < cap]


def transformation_features(tv, shift, n_var) -> torch.Tensor:
    """[n_var, 4] (log|u|, 1[u<0], lambda, mu) of each variable."""
    ei, attr = tv.edge_index, tv.edge_attr.squeeze(-1)
    row, col = ei[1], ei[0]
    diag = row == col
    u = attr[diag] + 1.0
    f = torch.zeros(n_var, T_DIM, device=shift.device)
    f[row[diag], 0] = u.abs().clamp(min=1e-6).log()
    f[row[diag], 1] = (u < 0).float()
    f[row[~diag], 2] = attr[~diag]
    f[:, 3] = torch.where(shift != 0, shift, f[:, 3])
    return f


def partner_index(tv, n_var) -> torch.Tensor:
    """Partner of each transformed variable, -1 without one."""
    p = torch.full((n_var,), -1, dtype=torch.long, device=tv.edge_index.device)
    ei = tv.edge_index
    if ei.numel():
        off = ei[1] != ei[0]
        p[ei[1][off]] = ei[0][off]
    return p
