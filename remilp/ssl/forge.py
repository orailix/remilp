"""FORGE: vector-quantized reconstruction of node features and coefficients."""

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.utils import negative_sampling

from remilp.config import ForgeConfig
from remilp.data.dataset import CON_FEATURES, V2C, VAR_FEATURES
from remilp.models.encoder import VQPooler, input_features, mlp
from remilp.ssl.base import Objective

BINARY_FEATURES = {
    "has_lb",
    "has_ub",
    "is_binary",
    "is_continuous",
    "has_lhs",
    "has_rhs",
}


class FeatureRecon(nn.Module):

    def __init__(self, h: int):
        super().__init__()
        self.var_head = mlp(h, h, len(VAR_FEATURES))
        self.con_head = mlp(h, h, len(CON_FEATURES))
        self.var_bin = [i for i, f in enumerate(VAR_FEATURES) if f in BINARY_FEATURES]
        self.var_cont = [
            i for i, f in enumerate(VAR_FEATURES) if f not in BINARY_FEATURES
        ]
        self.con_bin = [i for i, f in enumerate(CON_FEATURES) if f in BINARY_FEATURES]
        self.con_cont = [
            i for i, f in enumerate(CON_FEATURES) if f not in BINARY_FEATURES
        ]

    @staticmethod
    def _loss(pred, target, bin_idx, cont_idx):
        return F.binary_cross_entropy_with_logits(
            pred[:, bin_idx], target[:, bin_idx]
        ) + F.mse_loss(pred[:, cont_idx], target[:, cont_idx])

    def forward(self, q: dict, batch) -> torch.Tensor:
        var_target = input_features(batch["variables"], VAR_FEATURES)
        con_target = input_features(batch["constraints"], CON_FEATURES)
        return self._loss(
            self.var_head(q["variables"]), var_target, self.var_bin, self.var_cont
        ) + self._loss(
            self.con_head(q["constraints"]), con_target, self.con_bin, self.con_cont
        )


class CoeffRecon(nn.Module):

    def __init__(self, h: int, cfg: ForgeConfig):
        super().__init__()
        self.cfg = cfg
        self.pred_head = mlp(2 * h, h, 1)

    def _predict(self, x_var, x_con, ei):
        return self.pred_head(torch.cat([x_var[ei[0]], x_con[ei[1]]], dim=1)).squeeze(
            -1
        )

    def forward(self, q: dict, batch) -> torch.Tensor:
        x_var, x_con = q["variables"], q["constraints"]
        pos_ei = batch[V2C].edge_index
        coeff = batch[V2C].edge_attr.squeeze(-1)
        n_pos = pos_ei.shape[1]
        n_neg = int(n_pos * self.cfg.neg_to_pos_ratio)
        if n_pos + n_neg > self.cfg.max_edges_per_batch:
            n_pos = int(self.cfg.max_edges_per_batch / (1 + self.cfg.neg_to_pos_ratio))
            n_neg = self.cfg.max_edges_per_batch - n_pos
            keep = torch.randperm(pos_ei.shape[1], device=pos_ei.device)[:n_pos]
            pos_ei, coeff = pos_ei[:, keep], coeff[keep]
        neg_ei = negative_sampling(
            pos_ei, num_nodes=(x_var.shape[0], x_con.shape[0]), num_neg_samples=n_neg
        )
        preds = torch.cat(
            [self._predict(x_var, x_con, pos_ei), self._predict(x_var, x_con, neg_ei)]
        )
        targets = torch.cat([coeff, torch.zeros(neg_ei.shape[1], device=coeff.device)])
        return F.mse_loss(preds, targets)


class Forge(Objective):
    def __init__(self, encoder, cfg: ForgeConfig):
        super().__init__(encoder)
        if not isinstance(encoder.pooler, VQPooler):
            raise ValueError("the forge objective needs an encoder with pooler='vq'")
        self.cfg = cfg
        h = encoder.hidden_dim
        self.feat = FeatureRecon(h)
        self.coeff = CoeffRecon(h, cfg)

    def forward(self, batch):
        emb = self.encoder(batch)
        q = {"variables": emb["variables_q"], "constraints": emb["constraints_q"]}
        feat_loss = self.feat(q, batch)
        coeff_loss = self.coeff(q, batch)
        commit = emb["vq_commit_loss"]
        loss = feat_loss + coeff_loss + self.cfg.commitment * commit
        logs = {
            "feat_loss": feat_loss.item(),
            "coeff_loss": coeff_loss.item(),
            "commit": commit.item(),
            "perplexity": emb["vq_perplexity"].item(),
        }
        return loss, logs
