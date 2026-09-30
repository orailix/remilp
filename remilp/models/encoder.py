"""Bipartite GATv2 encoder with jumping knowledge and an instance pooler."""

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import scatter, softmax

from remilp.config import EncoderConfig
from remilp.data.dataset import C2V, CON_FEATURES, V2C, VAR_FEATURES


def mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, out_dim)
    )


def _batch_index(x: torch.Tensor, batch) -> torch.Tensor:
    return (
        torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        if batch is None
        else batch
    )


class CoordinateAttentionAggregation(nn.Module):
    """Attention pooling with one softmax per output coordinate."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.value_nn = nn.Linear(in_dim, out_dim)
        self.gate_nn = mlp(in_dim, in_dim, out_dim)

    def forward(self, x, index, num_graphs):
        gate = softmax(self.gate_nn(x), index, dim=0, num_nodes=num_graphs)
        return scatter(
            gate * self.value_nn(x), index, dim=0, dim_size=num_graphs, reduce="sum"
        )


class CoordinateAttentionPooler(nn.Module):
    """Pools variables and constraints separately (out_dim // 2 each) and concatenates."""

    def __init__(self, hidden_dim: int, out_dim: int):
        super().__init__()
        self.out_dim = out_dim
        self.var_pool = CoordinateAttentionAggregation(hidden_dim, out_dim // 2)
        self.con_pool = CoordinateAttentionAggregation(hidden_dim, out_dim // 2)

    def forward(self, node_embeddings: dict, data: HeteroData) -> torch.Tensor:
        var, con = node_embeddings["variables"], node_embeddings["constraints"]
        n = data.num_graphs
        return torch.cat(
            [
                self.var_pool(
                    var, _batch_index(var, data["variables"].get("batch")), n
                ),
                self.con_pool(
                    con, _batch_index(con, data["constraints"].get("batch")), n
                ),
            ],
            dim=-1,
        )


class VQPooler(nn.Module):
    """EMA vector quantization of node embeddings; the instance embedding is the
    histogram of codes. Quantized latents and commitment loss are in `last_vq`."""

    def __init__(
        self, hidden_dim: int, codebook_size: int, decay: float, eps: float = 1e-5
    ):
        super().__init__()
        self.out_dim = codebook_size
        self.decay = decay
        self.eps = eps
        codebook = torch.randn(codebook_size, hidden_dim)
        self.register_buffer("codebook", codebook)
        self.register_buffer("ema_count", torch.ones(codebook_size))
        self.register_buffer("ema_sum", codebook.clone())
        self.register_buffer("initialized", torch.zeros((), dtype=torch.bool))
        self.last_vq: dict | None = None

    def _quantize(self, x: torch.Tensor):
        updating = self.training and x.requires_grad
        if updating and not self.initialized:
            with torch.no_grad():
                idx = torch.randint(
                    0, x.shape[0], (self.codebook.shape[0],), device=x.device
                )
                self.codebook.copy_(x.detach().float()[idx])
                self.ema_sum.copy_(self.codebook)
                self.initialized.fill_(True)
        assignment = self._assign(x.detach().float())
        k = self.codebook.shape[0]
        q = self.codebook[assignment].to(x.dtype)
        if updating:
            with torch.no_grad():
                d = self.decay
                hits = torch.bincount(assignment, minlength=k).float()
                sums = torch.zeros_like(self.ema_sum)
                sums.index_add_(0, assignment, x.detach().float())
                self.ema_count.mul_(d).add_(hits, alpha=1 - d)
                self.ema_sum.mul_(d).add_(sums, alpha=1 - d)
                n = self.ema_count.sum()
                count = (self.ema_count + self.eps) / (n + k * self.eps) * n
                self.codebook.copy_(self.ema_sum / count.unsqueeze(1))
        commit = F.mse_loss(x, q.detach())
        return x + (q - x).detach(), assignment, commit

    def _assign(self, x: torch.Tensor) -> torch.Tensor:
        """Nearest code of every row, in chunks."""
        k = self.codebook.shape[0]
        step = max(1, 2**22 // max(k, 1))
        if x.shape[0] <= step:
            return torch.cdist(x, self.codebook).argmin(dim=1)
        out = torch.empty(x.shape[0], dtype=torch.long, device=x.device)
        for i in range(0, x.shape[0], step):
            out[i : i + step] = torch.cdist(x[i : i + step], self.codebook).argmin(
                dim=1
            )
        return out

    def forward(self, node_embeddings: dict, data: HeteroData) -> torch.Tensor:
        var, con = node_embeddings["variables"], node_embeddings["constraints"]
        q, assignment, commit = self._quantize(torch.cat([var, con], dim=0))
        batch = torch.cat(
            [
                _batch_index(var, data["variables"].get("batch")),
                _batch_index(con, data["constraints"].get("batch")),
            ]
        )
        k = self.codebook.shape[0]
        counts = (
            torch.bincount(batch * k + assignment, minlength=data.num_graphs * k)
            .view(data.num_graphs, k)
            .float()
        )
        probs = counts.sum(0) / assignment.shape[0]
        self.last_vq = {
            "variables_q": q[: var.shape[0]],
            "constraints_q": q[var.shape[0] :],
            "vq_commit_loss": commit,
            "vq_perplexity": torch.exp(-(probs * (probs + 1e-10).log()).sum()),
        }
        return (counts / counts.sum(dim=1, keepdim=True).clamp(min=1)).to(var.dtype)


def _gatv2(cfg: EncoderConfig) -> GATv2Conv:
    h = cfg.hidden_dim
    if h % cfg.heads:
        raise ValueError(f"hidden_dim ({h}) must be divisible by heads ({cfg.heads})")
    # No self-loops: the graph is bipartite.
    return GATv2Conv(
        (h, h), h // cfg.heads, heads=cfg.heads, edge_dim=h, add_self_loops=False
    )


def input_features(store, names) -> torch.Tensor:
    return torch.cat([store[f].float() for f in names], dim=1)


NODE_INPUTS = {"variables": VAR_FEATURES, "constraints": CON_FEATURES}


class MILPEncoder(nn.Module):
    """Returns {"variables": [n_var, h], "constraints": [n_con, h], "instance": [n_graphs, d]}."""

    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        h, n_layers = cfg.hidden_dim, cfg.n_layers
        self.cfg = cfg
        self.hidden_dim = h
        self.var_proj = nn.Linear(len(VAR_FEATURES), h)
        self.con_proj = nn.Linear(len(CON_FEATURES), h)
        self.edge_proj = nn.Linear(1, h)
        self.var_to_con_convs = nn.ModuleList([_gatv2(cfg) for _ in range(n_layers)])
        self.con_to_var_convs = nn.ModuleList([_gatv2(cfg) for _ in range(n_layers)])
        self.jk_con_proj = nn.Linear((n_layers + 1) * h, h)
        self.jk_var_proj = nn.Linear((n_layers + 1) * h, h)
        if cfg.pooler == "coord_attention":
            self.pooler = CoordinateAttentionPooler(h, cfg.pooler_out_dim)
        elif cfg.pooler == "vq":
            self.pooler = VQPooler(h, cfg.vq_codebook_size, cfg.vq_decay)
        else:
            raise ValueError(f"unknown pooler '{cfg.pooler}'")
        pooler_out = self.pooler.out_dim
        self.instance_dim = max(h, min(pooler_out, round(math.sqrt(h * pooler_out))))
        self.inst_mlp = mlp(pooler_out, self.instance_dim, self.instance_dim)

    def forward(self, data: HeteroData) -> dict:
        x_var = input_features(data["variables"], VAR_FEATURES)
        x_con = input_features(data["constraints"], CON_FEATURES)
        x_var = torch.relu(self.var_proj(x_var))
        x_con = torch.relu(self.con_proj(x_con))
        v2c_edges = self.edge_proj(data[V2C].edge_attr)
        c2v_edges = self.edge_proj(data[C2V].edge_attr)

        layers_con, layers_var = [x_con], [x_var]
        for v2c_conv, c2v_conv in zip(self.var_to_con_convs, self.con_to_var_convs):
            x_con = torch.relu(
                v2c_conv((x_var, x_con), data[V2C].edge_index, edge_attr=v2c_edges)
            )
            x_var = torch.relu(
                c2v_conv((x_con, x_var), data[C2V].edge_index, edge_attr=c2v_edges)
            )
            layers_con.append(x_con)
            layers_var.append(x_var)
        x_con = self.jk_con_proj(torch.cat(layers_con, dim=1))
        x_var = self.jk_var_proj(torch.cat(layers_var, dim=1))

        nodes = {"variables": x_var, "constraints": x_con}
        return {**nodes, **self.pool(nodes, data)}

    def pool(self, nodes: dict, data: HeteroData) -> dict:
        """Instance embedding (and the VQ outputs) from node embeddings."""
        out = {"instance": self.inst_mlp(self.pooler(nodes, data))}
        if isinstance(self.pooler, VQPooler):
            out.update(self.pooler.last_vq)
        return out
