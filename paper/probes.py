"""Invariance and equivariance retrieval probe."""

import argparse
import dataclasses
import json
import statistics as st

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.transforms import Compose

from remilp.config import MODELS, RUNS_ROOT, ModelConfig, PretrainConfig, from_dict
from remilp.data.dataset import GraphDataset
from remilp.data.sources import source_root
from remilp.data.transforms import NormalizeObjective, NormalizeRows
from remilp.models.pretrained import build_encoder, encoder_file
from remilp.ssl.remilp import (
    T_DIM,
    TRANSVECTION,
    ReMILPViews,
    partner_index,
    scored_variables,
    transformation_features,
)
from remilp.tasks import TIERS
from remilp.train.common import seed_everything
from remilp.train.pretrain import pretrain_dir

STEP = 20000
ARMS = ("random", "forge", "remilp")


def load_arm(name, seed, device):
    """(encoder, hypernetwork or None, stored model config or None)."""
    seed_everything(seed)
    if name == "random":
        return build_encoder(MODELS["random"]).to(device).eval(), None, None
    run = pretrain_dir(MODELS[name], PretrainConfig(), seed)
    stored = json.loads((run / "config.json").read_text())["model"]
    encoder = build_encoder(from_dict(ModelConfig, stored))
    encoder.load_state_dict(
        torch.load(encoder_file(run, STEP), map_location="cpu", weights_only=True)
    )
    hyper = None
    if name == "remilp":
        state = torch.load(
            run / "checkpoint.pt", map_location="cpu", weights_only=False
        )
        h = encoder.hidden_dim
        hyper = torch.nn.Linear(T_DIM + h, h * h)
        hyper.load_state_dict(
            {
                k[len("hyper.") :]: v
                for k, v in state["objective"].items()
                if k.startswith("hyper.")
            }
        )
        hyper.to(device).eval()
    return encoder.to(device).eval(), hyper, stored


def ranks(query, keys, own):
    """Top-1 and reciprocal rank of key own[i] for query i, by cosine similarity."""
    hits, rr = 0.0, 0.0
    for i in range(0, query.shape[0], 4096):
        sim = query[i : i + 4096] @ keys.T
        idx = own[i : i + 4096]
        true = sim[torch.arange(idx.numel(), device=sim.device), idx]
        rank = 1 + (sim > true.unsqueeze(1)).sum(1)
        hits += (rank == 1).sum().item()
        rr += (1.0 / rank).sum().item()
    return hits / query.shape[0], rr / query.shape[0]


def embed(encoder, g, device):
    return F.normalize(
        encoder(Batch.from_data_list([g]).to(device))["variables"].float(), dim=1
    )


@torch.no_grad()
def invariance(encoder, views, ds, device):
    norm = Compose([NormalizeRows(), NormalizeObjective()])
    top1, mrr = [], []
    for i in range(len(ds)):
        g = ds[i]
        torch.manual_seed(i)
        za = embed(encoder, norm(g.clone()), device)
        zb = embed(encoder, views.reference(g.clone()), device)
        t, r = ranks(za, zb, torch.arange(za.shape[0], device=device))
        top1.append(t)
        mrr.append(r)
    return st.mean(top1), st.mean(mrr)


@torch.no_grad()
def equivariance(arms, views, ds, device):
    sums, n = {}, 0
    for i in range(len(ds)):
        torch.manual_seed(1000 + i)
        reference, transformed = views(ds[i].clone())
        ref = Batch.from_data_list([reference]).to(device)
        tr = Batch.from_data_list([transformed]).to(device)
        n_var = tr["variables"].num_nodes
        tv, shift = tr[TRANSVECTION], tr["variables"].transvection_shift.squeeze(-1)
        idx = scored_variables(tv, shift, n_var)
        if idx.numel() < 2:
            continue
        for name, (encoder, hyper, _) in arms.items():
            x_ref = encoder(ref)["variables"].float()
            target = F.normalize(encoder(tr)["variables"].float(), dim=1)
            queries = {name: F.normalize(x_ref[idx], dim=1)}
            if hyper is not None:
                h = x_ref.shape[1]
                partner = partner_index(tv, n_var)[idx]
                z_partner = x_ref.new_zeros(idx.numel(), h)
                has = partner >= 0
                z_partner[has] = x_ref[partner[has]]
                tfeat = transformation_features(tv, shift, n_var)[idx]
                ops = hyper(torch.cat([tfeat, z_partner], dim=1)).view(-1, h, h)
                queries[f"{name}+hyper"] = F.normalize(
                    torch.bmm(ops, x_ref[idx].unsqueeze(-1)).squeeze(-1), dim=1
                )
            for key, q in queries.items():
                t, r = ranks(q, target, idx)
                sums[key] = [a + b for a, b in zip(sums.get(key, [0.0, 0.0]), (t, r))]
        n += 1
    return {k: (t / n, r / n) for k, (t, r) in sums.items()}


def probe(seed, cls, diff, device):
    arms = {name: load_arm(name, seed, device) for name in ARMS}
    ds = GraphDataset(source_root("dmiplib", cls, diff), "test")
    cfg = from_dict(ModelConfig, arms["remilp"][2]).remilp
    redescribe = ReMILPViews(cfg)
    substitute = ReMILPViews(dataclasses.replace(cfg, aug_k_frac=0.0))
    out = {"invariance": {}, "equivariance": {}}
    for name, (encoder, _, _) in arms.items():
        out["invariance"][name] = invariance(encoder, redescribe, ds, device)
    out["equivariance"] = equivariance(arms, substitute, ds, device)
    return out


def table(results, pairs):
    rows = [
        ("Random", "random", "random"),
        ("FORGE", "forge", "forge"),
        ("ReMILP", "remilp", "remilp"),
        ("ReMILP+hyper", None, "remilp+hyper"),
    ]
    print(f"{len(pairs)} pairs, seeds {sorted(results)}")
    print(
        f"{'':14s} {'inv top-1':>16s} {'inv MRR':>16s} {'eq top-1':>16s} {'eq MRR':>16s}"
    )
    for label, inv, eq in rows:
        cols = []
        for probe_name, key in (("invariance", inv), ("equivariance", eq)):
            for j in (0, 1):
                if key is None:
                    cols.append("--")
                    continue
                per_seed = [
                    st.mean(results[s][p][probe_name][key][j] for p in pairs)
                    for s in results
                ]
                cols.append(f"{st.mean(per_seed):.3f} +- {st.stdev(per_seed):.3f}")
        print(f"{label:14s} " + " ".join(f"{c:>16s}" for c in cols))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = RUNS_ROOT / "probes"
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = [f"{c}-{d}" for c, d in TIERS]
    results = {}
    for seed in (int(s) for s in args.seeds.split(",")):
        results[seed] = {}
        for cls, diff in TIERS:
            path = out_dir / f"s{seed}_{cls}-{diff}.json"
            if not path.exists():
                path.write_text(json.dumps(probe(seed, cls, diff, device)))
                print(f"wrote {path}")
            results[seed][f"{cls}-{diff}"] = json.loads(path.read_text())
    table(results, pairs)


if __name__ == "__main__":
    main()
