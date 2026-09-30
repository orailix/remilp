"""Train a task head on an encoder, frozen or fine-tuned, with early stopping on
validation; results.json holds the test metrics of the selected checkpoint."""

import copy
import json
import time
from pathlib import Path

import torch
from torch_geometric.data import Batch, HeteroData
from tqdm import tqdm

from remilp.config import (
    EvalConfig,
    ModelConfig,
    config_hash,
    run_relevant,
    to_dict,
)
from remilp.data.dataset import DualBudgetBatchSampler, make_loader
from remilp.data.sources import task_datasets
from remilp.models.encoder import NODE_INPUTS, VQPooler
from remilp.models.heads import HEADS
from remilp.models.pretrained import build_encoder, encoder_file, load_encoder_state
from remilp.tasks import parse_task
from remilp.train.common import (
    STALL_TIMEOUT_S,
    ProgressWatchdog,
    RunLogger,
    autocast,
    compile_encoder,
    device_and_bf16,
    load_checkpoint,
    read_json,
    save_checkpoint,
    seed_everything,
    unwrap,
)


def eval_hash(
    model: ModelConfig, cfg: EvalConfig, seed: int, pretrain_run, step
) -> str:
    return config_hash(
        {
            "model": to_dict(model),
            "eval": run_relevant(cfg),
            "seed": seed,
            "pretrain_run": None if pretrain_run is None else str(pretrain_run),
            "step": step,
        }
    )


def run_eval(
    model: ModelConfig,
    cfg: EvalConfig,
    seed: int,
    run_dir: Path,
    pretrain_run: Path | None = None,
    step: int | None = None,
    force: bool = False,
) -> dict:
    run_dir = Path(run_dir)
    config = {
        "kind": "eval",
        "model": to_dict(model),
        "eval": to_dict(cfg),
        "seed": seed,
        "model_hash": config_hash(model),
        "pretrain_run": None if pretrain_run is None else str(pretrain_run),
        "pretrain_step": step,
        "eval_hash": eval_hash(model, cfg, seed, pretrain_run, step),
    }
    if pretrain_run is not None:
        config["pretrain_hash"] = read_json(Path(pretrain_run) / "config.json").get(
            "pretrain_hash"
        )
    if (run_dir / "config.json").exists() and not force:
        existing = read_json(run_dir / "config.json").get("eval_hash")
        if existing != config["eval_hash"]:
            raise RuntimeError(
                f"{run_dir} holds a run with a different config; pass --force to overwrite"
            )
    if (run_dir / "results.json").exists() and not force:
        print(f"{run_dir}: already done")
        return read_json(run_dir / "results.json")
    ckpt = run_dir / "checkpoint.pt"
    if ckpt.exists():
        step_done = torch.load(ckpt, map_location="cpu", weights_only=False)["step"]
        _trim_metrics(run_dir / "metrics.jsonl", step_done)
    else:
        (run_dir / "metrics.jsonl").unlink(missing_ok=True)
    logger = RunLogger(run_dir, config)
    try:
        return _evaluate(model, cfg, seed, run_dir, pretrain_run, step, logger)
    except BaseException:
        logger.fail()
        raise


def _trim_metrics(path: Path, step: int) -> None:
    if not path.exists():
        return
    rows = [l for l in path.open() if json.loads(l)["step"] <= step]
    path.write_text("".join(rows))


def _evaluate(model, cfg, seed, run_dir, pretrain_run, step, logger):
    device, bf16 = device_and_bf16(cfg.bf16)
    if device == "cuda":
        torch.set_num_threads(1)
    t0 = time.time()
    task = parse_task(cfg.task)
    datasets = task_datasets(task)

    seed_everything(seed)
    train_ds = datasets["train"]
    n_train = min(cfg.train_size, len(train_ds))
    train_ds = train_ds.subset(list(range(n_train)))  # the split is in id order

    encoder = build_encoder(model)
    net = HEADS[task.kind](encoder)
    if pretrain_run is not None:
        state = torch.load(
            encoder_file(pretrain_run, step), map_location="cpu", weights_only=True
        )
        left = load_encoder_state(encoder, state)
        if left:
            print(
                f"pooling head initialised afresh ({len(left)} tensors not in the saved encoder)"
            )
    net.to(device)
    if cfg.frozen:
        # Frozen: only the pooler, instance MLP and head train.
        trainable = {
            id(p) for m in (encoder.pooler, encoder.inst_mlp) for p in m.parameters()
        }
        for p in encoder.parameters():
            p.requires_grad_(id(p) in trainable)
    optimizer = torch.optim.Adam(
        [p for p in net.parameters() if p.requires_grad], lr=cfg.lr
    )
    ckpt_path = run_dir / "checkpoint.pt"
    best_val, best_step, stale, stopped_early = float("inf"), 0, 0, False
    step_i, recent, nonfinite = 0, [], 0
    if ckpt_path.exists():
        ck = load_checkpoint(ckpt_path, device)
        net.load_state_dict(ck["net"])
        optimizer.load_state_dict(ck["optimizer"])
        step_i, best_val, best_step = ck["step"], ck["best_val"], ck["best_step"]
        stale, nonfinite = ck["stale"], ck["nonfinite"]
        print(f"resumed from step {step_i:,}")

    cache, budget = {}, cfg.cache_gb * 2**30
    if cfg.frozen and cfg.cache_gb > 0:
        splits = [("train", train_ds), ("val", datasets["val"])]
        if cfg.test_curve:
            splits.append(("test", datasets["test"]))
        for split, ds in splits:
            built = build_cache(net, ds, device, bf16, cfg, budget)
            if built is None:
                print(
                    f"{split} split: node embeddings exceed the cache budget, not cached"
                )
                continue
            cache[split], used = built
            budget -= used
            print(f"{split} split: node embeddings cached ({used / 2**20:.0f} MiB)")
    split_ds = {"train": train_ds, "val": datasets["val"], "test": datasets["test"]}
    split_ds.update(cache)
    loaders = {
        split: make_loader(
            ds,
            cfg.batch_nodes,
            cfg.batch_edges,
            0 if split in cache else cfg.num_workers,
            shuffle=split == "train",
        )
        for split, ds in split_ds.items()
    }
    if cfg.compile and device == "cuda" and not {"train", "val"} <= set(cache):
        if isinstance(encoder.pooler, VQPooler):
            raw = make_loader(
                train_ds, cfg.batch_nodes, cfg.batch_edges, 0, shuffle=False
            )
            with torch.no_grad(), autocast(bf16):  # codebook initialisation
                encoder(next(iter(raw)).to(device))
        net.encoder = compile_encoder(encoder)

    metric = net.CKPT_METRIC
    patience_checks = -(-cfg.patience // cfg.val_every) if cfg.patience else 0
    progress = tqdm(
        total=cfg.max_steps,
        initial=step_i,
        desc=f"{cfg.task} {model.name}",
        smoothing=0,
    )
    watchdog = ProgressWatchdog(STALL_TIMEOUT_S, f"{cfg.task}")
    with watchdog:
        while step_i < cfg.max_steps and not stopped_early:
            net.train()
            for batch in loaders["train"]:
                watchdog.tick()
                if step_i >= cfg.max_steps:
                    break
                batch = batch.to(device, non_blocking=True)
                with autocast(bf16):
                    metrics = net.compute_metrics(batch)
                loss = metrics["loss"]
                optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                if torch.isfinite(grad_norm):
                    optimizer.step()
                else:
                    nonfinite += 1
                recent.append(loss.item())
                if step_i % cfg.log_every == 0:
                    logger.log(
                        step_i, "train", loss=loss.item(), grad_norm=grad_norm.item()
                    )
                step_i += 1
                progress.update(1)

                if step_i % cfg.val_every == 0:
                    val = evaluate_loader(net, loaders["val"], device, bf16)
                    score = val[metric]
                    logger.log(
                        step_i,
                        "val",
                        recent_loss=sum(recent) / len(recent),
                        **{f"val_{k}": v for k, v in val.items()},
                    )
                    recent = []
                    if cfg.test_curve:
                        curve = evaluate_loader(net, loaders["test"], device, bf16)
                        logger.log(
                            step_i, "test", **{f"test_{k}": v for k, v in curve.items()}
                        )
                    if score < best_val:
                        best_val, best_step, stale = score, step_i, 0
                        torch.save(net.state_dict(), run_dir / "best.pt")
                    else:
                        stale += 1
                    if patience_checks and stale >= patience_checks:
                        stopped_early = True
                        print(
                            f"early stop at step {step_i}: best {metric} {best_val:.5f} at step {best_step}"
                        )
                        break
                    if step_i % cfg.ckpt_every == 0 or step_i == cfg.max_steps:
                        save_checkpoint(
                            ckpt_path,
                            {
                                "step": step_i,
                                "net": {
                                    k.replace("_orig_mod.", ""): v
                                    for k, v in net.state_dict().items()
                                },
                                "optimizer": optimizer.state_dict(),
                                "best_val": best_val,
                                "best_step": best_step,
                                "stale": stale,
                                "nonfinite": nonfinite,
                            },
                        )
                    net.train()
    progress.close()
    if best_step == 0:
        torch.save(net.state_dict(), run_dir / "best.pt")

    net.load_state_dict(
        torch.load(run_dir / "best.pt", map_location=device, weights_only=True)
    )
    test = evaluate_loader(net, loaders["test"], device, bf16)
    test.pop("loss", None)
    logger.log(step_i, "test", checkpoint="best", **test)
    results = {
        **test,
        "n_train": n_train,
        "fit_final_step": step_i,
        "fit_best_step": best_step,
        "fit_best_val": None if best_val == float("inf") else best_val,
        "fit_stopped_early": stopped_early,
        "nonfinite_skipped": nonfinite,
        "wall_s": time.time() - t0,
    }
    print(", ".join(f"{k}={v:.4f}" for k, v in test.items()))
    logger.finish(results)
    ckpt_path.unlink(missing_ok=True)
    return results


class _CachedSplit(torch.utils.data.Dataset):

    def __init__(self, items, sizes):
        self.items, self.sizes = items, sizes

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        return self.items[i]


def _reduced(g: HeteroData, emb: dict, keep: dict) -> HeteroData:
    """The graph without edges and encoder inputs, with node embeddings as `emb`."""
    out = copy.copy(g)
    for edge_type in out.edge_types:
        del out[edge_type]
    for t in out.node_types:
        out[t].num_nodes = g[t].num_nodes
        for f in NODE_INPUTS[t]:
            if f in out[t] and f not in keep.get(t, ()):
                del out[t][f]
    for t, z in emb.items():
        out[t].emb = z
    return out


@torch.no_grad()
def build_cache(net, dataset, device, bf16, cfg, budget: float):
    """(reduced graphs, bytes) of one frozen-encoder pass, or None over `budget`."""
    types = net.cache_types
    h = unwrap(net.encoder).hidden_dim
    itemsize = 2 if bf16 else 4
    if len(types) == 2 and dataset.sizes[:, 0].sum() * h * itemsize > budget:
        return None
    sampler = DualBudgetBatchSampler(dataset.sizes, cfg.batch_nodes, cfg.batch_edges)
    items, used = [None] * len(dataset), 0
    encoder = unwrap(net.encoder)
    was_training = encoder.training
    encoder.eval()
    try:
        for idx in sampler:
            graphs = [dataset[i] for i in idx]
            batch = Batch.from_data_list(graphs).to(device)
            with autocast(bf16):
                out = encoder(batch)
            parts = {}
            for t in types:
                z = out[t].detach().cpu()
                used += z.numel() * z.element_size()
                counts = torch.bincount(batch[t].batch.cpu(), minlength=len(idx))
                parts[t] = z.split(counts.tolist())
            for j, (i, g) in enumerate(zip(idx, graphs)):
                items[i] = _reduced(g, {t: parts[t][j] for t in types}, net.keep_fields)
            if used > budget:
                return None
    finally:
        encoder.train(was_training)
    return _CachedSplit(items, dataset.sizes), used


@torch.no_grad()
def evaluate_loader(net, loader, device, bf16) -> dict:
    net.eval()
    sums, n = {}, 0
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        with autocast(bf16):
            metrics = net.compute_metrics(batch)
        for k, v in metrics.items():
            sums[k] = sums.get(k, 0.0) + float(v)
        n += 1
    return {k: v / max(n, 1) for k, v in sums.items()}
