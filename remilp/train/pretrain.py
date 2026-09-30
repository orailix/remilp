import time
from pathlib import Path

import torch
from tqdm import tqdm

from remilp.config import (
    RUNS_ROOT,
    ModelConfig,
    PretrainConfig,
    config_hash,
    model_tag,
    run_relevant,
    to_dict,
)
from remilp.data.dataset import make_loader, num_graphs, to_device
from remilp.data.sources import pretraining_pool
from remilp.models.encoder import VQPooler
from remilp.models.pretrained import build_encoder, save_encoder
from remilp.ssl import OBJECTIVES
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


def pretrain_hash(model: ModelConfig, cfg: PretrainConfig, seed: int) -> str:
    return config_hash(
        {"model": to_dict(model), "pretrain": run_relevant(cfg), "seed": seed}
    )


def pretrain_dir(model: ModelConfig, cfg: PretrainConfig, seed: int) -> Path:
    """runs/pretrain/<model_tag>[-<non-default pretrain fields>]/seed<k>"""
    tag = model_tag(model)
    extra = [
        f"{k.replace('_', '')}{v}"
        for k, v in run_relevant(cfg).items()
        if v != run_relevant(PretrainConfig())[k]
    ]
    if extra:
        tag += "-" + "-".join(extra)
    return RUNS_ROOT / "pretrain" / tag / f"seed{seed}"


def run_pretrain(
    model: ModelConfig,
    cfg: PretrainConfig,
    seed: int,
    run_dir: Path,
    force=False,
) -> Path:
    run_dir = Path(run_dir)
    if model.name == "random":
        raise ValueError("the random model is not pretrained")
    config = {
        "kind": "pretrain",
        "model": to_dict(model),
        "pretrain": to_dict(cfg),
        "seed": seed,
        "model_hash": config_hash(model),
        "pretrain_hash": pretrain_hash(model, cfg, seed),
    }
    if (run_dir / "config.json").exists() and not force:
        existing = read_json(run_dir / "config.json").get("pretrain_hash")
        if existing != config["pretrain_hash"]:
            raise RuntimeError(
                f"{run_dir} holds a run with a different config; pass --force to overwrite"
            )
    if (
        (run_dir / "results.json").exists()
        and (run_dir / "encoder.pt").exists()
        and not force
    ):
        print(f"{run_dir}: already done")
        return run_dir
    logger = RunLogger(run_dir, config)
    try:
        _pretrain(model, cfg, seed, run_dir, logger)
    except BaseException:
        logger.fail()
        raise
    return run_dir


def _pretrain(model, cfg, seed, run_dir, logger):
    device, bf16 = device_and_bf16(cfg.bf16)
    t0 = time.time()
    seed_everything(seed)
    encoder = build_encoder(model).to(device)
    train_ds, val_ds = pretraining_pool()
    objective = OBJECTIVES[model.name](encoder, model.objective).to(device)
    train_ds.set_transform(objective.views)
    val_ds.set_transform(objective.views)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(objective.parameters()), lr=cfg.lr
    )
    all_params = list(encoder.parameters()) + list(objective.parameters())

    step, best_val, best_state, graphs_seen, nonfinite = 0, float("inf"), None, 0, 0
    ckpt_path = run_dir / "checkpoint.pt"
    if ckpt_path.exists():
        state = load_checkpoint(ckpt_path, device)
        encoder.load_state_dict(state["encoder"])
        objective.load_state_dict(state["objective"])
        optimizer.load_state_dict(state["optimizer"])
        step, best_val, best_state = (
            state["step"],
            state["best_val"],
            state["best_encoder_state"],
        )
        graphs_seen, nonfinite = state["graphs_seen"], state["nonfinite_skipped"]
        print(f"resumed from step {step:,}")

    loader = make_loader(
        train_ds, cfg.batch_nodes, cfg.batch_edges, cfg.num_workers, shuffle=True
    )
    val_loader = make_loader(
        val_ds, cfg.batch_nodes, cfg.batch_edges, cfg.num_workers, shuffle=False
    )

    if cfg.compile and device == "cuda":
        if isinstance(encoder.pooler, VQPooler):
            with autocast(bf16):  # initialize the codebook before compiling
                encoder(to_device(next(iter(loader)), device))
        encoder = compile_encoder(encoder)
        objective._encoder[0] = encoder

    def save(path):
        save_checkpoint(
            path,
            {
                "step": step,
                "encoder": unwrap(encoder).state_dict(),
                "objective": objective.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val": best_val,
                "best_encoder_state": best_state,
                "graphs_seen": graphs_seen,
                "nonfinite_skipped": nonfinite,
            },
        )

    seed_everything(seed)
    encoder.train()
    objective.train()
    progress = tqdm(
        total=cfg.steps, initial=step, desc=f"pretrain {model.name}", smoothing=0
    )
    watchdog = ProgressWatchdog(STALL_TIMEOUT_S, f"pretrain {model.name}")
    with watchdog:
        while step < cfg.steps:
            for batch in loader:
                watchdog.tick()
                if step >= cfg.steps:
                    break
                batch = to_device(batch, device)
                with autocast(bf16):
                    loss, logs = objective(batch)
                graphs_seen += num_graphs(batch)

                optimizer.zero_grad()
                if not torch.isfinite(loss):
                    nonfinite += 1
                else:
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        all_params, cfg.grad_clip
                    )
                    if torch.isfinite(grad_norm):
                        optimizer.step()
                    else:
                        nonfinite += 1
                        optimizer.zero_grad()
                    logs["grad_norm"] = grad_norm.item()

                if step % cfg.log_every == 0:
                    logger.log(
                        step,
                        "train",
                        loss=loss.item(),
                        batch_graphs=num_graphs(batch),
                        nonfinite_skipped=nonfinite,
                        **logs,
                    )
                if step % cfg.val_every == 0:
                    val = validate(objective, val_loader, device, bf16)
                    logger.log(step, "val", **val)
                    if val["loss"] < best_val:
                        best_val = val["loss"]
                        best_state = {
                            k: v.detach().cpu().clone()
                            for k, v in unwrap(encoder).state_dict().items()
                        }
                step += 1
                progress.update(1)
                if step % cfg.ckpt_every == 0:
                    save(ckpt_path)
                    save_encoder(encoder, run_dir / f"encoder_step{step:07d}.pt")
    progress.close()
    save(ckpt_path)
    if not (run_dir / f"encoder_step{step:07d}.pt").exists():
        save_encoder(encoder, run_dir / f"encoder_step{step:07d}.pt")

    if best_state is not None:
        unwrap(encoder).load_state_dict(best_state)
    save_encoder(encoder, run_dir / "encoder.pt")
    logger.finish(
        {
            "steps": step,
            "best_val_loss": best_val,
            "graphs_seen": graphs_seen,
            "nonfinite_skipped": nonfinite,
            "wall_s": time.time() - t0,
        }
    )


@torch.no_grad()
def validate(objective, loader, device, bf16) -> dict:
    encoder = objective.encoder
    encoder.eval()
    objective.eval()
    sums, n = {}, 0
    for batch in loader:
        batch = to_device(batch, device)
        with autocast(bf16):
            loss, logs = objective(batch)
        for k, v in {"loss": loss.item(), **logs}.items():
            sums[k] = sums.get(k, 0.0) + float(v)
        n += 1
    encoder.train()
    objective.train()
    return {k: v / max(n, 1) for k, v in sums.items()}
