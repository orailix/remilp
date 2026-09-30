"""Optuna search over pretraining configurations, scored on frozen cells.

A sweep file defines MODEL, FIXED, SPACE ({"section.field": ("float", lo, hi, "log")
| ("int", lo, hi) | ("categorical", [...])}), TASKS, SEEDS, STEP, EVAL, SEED_PENALTY,
CELL_PENALTY, N_TRIALS, N_STARTUP_TRIALS, MAX_CONCURRENT, PRETRAIN_CONCURRENT,
SAMPLER_SEED. With r_{s,c} the best validation metric over the random encoder's, a
trial minimises mean_s G_s + SEED_PENALTY std_s G_s + CELL_PENALTY mean max(r - 1, 0),
G_s the geometric mean of r_{s,c} over cells."""

import math
import signal
import statistics
from pathlib import Path

from remilp.config import (
    MODELS,
    RUNS_ROOT,
    EvalConfig,
    PretrainConfig,
    apply_overrides,
)
from remilp.launch import (
    Job,
    auto_concurrent,
    eval_job,
    load_module,
    overrides_from,
    pretrain_job,
    run_job,
    run_jobs,
    split_overrides,
)
from remilp.train.common import read_json


def load_sweep(path: str | Path) -> dict:
    ns = load_module(path)
    return {
        "name": Path(path).stem,
        "model": ns["MODEL"],
        "fixed": ns.get("FIXED", {}),
        "space": ns["SPACE"],
        "tasks": list(ns["TASKS"]),
        "seeds": list(ns.get("SEEDS", [0])),
        "step": ns.get("STEP"),
        "eval": ns.get("EVAL", {}),
        "n_trials": int(ns.get("N_TRIALS", 40)),
        "n_startup_trials": int(ns.get("N_STARTUP_TRIALS", 10)),
        "pretrain_concurrent": int(ns.get("PRETRAIN_CONCURRENT", 1)),
        "seed_penalty": float(ns.get("SEED_PENALTY", 0.0)),
        "cell_penalty": float(ns.get("CELL_PENALTY", 0.0)),
        "max_concurrent": int(ns.get("MAX_CONCURRENT", 4)),
        "sampler_seed": int(ns.get("SAMPLER_SEED", 0)),
    }


CELL_VRAM_GB, CELL_RAM_GB = 7.0, 10.0
PRETRAIN_VRAM_GB, PRETRAIN_RAM_GB = 30.0, 20.0


def size_to_machine(sweep: dict) -> dict:
    workers = int(
        sweep["fixed"].get("pretrain.num_workers", PretrainConfig().num_workers)
    )
    cell_workers = int(sweep["eval"].get("eval.num_workers", EvalConfig().num_workers))
    return {
        **sweep,
        "max_concurrent": auto_concurrent(CELL_VRAM_GB, CELL_RAM_GB, cell_workers + 1),
        "pretrain_concurrent": auto_concurrent(
            PRETRAIN_VRAM_GB, PRETRAIN_RAM_GB, workers + 1
        ),
    }


def sweep_dir(sweep: dict) -> Path:
    return RUNS_ROOT / "sweeps" / sweep["name"]


def suggest(trial, space: dict) -> dict:
    params = {}
    for key, spec in space.items():
        kind, *rest = spec
        if kind == "float":
            lo, hi, *flags = rest
            params[key] = trial.suggest_float(key, lo, hi, log="log" in flags)
        elif kind == "int":
            lo, hi, *flags = rest
            params[key] = trial.suggest_int(key, lo, hi, log="log" in flags)
        elif kind == "categorical":
            params[key] = trial.suggest_categorical(key, list(rest[0]))
        else:
            raise ValueError(f"unknown distribution '{kind}' for {key}")
    return params


def geomean(ratios: list[float]) -> float:
    return math.exp(sum(math.log(r) for r in ratios) / len(ratios))


def trial_jobs(sweep: dict, params: dict, label: str) -> tuple[list[Job], list[Job]]:
    model_over, pre_over, eval_over = split_overrides(
        overrides_from(sweep["fixed"], sweep["eval"], params)
    )
    base = MODELS[sweep["model"]]
    model, pcfg = apply_overrides(base, PretrainConfig(), model_over + pre_over)
    step = sweep["step"]
    pretrain_jobs, eval_jobs = [], []
    for seed in sweep["seeds"]:
        if model.name == "random":
            prun = None
        else:
            job = pretrain_job(model, pcfg, seed, model_over + pre_over)
            prun = job.run_dir
            pretrain_jobs.append(job)
        for task in sweep["tasks"]:
            run_dir = (
                RUNS_ROOT
                / "eval"
                / sweep["name"]
                / label
                / "frozen"
                / task
                / f"seed{seed}"
            )
            eval_jobs.append(
                eval_job(
                    base,
                    model,
                    run_dir,
                    task,
                    True,
                    seed,
                    prun,
                    step,
                    model_over + eval_over,
                )
            )
    return pretrain_jobs, eval_jobs


def run_all(jobs: list[Job], workers: int, runner=run_job) -> None:
    failed = run_jobs(jobs, workers, runner)
    if failed:
        raise RuntimeError(f"{failed[0].run_dir} failed (see log.txt)")


def metric_of(run_dir: Path) -> float:
    """The head's best validation value."""
    return read_json(run_dir / "results.json")["fit_best_val"]


def baseline_jobs(sweep: dict) -> list[Job]:
    random_sweep = {
        **sweep,
        "model": "random",
        "fixed": {},
        "step": None,
    }
    return trial_jobs(random_sweep, {}, "random")[1]


def ensure_baseline(sweep: dict, workers: int, runner=run_job) -> dict:
    jobs = baseline_jobs(sweep)
    run_all(jobs, workers, runner)
    out = {}
    for j in jobs:
        task, seed = j.run_dir.parts[-2], int(j.run_dir.parts[-1][4:])
        out[(task, seed)] = metric_of(j.run_dir)
    return out


def evaluate_params(
    sweep: dict, params: dict, label: str, baseline: dict, runner=run_job
) -> tuple[float, dict]:
    pretrain_jobs, eval_jobs = trial_jobs(sweep, params, label)
    run_all(pretrain_jobs, sweep["pretrain_concurrent"], runner)
    run_all(eval_jobs, sweep["max_concurrent"], runner)
    details, by_seed = {}, {}
    for j in eval_jobs:
        task, seed = j.run_dir.parts[-2], int(j.run_dir.parts[-1][4:])
        value = metric_of(j.run_dir)
        ratio = value / baseline[(task, seed)]
        details[f"{task}/seed{seed}"] = value
        details[f"{task}/seed{seed}/ratio"] = ratio
        by_seed.setdefault(seed, []).append(ratio)
    if pretrain_jobs:
        details["pretrain_run"] = str(pretrain_jobs[0].run_dir.parent)
    per_seed = {s: geomean(r) for s, r in by_seed.items()}
    for s, g in per_seed.items():
        details[f"geomean/seed{s}"] = g
    mean = sum(per_seed.values()) / len(per_seed)
    spread = statistics.stdev(per_seed.values()) if len(per_seed) > 1 else 0.0
    details["geomean_mean"], details["geomean_stdev"] = mean, spread
    harm = [sum(max(r - 1.0, 0.0) for r in rs) / len(rs) for rs in by_seed.values()]
    harm = sum(harm) / len(harm)
    details["cell_harm"] = harm
    return (
        mean + sweep["seed_penalty"] * spread + sweep["cell_penalty"] * harm,
        details,
    )


def load_study(sweep: dict, sampler_seed: int | None = None):
    import optuna
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend

    d = sweep_dir(sweep)
    d.mkdir(parents=True, exist_ok=True)
    storage = JournalStorage(JournalFileBackend(str(d / "journal.log")))
    sampler = optuna.samplers.TPESampler(
        seed=sweep["sampler_seed"] if sampler_seed is None else sampler_seed,
        n_startup_trials=sweep["n_startup_trials"],
        multivariate=True,
        constant_liar=True,
    )
    return optuna.create_study(
        study_name=sweep["name"],
        storage=storage,
        sampler=sampler,
        direction="minimize",
        load_if_exists=True,
    )


def run_sweep(sweep: dict, n_trials: int, worker: int = 0, runner=run_job) -> None:
    baseline = ensure_baseline(sweep, sweep["max_concurrent"], runner)
    study = load_study(sweep, sweep["sampler_seed"] + worker)

    def objective(trial):
        params = suggest(trial, sweep["space"])
        value, details = evaluate_params(
            sweep, params, f"trial{trial.number:04d}", baseline, runner
        )
        for k, v in details.items():
            trial.set_user_attr(k, v)
        return value

    study.optimize(objective, n_trials=n_trials, catch=(RuntimeError,))


def completed(sweep: dict) -> int:
    import optuna

    trials = load_study(sweep).trials
    return sum(t.state == optuna.trial.TrialState.COMPLETE for t in trials)


def status(sweep: dict) -> str:
    import optuna

    study = load_study(sweep)
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    lines = [
        f"{sweep['name']}: {len(study.trials)} trials, {len(trials)} complete, storage {sweep_dir(sweep) / 'journal.log'}"
    ]
    keys = list(sweep["space"])
    lines.append(
        f"{'trial':6s} {'objective':10s} "
        + " ".join(f"{k.split('.')[-1]:>16s}" for k in keys)
    )
    for t in sorted(trials, key=lambda t: t.value):
        lines.append(
            f"{t.number:6d} {t.value:10.4f} "
            + " ".join(f"{t.params.get(k, '-')!s:>16s}" for k in keys)
        )
    if trials:
        best = study.best_trial
        lines.append(
            f"best: trial {best.number} objective {best.value:.4f} params {best.params}"
        )
        cells = {k: v for k, v in best.user_attrs.items() if k.endswith("/ratio")}
        lines.append(
            "  ratios vs random: "
            + ", ".join(f"{k[:-6]} {v:.3f}" for k, v in cells.items())
        )
    return "\n".join(lines)


def dry_run(sweep: dict) -> str:
    lines = [
        f"sweep {sweep['name']}: model {sweep['model']}, fixed {sweep['fixed']}, eval {sweep['eval']}, step {sweep['step']}"
    ]
    lines.append(f"space: {sweep['space']}")
    lines.append(f"objective tasks {sweep['tasks']}, seeds {sweep['seeds']}")
    jobs = baseline_jobs(sweep)
    counts = {}
    for j in jobs:
        counts[j.status] = counts.get(j.status, 0) + 1
    lines.append(f"baseline (random) cells: {len(jobs)} {counts}")
    lines.append(
        f"in parallel: cells {sweep['max_concurrent']}, "
        f"pretraining {sweep['pretrain_concurrent']}"
    )
    lines.append(
        f"{sweep['n_trials']} trials ({sweep['n_startup_trials']} random), "
        f"storage {sweep_dir(sweep) / 'journal.log'}"
    )
    return "\n".join(lines)


def main(args) -> int:
    sweep = load_sweep(args.sweep)
    if args.max_concurrent == "auto":
        sweep = size_to_machine(sweep)
    elif args.max_concurrent:
        n = int(args.max_concurrent)
        sweep["max_concurrent"] = n
        sweep["pretrain_concurrent"] = min(sweep["pretrain_concurrent"], n)
    if args.dry_run:
        print(dry_run(sweep))
        return 0
    if args.status:
        print(status(sweep))
        return 0 if completed(sweep) else 1
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    try:
        if args.baseline:
            ensure_baseline(sweep, sweep["max_concurrent"])
            print("baseline cells done")
            return 0
        n = args.trials if args.trials is not None else sweep["n_trials"]
        run_sweep(sweep, n, worker=args.worker)
    except KeyboardInterrupt:
        return 130
    print(status(sweep))
    return 0 if completed(sweep) else 1
