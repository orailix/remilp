"""Expand an experiment file into pretraining and evaluation jobs, and run them.

ARMS = {"remilp": {"model": "remilp", "step": 20000},
        "other": {"model": "forge", "set": {...}, "pretrain_run": "runs/pretrain/x",
                  "tasks": ["gap"]}}
MODES, TASKS, SEEDS, PRETRAIN, EVAL, MAX_CONCURRENT"""

import os
import runpy
import signal
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from remilp.config import (
    MODELS,
    RUNS_ROOT,
    EvalConfig,
    PretrainConfig,
    apply_overrides,
)
from remilp.train.common import STALL_TIMEOUT_S, read_json
from remilp.train.evaluate import eval_hash
from remilp.train.pretrain import pretrain_dir, pretrain_hash


@dataclass
class Job:
    kind: str  # "pretrain" | "eval"
    run_dir: Path
    argv: list[str] = field(default_factory=list)
    expected_hash: str | None = None

    @property
    def status(self) -> str:
        """pending | running | done | failed | stale (another configuration)."""
        config = self.run_dir / "config.json"
        if config.exists() and self.expected_hash is not None:
            recorded = read_json(config).get(f"{self.kind}_hash")
            if recorded != self.expected_hash:
                return "stale"
        if (self.run_dir / "results.json").exists():
            return "done"
        if (self.run_dir / "status.json").exists():
            return read_json(self.run_dir / "status.json")["status"]
        return "pending"

    @property
    def updated(self) -> float:
        path = self.run_dir / "status.json"
        return read_json(path).get("updated", 0.0) if path.exists() else 0.0


def load_module(path: str | Path) -> dict:
    repo = str(Path(path).resolve().parent.parent)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    return runpy.run_path(str(path))


def load_experiment(path: str | Path) -> dict:
    ns = load_module(path)
    return {
        "name": Path(path).stem,
        "arms": ns["ARMS"],
        "modes": ns.get("MODES", ["frozen"]),
        "tasks": ns.get("TASKS"),
        "seeds": ns.get("SEEDS", [0]),
        "pretrain": ns.get("PRETRAIN", {}),
        "eval": ns.get("EVAL", {}),
        "max_concurrent": ns.get("MAX_CONCURRENT", 8),
    }


def overrides_from(*dicts) -> list[str]:
    merged = {}
    for d in dicts:
        merged.update(d)
    return [
        f"{k}={str(v).lower() if isinstance(v, bool) else v}" for k, v in merged.items()
    ]


def split_overrides(overrides: list[str]) -> tuple[list[str], list[str], list[str]]:
    """(model, pretrain, eval) overrides."""
    model = [o for o in overrides if not o.startswith(("pretrain.", "eval."))]
    return (
        model,
        [o for o in overrides if o.startswith("pretrain.")],
        [o for o in overrides if o.startswith("eval.")],
    )


def pretrain_job(model, pcfg, seed: int, overrides: list[str]) -> Job:
    run_dir = pretrain_dir(model, pcfg, seed)
    argv = [
        "pretrain",
        "--model",
        model.name,
        "--seed",
        str(seed),
        "--run-dir",
        str(run_dir),
    ]
    for o in overrides:
        argv += ["--set", o]
    return Job("pretrain", run_dir, argv, pretrain_hash(model, pcfg, seed))


def eval_job(
    base_model,
    model,
    run_dir: Path,
    task: str,
    frozen: bool,
    seed: int,
    pretrain_run: Path | None,
    step: int | None,
    overrides: list[str],
) -> Job:
    step = step if pretrain_run is not None else None
    argv = [
        "evaluate",
        "--model",
        model.name,
        "--task",
        task,
        "--frozen" if frozen else "--finetune",
        "--seed",
        str(seed),
        "--run-dir",
        str(run_dir),
    ]
    if pretrain_run is not None:
        argv += ["--pretrain-run", str(pretrain_run)]
    if step is not None:
        argv += ["--step", str(step)]
    for o in overrides:
        argv += ["--set", o]
    ecfg = apply_overrides(base_model, EvalConfig(task=task, frozen=frozen), overrides)[
        1
    ]
    return Job("eval", run_dir, argv, eval_hash(model, ecfg, seed, pretrain_run, step))


def auto_concurrent(vram_gb: float, ram_gb: float, cpus_per_job: int = 1) -> int:
    """Jobs of that size that fit this machine's GPU memory, host memory and CPUs."""
    import torch

    bounds = [len(os.sched_getaffinity(0)) // max(1, cpus_per_job)]
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory
        bounds.append(int(vram / (vram_gb * 1024**3)))
    bounds.append(int(_ram_budget() / (ram_gb * 1024**3)))
    return max(1, min(bounds))


def _ram_budget() -> float:
    """Available host memory per task of this node, in bytes."""
    with open("/proc/meminfo") as f:
        available = next(l for l in f if l.startswith("MemAvailable")).split()[1]
    tasks = int(os.environ.get("SLURM_NTASKS_PER_NODE", 1))
    return float(available) * 1024 / tasks


def expand(exp: dict) -> tuple[list[Job], list[Job]]:
    if exp["tasks"] is None:
        from remilp.tasks import NODE_TASKS

        exp["tasks"] = NODE_TASKS
    pretrain_jobs, eval_jobs, seen = [], [], set()
    for arm, spec in exp["arms"].items():
        model_over, pre_over, eval_over = split_overrides(
            overrides_from(exp["pretrain"], exp["eval"], spec.get("set", {}))
        )
        base_model = MODELS[spec["model"]]
        model, pcfg = apply_overrides(
            base_model, PretrainConfig(), model_over + pre_over
        )
        for seed in exp["seeds"]:
            if "pretrain_run" in spec:
                prun = Path(spec["pretrain_run"]) / f"seed{seed}"
            elif model.name == "random":
                prun = None
            else:
                job = pretrain_job(model, pcfg, seed, model_over + pre_over)
                prun = job.run_dir
                if job.expected_hash not in seen:
                    seen.add(job.expected_hash)
                    pretrain_jobs.append(job)
            for mode in exp["modes"]:
                for task in spec.get("tasks", exp["tasks"]):
                    run_dir = (
                        RUNS_ROOT
                        / "eval"
                        / exp["name"]
                        / arm
                        / mode
                        / task
                        / f"seed{seed}"
                    )
                    eval_jobs.append(
                        eval_job(
                            base_model,
                            model,
                            run_dir,
                            task,
                            mode == "frozen",
                            seed,
                            prun,
                            spec.get("step"),
                            model_over + eval_over,
                        )
                    )

    def order(job: Job) -> tuple:
        arm, mode, task, seed = job.run_dir.parts[-4:]
        return task, mode, seed, arm

    eval_jobs.sort(key=order)
    return pretrain_jobs, eval_jobs


def run_job(job: Job, gpu: str | None = None) -> int:
    env = dict(os.environ)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    job.run_dir.mkdir(parents=True, exist_ok=True)
    with (job.run_dir / "log.txt").open("a") as log:
        return subprocess.call(
            [sys.executable, "-m", "remilp", *job.argv],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
        )


LOCK = ".sweep_lock"
STALE_CLAIM_S = 2 * STALL_TIMEOUT_S
POLL_S = 30.0


def _owner() -> str:
    return f"{socket.gethostname()}:{os.getpid()}"


def _claim(run_dir: Path) -> bool:
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(run_dir / LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.write(fd, _owner().encode())
    os.close(fd)
    return True


def _release(run_dir: Path) -> None:
    lock = run_dir / LOCK
    try:
        if lock.read_text() == _owner():
            lock.unlink()
    except FileNotFoundError:
        pass


def _idle_s(run_dir: Path) -> float:
    last = 0.0
    for p in run_dir.iterdir():
        try:
            last = max(last, p.stat().st_mtime)
        except FileNotFoundError:
            pass
    return time.time() - last


def _failed_since(job: Job, since: float) -> bool:
    return job.status == "failed" and job.updated >= since


def _run_claimed(job: Job, runner, since: float) -> int | None:
    """The runner's exit code, or None when the job was not run."""
    if not _claim(job.run_dir):
        return None
    try:
        if job.status == "done" or _failed_since(job, since):
            return None
        return runner(job)
    finally:
        _release(job.run_dir)


def run_jobs(
    jobs: list[Job], workers: int, runner=run_job, since: float | None = None
) -> list[Job]:
    """Run the unfinished jobs, `workers` at a time, sharing them with other workers
    through lock files; returns the failed ones."""
    since = time.time() if since is None else since
    pending = [j for j in jobs if j.status != "done"]
    failed = []
    while pending:
        pool = ThreadPoolExecutor(max(1, workers))
        try:
            codes = list(pool.map(lambda j: _run_claimed(j, runner, since), pending))
        finally:
            pool.shutdown(cancel_futures=True)
        still = []
        for j, c in zip(pending, codes):
            if j.status == "done":
                continue
            (failed if c is not None or _failed_since(j, since) else still).append(j)
        pending = still
        for j in pending:
            if _idle_s(j.run_dir) > STALE_CLAIM_S:
                (j.run_dir / LOCK).unlink(missing_ok=True)
        if pending and all((j.run_dir / LOCK).exists() for j in pending):
            time.sleep(POLL_S)
    return failed


def run_claimed(
    pretrain_jobs: list[Job], eval_jobs: list[Job], workers: int, pretrain_only: bool
) -> int:
    start = time.time()
    failed = run_jobs(pretrain_jobs, workers, since=start)
    if not pretrain_only:
        failed += run_jobs(eval_jobs, workers, since=start)
    for j in failed:
        print(f"FAILED: {j.run_dir} (see log.txt)")
    print(f"{len(failed)} failed")
    return 1 if failed else 0


def run_local(jobs: list[Job], workers: int, gpus: list[str] | None) -> None:
    pending = [j for j in jobs if j.status != "done"]
    print(f"running {len(pending)} of {len(jobs)} jobs with {workers} workers")
    with ThreadPoolExecutor(workers) as pool:
        codes = list(
            pool.map(
                lambda ij: run_job(ij[1], gpus[ij[0] % len(gpus)] if gpus else None),
                enumerate(pending),
            )
        )
    failed = [j for j, c in zip(pending, codes) if c != 0]
    for j in failed:
        print(f"FAILED: {j.run_dir} (see log.txt)")
    print(f"{len(pending) - len(failed)} succeeded, {len(failed)} failed")


def submit_slurm(
    experiment: str,
    phase: str,
    n_jobs: int,
    max_concurrent: int,
    dependency: str | None,
    extra: str,
) -> str:
    Path("logs").mkdir(exist_ok=True)
    cmd = ["sbatch", "--parsable", f"--array=0-{n_jobs - 1}%{max_concurrent}"]
    if dependency:
        cmd.append(f"--dependency=afterok:{dependency}")
    cmd += extra.split() + ["slurm/array.sh", experiment, phase]
    out = subprocess.check_output(cmd, text=True).strip()
    print(f"submitted {phase} array {out}: {' '.join(cmd)}")
    return out.split(";")[0]


def main(args) -> int:
    exp = load_experiment(args.experiment)
    pretrain_jobs, eval_jobs = expand(exp)
    if args.count:
        print(len(pretrain_jobs if args.count == "pretrain" else eval_jobs))
        return 0
    if args.job is not None:
        jobs = pretrain_jobs if args.phase == "pretrain" else eval_jobs
        job = jobs[args.job]
        if job.status == "done":
            print(f"{job.run_dir}: already done")
            return 0
        return subprocess.call([sys.executable, "-m", "remilp", *job.argv])
    for kind, jobs in (("pretrain", pretrain_jobs), ("eval", eval_jobs)):
        counts = {}
        for j in jobs:
            counts[j.status] = counts.get(j.status, 0) + 1
        print(f"{kind}: {len(jobs)} jobs {counts}")
        if args.dry_run:
            for j in jobs:
                print(f"  [{j.status:8s}] {' '.join(j.argv)}")
    if args.dry_run:
        return 0
    if args.local:
        run_local(pretrain_jobs, args.local, args.gpus)
        if not args.pretrain_only:
            run_local(eval_jobs, args.local, args.gpus)
        return 0
    if args.claim:
        signal.signal(signal.SIGTERM, signal.default_int_handler)
        try:
            return run_claimed(pretrain_jobs, eval_jobs, args.claim, args.pretrain_only)
        except KeyboardInterrupt:
            return 130
    if args.slurm:
        dep = None
        pending_pre = [j for j in pretrain_jobs if j.status != "done"]
        if pending_pre:
            dep = submit_slurm(
                args.experiment,
                "pretrain",
                len(pretrain_jobs),
                args.max_concurrent or exp["max_concurrent"],
                None,
                args.sbatch,
            )
        if not args.pretrain_only and eval_jobs:
            submit_slurm(
                args.experiment,
                "eval",
                len(eval_jobs),
                args.max_concurrent or exp["max_concurrent"],
                dep,
                args.sbatch,
            )
        return 0
    print("nothing to do: pass --dry-run, --local N, --claim N or --slurm")
    return 1
