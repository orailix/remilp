"""The `remilp` command."""

import argparse
import sys
from pathlib import Path

from remilp.config import MODELS


def main(argv=None):
    parser = argparse.ArgumentParser(prog="remilp")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("pretrain", help="pretrain one encoder")
    p.add_argument(
        "--model", required=True, choices=[m for m in MODELS if m != "random"]
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--set", action="append", default=[], metavar="SECTION.FIELD=VALUE")
    p.add_argument(
        "--run-dir", default=None, help="default runs/pretrain/<model_tag>/seed<seed>"
    )
    p.add_argument("--force", action="store_true")

    p = sub.add_parser(
        "evaluate",
        help="train a downstream head on one task, encoder frozen or fine-tuned",
    )
    p.add_argument("--model", required=True, choices=list(MODELS))
    p.add_argument("--task", required=True)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--frozen", dest="frozen", action="store_true", default=True)
    mode.add_argument("--finetune", dest="frozen", action="store_false")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--set", action="append", default=[], metavar="SECTION.FIELD=VALUE")
    p.add_argument(
        "--pretrain-run",
        default=None,
        help="default runs/pretrain/<model_tag>/seed<seed>",
    )
    p.add_argument(
        "--step",
        type=int,
        default=None,
        help="load encoder_step<N>.pt instead of encoder.pt",
    )
    p.add_argument(
        "--experiment", default="adhoc", help="name of the runs/eval/<experiment>/ tree"
    )
    p.add_argument("--run-dir", default=None)
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("launch", help="run every job of an experiment file")
    p.add_argument("experiment")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--local", type=int, metavar="N", help="run locally with N parallel jobs"
    )
    p.add_argument(
        "--gpus",
        type=lambda s: s.split(","),
        default=None,
        help="comma-separated GPU ids for --local",
    )
    p.add_argument(
        "--claim",
        type=int,
        metavar="N",
        help="run N jobs at a time as one of any number of workers sharing the experiment",
    )
    p.add_argument(
        "--slurm",
        action="store_true",
        help="submit one array per phase via slurm/array.sh",
    )
    p.add_argument("--max-concurrent", type=int, default=None)
    p.add_argument(
        "--sbatch", default="", help="extra sbatch options, e.g. '-A <account>'"
    )
    p.add_argument("--pretrain-only", action="store_true")
    p.add_argument(
        "--count",
        choices=["pretrain", "eval"],
        help="print the number of jobs of a phase",
    )
    p.add_argument(
        "--job", type=int, help="run one job by index (used by slurm/array.sh)"
    )
    p.add_argument("--phase", choices=["pretrain", "eval"], default="eval")

    p = sub.add_parser("sweep", help="run Optuna trials of a sweep file")
    p.add_argument("sweep")
    p.add_argument(
        "--trials",
        type=int,
        default=None,
        help="trials for this worker (default N_TRIALS)",
    )
    p.add_argument(
        "--worker", type=int, default=0, help="worker index (offsets the sampler seed)"
    )
    p.add_argument(
        "--baseline", action="store_true", help="only compute the random-encoder cells"
    )
    p.add_argument(
        "--max-concurrent",
        metavar="N|auto",
        default=None,
        help="cells in parallel within a trial (default MAX_CONCURRENT); "
        "auto sizes them to this machine",
    )
    p.add_argument(
        "--status",
        action="store_true",
        help="print the trials table; exits non-zero if no trial has completed",
    )
    p.add_argument("--dry-run", action="store_true")

    p = sub.add_parser("data", help="inspect or build datasets")
    ds = p.add_subparsers(dest="data_command", required=True)
    ds.add_parser("check", help="list every source with its splits and graphs")
    b = ds.add_parser("build", help="build the graphs of a source from its raw inputs")
    b.add_argument("--source", required=True, choices=["milp_evolve", "dmiplib"])
    b.add_argument("--cls")
    b.add_argument("--difficulty")
    b.add_argument("--num-processes", type=int, default=12)
    b.add_argument("--confirm-reprocess", action="store_true")
    u = ds.add_parser("unpack", help="extract every raw.tar in place")
    u.add_argument("root", nargs="?", default="data")

    args = parser.parse_args(argv)
    if args.command == "pretrain":
        return _pretrain(args)
    if args.command == "evaluate":
        return _evaluate(args)
    if args.command == "launch":
        from remilp.launch import main as launch_main

        return launch_main(args)
    if args.command == "sweep":
        from remilp.sweep import main as sweep_main

        return sweep_main(args)
    if args.command == "data":
        return _data(args)


def _pretrain(args):
    from remilp.config import MODELS, PretrainConfig, apply_overrides
    from remilp.train.common import tee_output
    from remilp.train.pretrain import pretrain_dir, run_pretrain

    model, cfg = apply_overrides(MODELS[args.model], PretrainConfig(), args.set)
    run_dir = Path(args.run_dir or pretrain_dir(model, cfg, args.seed))
    with tee_output(run_dir):
        run_pretrain(model, cfg, args.seed, run_dir, force=args.force)
    return 0


def _evaluate(args):
    from remilp.config import MODELS, RUNS_ROOT, EvalConfig, apply_overrides, model_tag
    from remilp.train.common import tee_output
    from remilp.train.evaluate import run_eval

    model, cfg = apply_overrides(
        MODELS[args.model], EvalConfig(task=args.task, frozen=args.frozen), args.set
    )
    tag = model_tag(model)
    pretrain_run = None
    if model.name != "random":
        pretrain_run = Path(
            args.pretrain_run or RUNS_ROOT / "pretrain" / tag / f"seed{args.seed}"
        )
    mode = "frozen" if cfg.frozen else "finetune"
    run_dir = Path(
        args.run_dir
        or RUNS_ROOT
        / "eval"
        / args.experiment
        / tag
        / mode
        / cfg.task
        / f"seed{args.seed}"
    )
    with tee_output(run_dir):
        run_eval(
            model,
            cfg,
            args.seed,
            run_dir,
            pretrain_run,
            args.step,
            force=args.force,
        )
    return 0


def _data(args):
    from remilp.config import DATA_ROOT
    from remilp.data import sources
    from remilp.data.dataset import MissingCacheError, graph_path, load_manifest
    from remilp.data.package import RAW_ARCHIVE, unpack

    if args.data_command == "check":
        ok = True
        for root in sources.all_source_roots():
            try:
                entries = load_manifest(root)["entries"]
            except MissingCacheError as e:
                print(f"{root}: {e}")
                ok = False
                continue
            for split in sources.source_splits(root):
                members = [e for e in entries if e["split"] == split]
                present = sum(graph_path(root, e).exists() for e in members)
                raw = "raw" if (root / split / RAW_ARCHIVE).exists() else "   "
                if (root / split / "instances").is_dir():
                    raw = "raw+"
                state = "ok" if present == len(members) else "INCOMPLETE"
                ok &= state == "ok"
                print(
                    f"{str(root):26s} {split:14s} entries {len(members):6d}"
                    f"  graphs {present:6d}  {raw:4s} {state}"
                )
        return 0 if ok else 1
    if args.data_command == "build":
        from remilp.data import build
        from remilp.data.dataset import manifest_path

        root = sources.source_root(args.source, args.cls, args.difficulty)
        if not args.confirm_reprocess:
            print(
                f"Would build {root}, overwriting nothing but missing graphs; "
                "re-run with --confirm-reprocess."
            )
            return 1
        if manifest_path(root).exists():
            manifest = load_manifest(root)
            print(f"using the existing manifest ({len(manifest['entries'])} entries)")
        else:
            source = root.relative_to(DATA_ROOT).as_posix()
            manifest = build.build_manifest(root, source, args.num_processes)
        build.process_all(root, manifest, args.num_processes)
        return 0
    if args.data_command == "unpack":
        for archive in unpack(Path(args.root)):
            print(f"extracted {archive}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
