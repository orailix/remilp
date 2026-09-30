import json
import math
import statistics as st
from pathlib import Path

import numpy as np

from remilp.config import RUNS_ROOT
from remilp.tasks import ILP_TIERS, MILP_TIERS, TIERS

EVAL = RUNS_ROOT / "eval"
OUT = Path("figures")
SEEDS = (0, 1, 2)
GROUPS = [  # (kind, tiers, file suffix, row label)
    ("solution", ILP_TIERS, "solution_ilp", "pure ILP"),
    ("solution", MILP_TIERS, "solution_milp", "MILP"),
    ("activity", ILP_TIERS, "activity_ilp", "pure ILP"),
    ("activity", MILP_TIERS, "activity_milp", "MILP"),
]
PAIRS = [f"{c}-{d}" for c, d in TIERS]


def run_dir(experiment, arm, mode, task, seed) -> Path:
    return EVAL / experiment / arm / mode / task / f"seed{seed}"


def result(experiment, arm, mode, task, seed, field) -> float:
    path = run_dir(experiment, arm, mode, task, seed) / "results.json"
    return json.loads(path.read_text())[field]


def group_tasks(kind, tiers) -> list[str]:
    return [f"{kind}-{c}-{d}" for c, d in tiers]


def geomean(values) -> float:
    return math.exp(st.mean(math.log(v) for v in values))


def mean_sd(values) -> tuple[float, float]:
    values = list(values)
    return st.mean(values), st.stdev(values)


def ratio_row(experiment, arm, ref, mode, tasks) -> tuple[float, float]:
    """Mean and sd over seeds of the geometric mean over tasks of arm / ref KL."""
    return mean_sd(
        geomean(
            result(experiment, arm, mode, t, s, "kl")
            / result(experiment, ref, mode, t, s, "kl")
            for t in tasks
        )
        for s in SEEDS
    )


def strict_winner(vals, digits=3):
    """Index of the lowest mean if its interval clears the runner-up's, else None."""
    order = sorted(range(len(vals)), key=lambda i: vals[i][0])
    a, b = order[0], order[1]
    if f"{vals[a][0]:.{digits}f}" == f"{vals[b][0]:.{digits}f}":
        return None
    return a if vals[a][0] + vals[a][1] < vals[b][0] - vals[b][1] else None


def cell(m, sd, bold=False) -> str:
    body = f"{m:.3f} \\pm {sd:.3f}"
    return f"$\\mathbf{{{body}}}$" if bold else f"${body}$"


def selected_test(path: Path, field: str, grid) -> np.ndarray:
    """Test metric of the best-validation checkpoint so far, at each step of `grid`."""
    val, test = {}, {}
    for line in path.open():
        r = json.loads(line)
        if r.get("phase") == "val" and f"val_{field}" in r:
            val[r["step"]] = r[f"val_{field}"]
        elif (
            r.get("phase") == "test" and f"test_{field}" in r and "checkpoint" not in r
        ):
            test[r["step"]] = r[f"test_{field}"]
    steps = [s for s in sorted(val) if s in test]
    out, best, chosen, i = [], np.inf, test[steps[0]], 0
    for g in grid:
        while i < len(steps) and steps[i] <= g:
            if val[steps[i]] < best:
                best, chosen = val[steps[i]], test[steps[i]]
            i += 1
        out.append(chosen)
    return np.array(out)
