"""Fine-tuning curves: aggregates, the ablation and per pair."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import GROUPS, OUT, PAIRS, SEEDS, group_tasks, run_dir, selected_test

SUPERVISED = ("supervised", "Supervised", "0.35", "--")
REMILP = ("remilp", "ReMILP", "#1f77b4", "-")
FORGE = ("forge", "FORGE$^\\dagger$", "#d62728", "-.")
FORGE_ATTN = ("forge-attn", "FORGE$^\\dagger$ + attn.", "#ff7f0e", ":")
NOSUB = ("remilp-nosub", "ReMILP w/o substitutions", "#2ca02c", "-.")
ROW_SIZE = (1.32, 1.05)
WARMUP = 2_000  # y-limits ignore earlier steps

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 8,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
    }
)


def curve(experiment, arm, task, seed, field, grid):
    path = run_dir(experiment, arm, "finetune", task, seed) / "metrics.jsonl"
    return selected_test(path, field, grid)


def node_band(experiment, arm, tasks, grid):
    per_seed = []
    for seed in SEEDS:
        ratios = np.vstack(
            [
                curve(experiment, arm, t, seed, "kl", grid)
                / curve(experiment, SUPERVISED[0], t, seed, "kl", grid)[-1]
                for t in tasks
            ]
        )
        per_seed.append(np.exp(np.log(ratios).mean(axis=0)))
    per_seed = np.vstack(per_seed)
    return per_seed.mean(axis=0), per_seed.std(axis=0, ddof=1)


def gap_band(experiment, arm, grid):
    stack = np.vstack([curve(experiment, arm, "gap", s, "mae", grid) for s in SEEDS])
    return stack.mean(axis=0), stack.std(axis=0, ddof=1)


def reach(grid, m, target):
    """First step at which the mean curve is at or below `target`, or None."""
    hit = np.nonzero(m <= target)[0]
    return int(grid[hit[0]]) if hit.size else None


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def node_panels(experiment, arms, steps, prefix, summary):
    grid = np.arange(100, steps + 1, 100)
    for kind, tiers, name, var in GROUPS:
        tasks = group_tasks(kind, tiers)
        fig, ax = plt.subplots(figsize=ROW_SIZE)
        lo, hi = 1.0, 1.0
        for arm, label, color, ls in arms:
            m, sd = node_band(experiment, arm, tasks, grid)
            after = grid >= WARMUP
            lo, hi = min(lo, (m - sd)[after].min()), max(hi, (m + sd)[after].max())
            ax.plot(
                grid / 1000, m, color=color, linestyle=ls, linewidth=1.1, label=label
            )
            ax.fill_between(
                grid / 1000, m - sd, m + sd, color=color, alpha=0.15, linewidth=0
            )
            summary[f"{prefix} {kind}/{var}/{arm}"] = (
                m[-1],
                sd[-1],
                reach(grid, m, 1.0),
            )
        ax.axhline(1.0, color="0.7", linewidth=0.5, zorder=0)
        ax.set_xlim(0, steps / 1000)
        pad = 0.08 * (hi - lo)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlabel("fine-tuning steps ($\\times 10^3$)")
        ax.set_ylabel("relative test KL")
        ax.locator_params(axis="y", nbins=4)
        fig.tight_layout(pad=0.2)
        save(fig, f"{prefix}_{name}")
    handles, labels = ax.get_legend_handles_labels()
    strip = plt.figure(figsize=(5.5, 0.25))
    strip.legend(handles, labels, ncol=len(arms), frameon=False, loc="center")
    save(strip, f"{prefix}_legend")


def gap_panel(curves, name, summary):
    grid = np.arange(100, 100_001, 100)
    fig, ax = plt.subplots(figsize=(3.2, 2.4))
    target = None
    for experiment, (arm, label, color, ls) in curves:
        m, sd = gap_band(experiment, arm, grid)
        target = m[-1] if target is None else target
        ax.plot(grid / 1000, m, color=color, linestyle=ls, linewidth=1.1, label=label)
        ax.fill_between(
            grid / 1000, m - sd, m + sd, color=color, alpha=0.15, linewidth=0
        )
        summary[f"{name} {experiment}/{arm}"] = (m[-1], sd[-1], reach(grid, m, target))
    ax.set_xlim(0, 100)
    ax.set_ylim(0.03, 0.10)
    ax.set_xlabel("fine-tuning steps ($\\times 10^3$)")
    ax.set_ylabel("test MAE")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)


def pair_panels(kind, arms, steps=50_000):
    grid = np.arange(100, steps + 1, 100)
    for index, pair in enumerate(PAIRS):
        fig, ax = plt.subplots(figsize=(1.75, 1.35))
        lo, hi = np.inf, -np.inf
        for arm, label, color, ls in arms:
            stack = np.vstack(
                [curve("nodes", arm, f"{kind}-{pair}", s, "kl", grid) for s in SEEDS]
            )
            m, sd = stack.mean(axis=0), stack.std(axis=0, ddof=1)
            ax.plot(
                grid / 1000, m, color=color, linestyle=ls, linewidth=0.9, label=label
            )
            ax.fill_between(
                grid / 1000, m - sd, m + sd, color=color, alpha=0.15, linewidth=0
            )
            after = grid >= WARMUP
            lo, hi = min(lo, (m - sd)[after].min()), max(hi, (m + sd)[after].max())
        level = 0.5 * (hi + lo)
        if hi - lo < 0.01 * level:
            lo, hi = level * 0.995, level * 1.005
        pad = 0.08 * (hi - lo) or 1e-6
        ax.set_ylim(lo - pad, hi + pad)
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.set_xlim(0, steps / 1000)
        ax.tick_params(length=2)
        ax.set_ylabel("test KL")
        ax.set_xlabel("steps ($\\times 10^3$)")
        if index == 0:
            ax.legend(frameon=False, loc="upper right", fontsize=6.5)
        fig.tight_layout(pad=0.2)
        save(fig, f"fig_app_{kind}_{pair.replace('-', '_')}")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    summary = {}
    node_panels("nodes", [SUPERVISED, REMILP, FORGE], 50_000, "fig_q3", summary)
    gap_panel(
        [("gap", SUPERVISED), ("gap", REMILP), ("gap", FORGE), ("gap", FORGE_ATTN)],
        "fig_q3_gap",
        summary,
    )
    node_panels("ablation", [SUPERVISED, REMILP, NOSUB], 20_000, "fig_abl", summary)
    gap_panel(
        [
            ("gap", SUPERVISED),
            ("gap", REMILP),
            ("ablation", ("remilp-nosub-gap",) + NOSUB[1:]),
        ],
        "fig_abl_gap",
        summary,
    )
    for kind in ("solution", "activity"):
        pair_panels(kind, [SUPERVISED, REMILP, FORGE])
    for k, (m, sd, step) in summary.items():
        print(f"{k:50s} final {m:.3f} +- {sd:.3f}   reaches supervised final at {step}")
    json.dump(
        {
            k: {"final": m, "sd": sd, "reaches_supervised": s}
            for k, (m, sd, s) in summary.items()
        },
        (OUT / "figures_summary.json").open("w"),
        indent=1,
        default=float,
    )
    print(f"wrote {OUT}/")
