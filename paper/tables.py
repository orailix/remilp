"""Tables of the frozen encoders: aggregates, per pair, and the gap task."""

from common import (
    GROUPS,
    OUT,
    SEEDS,
    cell,
    group_tasks,
    mean_sd,
    ratio_row,
    result,
    strict_winner,
)
from remilp.tasks import ILP_TIERS, MILP_TIERS

NODE_ARMS = [("remilp", "ReMILP"), ("forge", "FORGE")]
GAP_ARMS = [
    ("supervised", "Random"),
    ("remilp", "ReMILP"),
    ("forge", "FORGE"),
    ("forge-attn", "FORGE + attn."),
]


def frozen_table() -> str:
    lines = []
    for j, (kind, tiers, _, name) in enumerate(GROUPS):
        tasks = group_tasks(kind, tiers)
        vals = [
            ratio_row("nodes", arm, "supervised", "frozen", tasks)
            for arm, _ in NODE_ARMS
        ]
        best = strict_winner(vals)
        stub = "Binary solution" if kind == "solution" else "Constraint activity"
        head = f"\\multirow{{2}}{{*}}{{{stub}}}" if j % 2 == 0 else ""
        cols = " & ".join(cell(m, sd, i == best) for i, (m, sd) in enumerate(vals))
        lines.append(f"    {head} & {name} & $1$ & {cols} \\\\")
        if j == 1:
            lines.append("    \\cmidrule(l){2-5}")
        print(
            f"frozen {kind:9s} {name:9s} ({len(tasks):2d} pairs)  "
            + "  ".join(
                f"{lab} {m:.3f} +- {sd:.3f}"
                for (m, sd), (_, lab) in zip(vals, NODE_ARMS)
            )
        )
    gap = {lab: gap_frozen(arm) for arm, lab in GAP_ARMS}
    print(
        "frozen gap MAE  "
        + "  ".join(f"{k} {m:.3f} +- {sd:.3f}" for k, (m, sd) in gap.items())
    )
    trio = [gap["Random"], gap["ReMILP"], gap["FORGE"]]
    best = strict_winner(trio)
    g = [cell(m, sd, i == best) for i, (m, sd) in enumerate(trio)]
    return "\n".join(
        [
            "% frozen encoders, relative KL and gap MAE",
            "\\begin{tabular}{@{}llccc@{}}",
            "    \\toprule",
            "    Task & Variables & Random & ReMILP & FORGE\\textsuperscript{\\textdagger} \\\\",
            "    \\midrule",
            *lines,
            "    \\midrule",
            f"    Integrality gap & MILP & {g[0]} & {g[1]} & {g[2]} \\\\",
            "    \\bottomrule",
            "\\end{tabular}",
        ]
    )


def gap_frozen(arm):
    return mean_sd(result("gap", arm, "frozen", "gap", s, "mae") for s in SEEDS)


def family_table(kind) -> str:
    arms = ("supervised", "remilp", "forge")
    lines = []
    for block in (ILP_TIERS, MILP_TIERS):
        seen = None
        for cls, diff in block:
            task = f"{kind}-{cls}-{diff}"
            vals = [
                mean_sd(result("nodes", a, "frozen", task, s, "kl") for s in SEEDS)
                for a in arms
            ]
            best = strict_winner(vals)
            cells = [cell(m, sd, i == best) for i, (m, sd) in enumerate(vals)]
            lines.append(
                f"    {cls if cls != seen else ''} & {diff} & "
                + " & ".join(cells)
                + " \\\\"
            )
            seen = cls
        if block is ILP_TIERS:
            lines.append("    \\midrule")
    return "\n".join(
        [
            f"% frozen encoders, {kind} KL per pair",
            "\\begin{tabular}{@{}ll ccc@{}}",
            "    \\toprule",
            "    Class & Difficulty & Random & ReMILP & FORGE\\textsuperscript{\\textdagger} \\\\",
            "    \\midrule",
            *lines,
            "    \\bottomrule",
            "\\end{tabular}",
        ]
    )


def gap_table() -> str:
    labels = {
        "supervised": "Random",
        "remilp": "ReMILP",
        "forge": "FORGE\\textsuperscript{\\textdagger}",
        "forge-attn": "FORGE\\textsuperscript{\\textdagger} + attn.",
    }
    lines = [
        f"    {labels[arm]} & {cell(*gap_frozen(arm))} \\\\" for arm, _ in GAP_ARMS
    ]
    return "\n".join(
        [
            "% frozen encoders, gap MAE",
            "\\begin{tabular}{@{}lc@{}}",
            "    \\toprule",
            "    Encoder & Test MAE \\\\",
            "    \\midrule",
            *lines,
            "    \\bottomrule",
            "\\end{tabular}",
        ]
    )


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    body = "\n\n".join(
        [
            frozen_table(),
            family_table("solution"),
            family_table("activity"),
            gap_table(),
        ]
    )
    (OUT / "tables.tex").write_text(body + "\n")
    print(f"wrote {OUT / 'tables.tex'}")
