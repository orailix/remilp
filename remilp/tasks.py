"""Tasks: `gap`, `solution-<CLASS>-<difficulty>`, `activity-<CLASS>-<difficulty>`."""

from dataclasses import dataclass

# Distributional MIPLIB (class, difficulty) pairs.
ILP_TIERS = [
    ("IS", "easy"),
    ("IS", "medium"),
    ("CA", "easy"),
    ("CA", "medium"),
    ("SC", "easy"),
    ("SC", "medium"),
    ("SC", "hard"),
    ("VC", "easy"),
    ("VC", "medium"),
    ("VC", "hard"),
    ("GISP", "easy"),
    ("GISP", "medium"),
    ("GISP", "hard"),
    ("MMCN", "medium-BI"),
    ("MMCN", "hard-BI"),
]
# Pairs with continuous variables.
MILP_TIERS = [
    ("CFLP", "easy"),
    ("CFLP", "medium"),
    ("OTS", "easy"),
    ("OTS", "medium"),
    ("OTS", "hard"),
    ("MMCN", "medium-BC"),
    ("NNV", "easy"),
    ("LB", "hard"),
]
TIERS = ILP_TIERS + MILP_TIERS


@dataclass(frozen=True)
class Task:
    name: str
    kind: str  # "gap" | "solution" | "activity"
    source: str  # "milp_evolve" | "dmiplib"
    cls: str | None = None
    difficulty: str | None = None

    @property
    def metric(self) -> str:
        return "mae" if self.kind == "gap" else "kl"


def parse_task(name: str) -> Task:
    if name == "gap":
        return Task(name, "gap", "milp_evolve")
    kind, _, rest = name.partition("-")
    cls, _, difficulty = rest.partition("-")
    if kind in ("solution", "activity") and (cls, difficulty) in TIERS:
        return Task(name, kind, "dmiplib", cls, difficulty)
    raise ValueError(f"unknown task '{name}'")


def node_tasks(tiers=TIERS) -> list[str]:
    return [
        f"{kind}-{cls}-{diff}"
        for kind in ("solution", "activity")
        for cls, diff in tiers
    ]


NODE_TASKS = node_tasks()
