"""Dataset locations and splits."""

from pathlib import Path

from remilp.config import DATA_ROOT
from remilp.data.dataset import GraphDataset
from remilp.tasks import TIERS, Task

SPLITS = ("train", "val", "test")
EVOLVE_SPLITS = ("pretrain/train", "pretrain/val", "gap/train", "gap/val", "gap/test")


def source_root(source: str, cls: str | None = None, difficulty: str | None = None):
    if source == "dmiplib":
        return DATA_ROOT / source / cls / difficulty
    return DATA_ROOT / source


def source_splits(root: Path) -> tuple[str, ...]:
    return EVOLVE_SPLITS if Path(root).name == "milp_evolve" else SPLITS


def all_source_roots() -> list[Path]:
    return [source_root("milp_evolve")] + [
        source_root("dmiplib", c, d) for c, d in TIERS
    ]


def task_datasets(task: Task) -> dict[str, GraphDataset]:
    root = source_root(task.source, task.cls, task.difficulty)
    prefix = "gap/" if task.source == "milp_evolve" else ""
    return {split: GraphDataset(root, prefix + split) for split in SPLITS}


def pretraining_pool() -> tuple[GraphDataset, GraphDataset]:
    """(train, val) unlabeled graphs of milp_evolve's pretraining splits."""
    evolve = source_root("milp_evolve")
    return (
        GraphDataset(evolve, "pretrain/train", keep_labels=False),
        GraphDataset(evolve, "pretrain/val", keep_labels=False),
    )
