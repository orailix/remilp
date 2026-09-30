"""Graphs of one source: manifest.json and <split>/graphs/<id>.pt.zst, batched
under node and edge budgets."""

import io
import json
import os
import resource
from pathlib import Path

import numpy as np
import torch
import zstandard
from torch.utils.data import Dataset, Sampler
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader

from remilp.config import PROCESSED_VERSION

VAR_FEATURES = (
    "obj_coeffs",
    "has_lb",
    "has_ub",
    "is_continuous",
    "is_binary",
    "lb_val",
    "ub_val",
)
CON_FEATURES = ("has_lhs", "has_rhs", "lhs_val", "rhs_val")
V2C = ("variables", "participates_in", "constraints")
C2V = ("constraints", "has_variable", "variables")

LABEL_ATTRS = {
    "variables": ("optimal_sol", "solution_probs"),
    "constraints": ("active_at_optimum", "active_probs"),
}
GRAPH_LABELS = ("lp_ip_gap",)
# Unused fields of the published graphs, dropped on load.
UNUSED_ATTRS = {
    "graph": ("lp_objval",),
    "variables": ("lp_assignments", "lp_in_basis", "degree", "avg_coeff"),
    "constraints": ("lp_dualvals", "lp_is_active", "degree", "avg_coeff", "row_scale"),
}

MANIFEST = "manifest.json"
GRAPH_SUFFIX = ".pt.zst"
ZSTD_LEVEL = 6


class MissingCacheError(FileNotFoundError):
    pass


def write_json_atomic(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with tmp.open("w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def manifest_path(root: Path) -> Path:
    return Path(root) / MANIFEST


def load_manifest(root: Path) -> dict:
    path = manifest_path(root)
    if not path.exists():
        raise MissingCacheError(f"{path} not found")
    with path.open() as f:
        manifest = json.load(f)
    if manifest.get("processing") != PROCESSED_VERSION:
        raise MissingCacheError(
            f"{path} was processed with version {manifest.get('processing')!r}, "
            f"this code expects {PROCESSED_VERSION!r}"
        )
    return manifest


def graph_path(root: Path, entry: dict) -> Path:
    return Path(root) / entry["split"] / "graphs" / (entry["id"] + GRAPH_SUFFIX)


def load_graph(path: Path) -> HeteroData:
    with open(path, "rb") as f:
        raw = zstandard.ZstdDecompressor().decompress(f.read())
    g = torch.load(io.BytesIO(raw), weights_only=False)
    for key in UNUSED_ATTRS["graph"]:
        if hasattr(g, key):
            delattr(g, key)
    for store in ("variables", "constraints"):
        for key in UNUSED_ATTRS[store]:
            if key in g[store]:
                del g[store][key]
    return g


def save_graph(g: HeteroData, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with open(tmp, "wb") as f:
        f.write(graph_bytes(g))
    os.replace(tmp, path)


def graph_bytes(g: HeteroData) -> bytes:
    buf = io.BytesIO()
    torch.save(g, buf)
    return zstandard.ZstdCompressor(level=ZSTD_LEVEL).compress(buf.getvalue())


def drop_labels(g: HeteroData) -> HeteroData:
    g = g.clone()
    for store, keys in LABEL_ATTRS.items():
        for key in keys:
            if key in g[store]:
                del g[store][key]
    for key in GRAPH_LABELS:
        if hasattr(g, key):
            delattr(g, key)
    return g


class GraphDataset(Dataset):
    """The graphs of one split of a source (every split when `split` is None)."""

    def __init__(
        self,
        root: Path,
        split: str | None = None,
        keep_labels: bool = True,
        transform=None,
    ):
        self.root = Path(root)
        manifest = load_manifest(self.root)
        self.entries = [
            e for e in manifest["entries"] if split is None or e["split"] == split
        ]
        if not self.entries:
            raise MissingCacheError(f"{self.root} has no entries for split {split!r}")
        first = graph_path(self.root, self.entries[0])
        if not first.exists():
            raise MissingCacheError(
                f"{first} not found; build the graphs with "
                "`remilp data build --confirm-reprocess`"
            )
        self.keep_labels = keep_labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, i: int):
        g = load_graph(graph_path(self.root, self.entries[i]))
        g = g if self.keep_labels else drop_labels(g)
        return g if self.transform is None else self.transform(g)

    def set_transform(self, transform) -> None:
        self.transform = transform

    def subset(self, indices) -> "GraphDataset":
        ds = GraphDataset.__new__(GraphDataset)
        ds.__dict__.update(self.__dict__)
        ds.entries = [self.entries[i] for i in indices]
        return ds

    @property
    def sizes(self) -> np.ndarray:
        """(num_nodes, num_edges) per item, from the manifest."""
        return np.array(
            [[e["num_nodes"], e["num_edges"]] for e in self.entries], dtype=np.int64
        ).reshape(-1, 2)


class DualBudgetBatchSampler(Sampler):
    """Consecutive items under a node and an edge budget; oversized items are skipped."""

    def __init__(self, sizes, max_nodes, max_edges, shuffle=False):
        self.sizes = np.asarray(sizes)
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.shuffle = shuffle

    def __iter__(self):
        n = len(self.sizes)
        order = torch.randperm(n).tolist() if self.shuffle else range(n)
        batch, nodes, edges = [], 0, 0
        for idx in order:
            dn, de = int(self.sizes[idx, 0]), int(self.sizes[idx, 1])
            if dn > self.max_nodes or de > self.max_edges:
                continue
            if batch and (nodes + dn > self.max_nodes or edges + de > self.max_edges):
                yield batch
                batch, nodes, edges = [], 0, 0
            batch.append(idx)
            nodes += dn
            edges += de
        if batch:
            yield batch


def num_graphs(batch) -> int:
    return batch[0].num_graphs if isinstance(batch, (list, tuple)) else batch.num_graphs


def to_device(batch, device):
    """A collated batch, or the list of batches a tuple-valued transform yields."""
    if isinstance(batch, (list, tuple)):
        return [b.to(device, non_blocking=True) for b in batch]
    return batch.to(device, non_blocking=True)


FD_STRATEGY_MIN_NOFILE = 4096


def _worker_init(worker_id: int) -> None:
    soft, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft < FD_STRATEGY_MIN_NOFILE:
        torch.multiprocessing.set_sharing_strategy("file_system")


def make_loader(dataset, max_nodes, max_edges, num_workers, shuffle) -> DataLoader:
    sampler = DualBudgetBatchSampler(
        dataset.sizes, max_nodes, max_edges, shuffle=shuffle
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=2 if num_workers > 0 else None,
        worker_init_fn=_worker_init if num_workers > 0 else None,
    )
