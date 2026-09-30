"""Graphs from raw inputs: <split>/instances/<id>.<mps|lp>[.gz] and
<split>/labels/<id>.json.gz, either {"objective", "lp_ip_gap"} or
{"var_names", "objectives", "solutions"}."""

import gzip
import json
import multiprocessing
import re
from pathlib import Path

import numpy as np
import torch
from pyscipopt import Model
from torch_geometric.data import HeteroData
from torch_geometric.utils import scatter
from tqdm import tqdm

from remilp.config import PROCESSED_VERSION
from remilp.data.dataset import (
    C2V,
    V2C,
    graph_path,
    manifest_path,
    save_graph,
    write_json_atomic,
)
from remilp.data.package import split_dirs
from remilp.data.transforms import normalize

MIN_VARS, MIN_CONS = 2, 1
INSTANCE_SUFFIX = re.compile(r"\.(mps|lp)(\.gz)?$")


def _natural_key(path: Path):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", path.as_posix())]


def instance_files(root: Path) -> list[tuple[str, str, Path]]:
    """(split, id, path) of every instance file under <split>/instances/."""
    root = Path(root)
    found = []
    for split_dir in split_dirs(root):
        inst = split_dir / "instances"
        if not inst.is_dir():
            continue
        split = split_dir.relative_to(root).as_posix()
        for path in sorted(inst.rglob("*"), key=_natural_key):
            if path.is_file() and INSTANCE_SUFFIX.search(path.name):
                rel = path.relative_to(inst).as_posix()
                found.append((split, INSTANCE_SUFFIX.sub("", rel), path))
    return found


def instance_file(root: Path, entry: dict) -> Path:
    stem = Path(root) / entry["split"] / "instances" / entry["id"]
    matches = [
        p for p in stem.parent.glob(stem.name + ".*") if INSTANCE_SUFFIX.search(p.name)
    ]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one instance file for {stem}, found {matches}"
        )
    return matches[0]


def label_file(root: Path, entry: dict) -> Path:
    return Path(root) / entry["split"] / "labels" / (entry["id"] + ".json.gz")


def read_label(path: Path) -> dict:
    with gzip.open(path, "rt") as f:
        return json.load(f)


def _instance_info(path):
    """(degenerate, num_nodes, num_edges) of one instance file."""
    try:
        model = read_problem(str(path))
        variables = model.getVars(transformed=False)
        conss = model.getConss(transformed=False)
        all_linear = all(cons.isLinear() for cons in conss)
        n_edges = sum(len(model.getConsVars(c)) for c in conss)
        model.freeProb()
        degenerate = (
            len(variables) < MIN_VARS or len(conss) < MIN_CONS or not all_linear
        )
        return degenerate, len(variables) + len(conss), 2 * n_edges
    except Exception:
        return True, 0, 0


def build_manifest(root: Path, source: str, num_processes=12) -> dict:
    """Write manifest.json for the instances under root, dropping degenerate ones."""
    root = Path(root)
    found = instance_files(root)
    if not found:
        raise FileNotFoundError(f"no <split>/instances/ files under {root}")
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(num_processes) as pool:
        info = list(
            tqdm(
                pool.imap(_instance_info, [p for _, _, p in found]),
                total=len(found),
                desc="Scanning instances",
            )
        )
    entries = [
        {"id": id_, "split": split, "num_nodes": n_nodes, "num_edges": n_edges}
        for (split, id_, _), (degenerate, n_nodes, n_edges) in zip(found, info)
        if not degenerate
    ]
    manifest = {"source": source, "processing": PROCESSED_VERSION, "entries": entries}
    write_json_atomic(manifest_path(root), manifest)
    return manifest


def process_all(root: Path, manifest: dict, num_processes=12) -> None:
    """Write <split>/graphs/<id>.pt.zst for every manifest entry without one."""
    root = Path(root)
    jobs = [
        (str(root), entry)
        for entry in manifest["entries"]
        if not graph_path(root, entry).exists()
    ]
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(num_processes) as pool:
        list(
            tqdm(
                pool.imap_unordered(process_instance, jobs),
                total=len(jobs),
                desc="Processing",
            )
        )


def process_instance(args) -> None:
    root, entry = Path(args[0]), args[1]
    data = build_graph(instance_file(root, entry), label_file(root, entry))
    save_graph(data, graph_path(root, entry))


def build_graph(instance: Path, label: Path | None = None) -> HeteroData:
    model = read_problem(str(instance))
    data, var_to_index = extract_bipartite_graph(model)
    maximize = model.getObjectiveSense() == "maximize"
    model.freeProb()
    if label is not None and label.exists():
        attach_labels(data, read_label(label), var_to_index, maximize)
    return normalize(data)


def read_problem(filename: str) -> Model:
    model = Model()
    model.hideOutput()
    model.readProblem(filename)
    return model


def extract_bipartite_graph(model: Model) -> tuple[HeteroData, dict]:
    """The variable-constraint bipartite graph, objective stored in minimize convention."""
    conss = model.getConss(transformed=False)
    variables = model.getVars(transformed=False)
    var_to_index = {var.name: i for i, var in enumerate(variables)}
    inf = model.infinity()
    obj_scaling = -1.0 if model.getObjectiveSense() == "maximize" else 1.0

    vtypes = np.array([var.vtype() for var in variables])
    obj_coeffs = np.array([var.getObj() / obj_scaling for var in variables])
    lb = np.array([var.getLbOriginal() for var in variables])
    ub = np.array([var.getUbOriginal() for var in variables])
    has_lb, has_ub = lb > -inf, ub < inf
    lhs = np.array([model.getLhs(c) for c in conss])
    rhs = np.array([model.getRhs(c) for c in conss])
    has_lhs, has_rhs = lhs > -inf, rhs < inf

    rows, cols, coeffs = [], [], []
    for i, cons in enumerate(conss):
        for var, coeff in zip(model.getConsVars(cons), model.getConsVals(cons)):
            rows.append(i)
            cols.append(var_to_index.get(var.name, var.getIndex()))
            coeffs.append(coeff)

    data = HeteroData()
    edge_attr = torch.tensor(coeffs, dtype=torch.float32).unsqueeze(1)
    data[V2C].edge_index = torch.tensor([cols, rows], dtype=torch.long)
    data[V2C].edge_attr = edge_attr
    data[C2V].edge_index = torch.tensor([rows, cols], dtype=torch.long)
    data[C2V].edge_attr = edge_attr.clone()

    col = lambda a: torch.from_numpy(np.asarray(a)).float().unsqueeze(-1)
    v = data["variables"]
    v.num_nodes = len(variables)
    v.is_continuous = col(vtypes == "CONTINUOUS")
    v.is_binary = torch.from_numpy(vtypes == "BINARY").unsqueeze(-1)
    v.is_integer = torch.from_numpy(np.isin(vtypes, ["INTEGER", "IMPLINT"])).unsqueeze(
        -1
    )
    v.obj_coeffs = col(obj_coeffs)
    v.has_lb, v.has_ub = col(has_lb), col(has_ub)
    v.lb_val, v.ub_val = col(np.where(has_lb, lb, 0.0)), col(np.where(has_ub, ub, 0.0))
    c = data["constraints"]
    c.num_nodes = len(conss)
    c.has_lhs, c.has_rhs = col(has_lhs), col(has_rhs)
    c.lhs_val, c.rhs_val = col(np.where(has_lhs, lhs, 0.0)), col(
        np.where(has_rhs, rhs, 0.0)
    )
    data.scaling_coeff = torch.tensor([obj_scaling], dtype=torch.float32)
    return data, var_to_index


def attach_labels(data, label: dict, var_to_index: dict, maximize: bool) -> None:
    if "lp_ip_gap" in label:
        data.lp_ip_gap = torch.tensor([label["lp_ip_gap"]], dtype=torch.float32)
    if "solutions" in label:
        attach_solution(data, label, var_to_index, maximize)
        attach_active_at_optimum(data, label, var_to_index, maximize)


def _pool_assignments_and_weights(label, data, var_to_index, maximize):
    """(pool values [n_sols, n_vars], softmax weights over objectives, best index)."""
    sols = np.asarray(label["solutions"], dtype=np.float64)
    scores = np.asarray(label["objectives"], dtype=np.float64)
    scores = scores if maximize else -scores
    exp_w = np.exp(scores - scores.max())
    weights = (exp_w / exp_w.sum()).astype(np.float32)
    v = data["variables"]
    aligned = np.zeros((sols.shape[0], v.num_nodes), dtype=np.float64)
    for j, name in enumerate(label["var_names"]):
        if name in var_to_index:
            aligned[:, var_to_index[name]] = sols[:, j]
    pool = torch.from_numpy(aligned)
    integral = (v.is_binary | v.is_integer).squeeze(-1)
    binary = v.is_binary.squeeze(-1)
    pool[:, integral] = pool[:, integral].round()
    pool[:, binary] = pool[:, binary].clamp(0.0, 1.0)
    return pool, torch.from_numpy(weights), int(np.argmax(scores))


def attach_solution(data, label, var_to_index, maximize) -> None:
    pool, weights, best = _pool_assignments_and_weights(
        label, data, var_to_index, maximize
    )
    v = data["variables"]
    probs = (weights.unsqueeze(1) * pool.float()).sum(0)
    binary = v.is_binary.squeeze(-1)
    probs[binary] = probs[binary].clamp(0.0, 1.0)
    v.solution_probs = probs
    v.optimal_sol = pool[best].float()


def attach_active_at_optimum(data, label, var_to_index, maximize) -> None:
    """Tight rows at the best solution, and their pool-weighted probability."""
    n_cons = data["constraints"].num_nodes
    pool, weights, best = _pool_assignments_and_weights(
        label, data, var_to_index, maximize
    )
    ei, ea = data[V2C].edge_index, data[V2C].edge_attr.squeeze(-1).double()
    n_sols = pool.shape[0]
    contrib = pool[:, ei[0]] * ea.unsqueeze(0)
    flat = (torch.arange(n_sols).unsqueeze(1) * n_cons + ei[1].unsqueeze(0)).reshape(-1)

    def row_total(values):
        total = scatter(
            values.reshape(-1), flat, dim_size=n_sols * n_cons, reduce="sum"
        )
        return total.view(n_sols, n_cons)

    row_sum, tol = row_total(contrib), 1e-6 * (1.0 + row_total(contrib.abs()))
    c = data["constraints"]
    has_lhs, has_rhs = c.has_lhs.squeeze(-1).bool(), c.has_rhs.squeeze(-1).bool()
    tight = (has_lhs & (row_sum - c.lhs_val.squeeze(-1).double()).abs().le(tol)) | (
        has_rhs & (row_sum - c.rhs_val.squeeze(-1).double()).abs().le(tol)
    )
    c.active_at_optimum = tight[best].float().unsqueeze(-1)
    c.active_probs = (
        (weights.unsqueeze(1) * tight.float()).sum(0).clamp(0.0, 1.0).unsqueeze(-1)
    )
