"""Configuration dataclasses, presets and `--set section.field=value` overrides."""

import dataclasses
import hashlib
import json
import os
import typing
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any

DATA_ROOT = Path(os.environ.get("REMILP_DATA_ROOT", "data"))
RUNS_ROOT = Path(os.environ.get("REMILP_RUNS_ROOT", "runs"))
PROCESSED_VERSION = "708c56a9eebb"


@dataclass(frozen=True)
class EncoderConfig:
    hidden_dim: int = 128
    n_layers: int = 2
    heads: int = 4
    pooler: str = "coord_attention"  # "coord_attention" | "vq"
    pooler_out_dim: int = 512
    vq_codebook_size: int = 256  # vq only
    vq_decay: float = 0.99  # vq only


@dataclass(frozen=True)
class ReMILPConfig:
    negatives: int = 63  # K, of each type below
    negative_types: str = "both"  # "both" | "sample"
    temperature: float = 0.09007080825746538
    equiv_k_frac: float = 0.05  # fraction of candidate variables transformed; 0 = none
    equiv_k_cap: int = (
        256  # max transformed (or, with equiv_k_frac 0, scored) variables per graph
    )
    lambda_bound: float = 4.011716655405568
    u_bound: float = 1.1211016606919455
    aug_k_frac: float = 0.2  # redundant rows added, as a fraction of constraints
    aug_r: int = 2  # source rows per redundant row


@dataclass(frozen=True)
class ForgeConfig:
    commitment: float = 0.04640404514001917
    neg_to_pos_ratio: float = 0.2
    max_edges_per_batch: int = 100_000


@dataclass(frozen=True)
class ModelConfig:
    name: str  # "random" | "remilp" | "forge"
    encoder: EncoderConfig = EncoderConfig()
    remilp: ReMILPConfig | None = None
    forge: ForgeConfig | None = None

    @property
    def objective(self):
        return {"remilp": self.remilp, "forge": self.forge}.get(self.name)


MODELS = {
    "random": ModelConfig("random"),
    "remilp": ModelConfig("remilp", remilp=ReMILPConfig()),
    "forge": ModelConfig(
        "forge", encoder=EncoderConfig(pooler="vq"), forge=ForgeConfig()
    ),
}


@dataclass(frozen=True)
class PretrainConfig:
    steps: int = 20_000
    lr: float = 1e-3
    grad_clip: float = 5.0
    batch_nodes: int = 200_000
    batch_edges: int = 1_200_000
    num_workers: int = 12
    val_every: int = 500
    ckpt_every: int = 5_000
    log_every: int = 100
    bf16: bool = True
    compile: bool = True


@dataclass(frozen=True)
class EvalConfig:
    task: str = "gap"
    frozen: bool = True
    lr: float = 1e-4
    train_size: int = 200  # the first train_size graphs of the training split
    max_steps: int = 100_000
    patience: int = 5_000  # steps without validation improvement; 0 disables
    val_every: int = 100
    batch_nodes: int = 200_000
    batch_edges: int = 1_200_000
    num_workers: int = 12
    bf16: bool = True
    log_every: int = 100
    compile: bool = True  # torch.compile the encoder when it runs every step
    # GB of CPU memory for the frozen encoder's embeddings of train, val, test; 0 disables
    cache_gb: float = 10.0
    test_curve: bool = False  # also measure the test split at every validation
    ckpt_every: int = 2_000  # multiple of val_every


# Fields that change how a run executes but not what it computes.
RUNTIME_FIELDS = {
    "num_workers",
    "compile",
    "bf16",
    "log_every",
    "cache_gb",
    "test_curve",
}
EVAL_RUNTIME_FIELDS = RUNTIME_FIELDS | {"ckpt_every"}


def run_relevant(cfg) -> dict:
    """A training config without its runtime-only fields."""
    skip = EVAL_RUNTIME_FIELDS if isinstance(cfg, EvalConfig) else RUNTIME_FIELDS
    return {k: v for k, v in to_dict(cfg).items() if k not in skip}


def to_dict(cfg) -> Any:
    if is_dataclass(cfg):
        return {f.name: to_dict(getattr(cfg, f.name)) for f in fields(cfg)}
    return cfg


def from_dict(cls, d: dict):
    kwargs = {}
    for f in fields(cls):
        value = d[f.name]
        inner = _dataclass_type(f.type)
        if inner is not None and isinstance(value, dict):
            value = from_dict(inner, value)
        kwargs[f.name] = value
    return cls(**kwargs)


def config_hash(cfg, length: int = 10) -> str:
    payload = json.dumps(to_dict(cfg), sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:length]


def model_tag(model: ModelConfig) -> str:
    """The model name plus every non-default field."""
    parts = [model.name]
    default = MODELS[model.name]
    for section in ("encoder", "remilp", "forge"):
        cfg, ref = getattr(model, section), getattr(default, section)
        if cfg is None or ref is None:
            continue
        for f in fields(cfg):
            value = getattr(cfg, f.name)
            if value != getattr(ref, f.name):
                parts.append(f"{f.name.replace('_', '')}{value}")
    return "-".join(parts)


_SECTIONS = {
    "encoder": EncoderConfig,
    "remilp": ReMILPConfig,
    "forge": ForgeConfig,
    "pretrain": PretrainConfig,
    "eval": EvalConfig,
}


def parse_override(text: str) -> tuple[str, str, Any]:
    """'section.field=value' -> (section, field, typed value)."""
    key, _, raw = text.partition("=")
    section, _, name = key.partition(".")
    if section not in _SECTIONS or not name:
        raise ValueError(f"bad override '{text}': expected <section>.<field>=<value>")
    cls = _SECTIONS[section]
    try:
        f = next(f for f in fields(cls) if f.name == name)
    except StopIteration:
        raise ValueError(f"{cls.__name__} has no field '{name}'") from None
    return section, name, _parse_value(raw, f.type)


def apply_overrides(model: ModelConfig, cfg, overrides: list[str]):
    """(model, cfg) with the overrides applied."""
    for text in overrides:
        section, name, value = parse_override(text)
        if section in ("pretrain", "eval"):
            if not isinstance(cfg, _SECTIONS[section]):
                raise ValueError(f"override '{text}' does not apply to this command")
            cfg = dataclasses.replace(cfg, **{name: value})
        else:
            current = getattr(model, section)
            if current is None:
                raise ValueError(f"model '{model.name}' has no '{section}' section")
            model = dataclasses.replace(
                model, **{section: dataclasses.replace(current, **{name: value})}
            )
    return model, cfg


def _parse_value(raw: str, annotation) -> Any:
    types = {annotation, *typing.get_args(annotation)}
    if bool in types:
        if raw.lower() in ("true", "1", "yes"):
            return True
        if raw.lower() in ("false", "0", "no"):
            return False
        raise ValueError(f"expected a boolean, got '{raw}'")
    if int in types:
        return int(raw)
    if float in types:
        return float(raw)
    return raw


def _dataclass_type(annotation):
    """The dataclass in a field annotation such as `ReMILPConfig | None`, or None."""
    for candidate in (annotation, *typing.get_args(annotation)):
        if is_dataclass(candidate):
            return candidate
    return None
