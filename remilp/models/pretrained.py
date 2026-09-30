"""Building encoders and loading pretrained ones from run directories."""

from pathlib import Path

import torch

from remilp.config import ModelConfig
from remilp.models.encoder import MILPEncoder


def build_encoder(model: ModelConfig) -> MILPEncoder:
    return MILPEncoder(model.encoder)


def encoder_file(run_dir: Path, step: int | None = None) -> Path:
    run_dir = Path(run_dir)
    return run_dir / ("encoder.pt" if step is None else f"encoder_step{step:07d}.pt")


def save_encoder(encoder: MILPEncoder, path: Path) -> None:
    state = getattr(encoder, "_orig_mod", encoder).state_dict()
    torch.save(state, path)


def load_encoder_state(encoder: MILPEncoder, state: dict) -> list[str]:
    """Load a saved encoder; a mismatched pooler and instance MLP stay initialised.
    Returns the parameters left at initialisation."""
    own = encoder.state_dict()
    usable = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
    left = [k for k in own if k not in usable]
    body = [k for k in left if not k.startswith(("pooler.", "inst_mlp."))]
    if body:
        raise RuntimeError(f"saved encoder does not match: {body[:3]}")
    encoder.load_state_dict(usable, strict=False)
    return left
