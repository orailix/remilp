import torch
from torch import nn


class Objective(nn.Module):
    """A pretraining objective: forward(batch) -> (loss, logs). The encoder is not a
    submodule; `views` is an optional per-graph transform run by the loader."""

    views = None

    def __init__(self, encoder):
        super().__init__()
        self._encoder = [encoder]

    @property
    def encoder(self):
        return self._encoder[0]

    def forward(self, batch) -> tuple[torch.Tensor, dict]:
        raise NotImplementedError
