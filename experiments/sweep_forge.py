"""FORGE hyperparameter search."""

from experiments.sweep_remilp import (
    CELL_PENALTY,
    EVAL,
    FIXED,
    MAX_CONCURRENT,
    N_STARTUP_TRIALS,
    N_TRIALS,
    PRETRAIN_CONCURRENT,
    SAMPLER_SEED,
    SEED_PENALTY,
    SEEDS,
    STEP,
    TASKS,
)

MODEL = "forge"
SPACE = {
    "encoder.vq_codebook_size": ("categorical", [64, 128, 256, 1024]),
    "encoder.vq_decay": ("categorical", [0.8, 0.9, 0.95, 0.99]),
    "forge.commitment": ("float", 0.01, 2.0, "log"),
}
