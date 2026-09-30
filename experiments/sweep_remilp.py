"""ReMILP hyperparameter search."""

from remilp.tasks import node_tasks

TIERS = [
    ("CA", "medium"),
    ("SC", "hard"),
    ("CFLP", "medium"),
    ("OTS", "easy"),
    ("MMCN", "medium-BC"),
    ("NNV", "easy"),
    ("LB", "hard"),
]
TASKS = node_tasks(TIERS)

MODEL = "remilp"
FIXED = {"pretrain.num_workers": 8}
SPACE = {
    "remilp.temperature": ("float", 0.05, 0.5, "log"),
    "remilp.negatives": ("categorical", [7, 15, 31, 63]),
    "remilp.equiv_k_frac": ("categorical", [0.01, 0.02, 0.05]),
    "remilp.aug_k_frac": ("categorical", [0.0, 0.05, 0.1, 0.2]),
    "remilp.aug_r": ("categorical", [2, 3]),
    "remilp.lambda_bound": ("float", 1.5, 6.0, "log"),
    "remilp.u_bound": ("float", 1.0, 4.0, "log"),
}
SEEDS = [0, 1, 2]
STEP = 20000
EVAL = {
    "eval.train_size": 100,
    "eval.max_steps": 5000,
    "eval.num_workers": 3,
    "eval.cache_gb": 40.0,
}
SEED_PENALTY = 0.5
CELL_PENALTY = 2.0
N_TRIALS = 56
N_STARTUP_TRIALS = 15
MAX_CONCURRENT = 8
PRETRAIN_CONCURRENT = 3
SAMPLER_SEED = 0
