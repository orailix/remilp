"""Integrality gap, frozen and fine-tuned, on the full training split."""

from remilp.config import MODELS, PretrainConfig
from remilp.train.pretrain import pretrain_dir

GAP_TRAIN = 8749
FORGE_RUN = pretrain_dir(MODELS["forge"], PretrainConfig(), 0).parent

ARMS = {
    "supervised": {"model": "random"},
    "remilp": {"model": "remilp", "step": 20000},
    "forge": {"model": "forge", "step": 20000},
    "forge-attn": {
        "model": "forge",
        "set": {"encoder.pooler": "coord_attention"},
        "pretrain_run": str(FORGE_RUN),
        "step": 20000,
    },
}
MODES = ["frozen", "finetune"]
TASKS = ["gap"]
SEEDS = [0, 1, 2]
PRETRAIN = {"pretrain.num_workers": 8}
EVAL = {
    "eval.train_size": GAP_TRAIN,
    "eval.max_steps": 100000,
    "eval.num_workers": 3,
    "eval.test_curve": True,
}
MAX_CONCURRENT = 8
