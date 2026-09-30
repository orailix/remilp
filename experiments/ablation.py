"""ReMILP without substitutions, fine-tuned."""

from experiments.gap import GAP_TRAIN

NOSUB = {"remilp.equiv_k_frac": 0.0, "remilp.negative_types": "sample"}

ARMS = {
    "supervised": {"model": "random"},
    "remilp": {"model": "remilp", "step": 20000},
    "remilp-nosub": {"model": "remilp", "set": NOSUB, "step": 20000},
    "remilp-nosub-gap": {
        "model": "remilp",
        "set": {**NOSUB, "eval.train_size": GAP_TRAIN, "eval.max_steps": 100000},
        "step": 20000,
        "tasks": ["gap"],
    },
}
MODES = ["finetune"]
SEEDS = [0, 1, 2]
PRETRAIN = {"pretrain.num_workers": 8}
EVAL = {"eval.max_steps": 20000, "eval.num_workers": 3, "eval.test_curve": True}
MAX_CONCURRENT = 8
