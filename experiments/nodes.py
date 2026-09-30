"""Solution and activity prediction on the 23 pairs, frozen and fine-tuned
with 200 training instances."""

from remilp.tasks import NODE_TASKS

ARMS = {
    "supervised": {"model": "random"},
    "remilp": {"model": "remilp", "step": 20000},
    "forge": {"model": "forge", "step": 20000},
}
MODES = ["frozen", "finetune"]
TASKS = NODE_TASKS
SEEDS = [0, 1, 2]
PRETRAIN = {"pretrain.num_workers": 8}
EVAL = {"eval.max_steps": 50000, "eval.num_workers": 3, "eval.test_curve": True}
MAX_CONCURRENT = 8
