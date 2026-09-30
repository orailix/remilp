# ReMILP

Pretraining of GNN encoders for MILPs with ReMILP, and the paper's downstream evaluation.

Models: `remilp` (ours), `forge` (FORGE re-implementation), `random` (random-encoder
baseline when frozen, supervised training when fine-tuned). Tasks: `gap` (MAE), and
`solution-<CLASS>-<difficulty>` / `activity-<CLASS>-<difficulty>` (KL) on the 23
Distributional MIPLIB pairs of `remilp/tasks.py`.

## Setup

```bash
pixi install                  # default env
pixi install -e analysis      # adds SCIP and matplotlib
pixi run hf download orailix/remilp-data --repo-type dataset --local-dir data --exclude "*/raw.tar"
pixi run remilp data check
```

## Usage

```bash
pixi run remilp pretrain --model remilp --seed 0
pixi run remilp evaluate --model remilp --task solution-SC-hard --frozen --seed 0 --step 20000
```

Defaults in `remilp/config.py` are derived from a hyperparameter sweep. Override with `--set section.field=value`.

## Reproducing the paper

```bash
pixi run remilp launch experiments/nodes.py --local 4     # also gap.py, ablation.py
pixi run remilp sweep experiments/sweep_remilp.py --trials 56   # also sweep_forge.py
pixi run -e analysis python paper/tables.py
pixi run -e analysis python paper/figures.py
pixi run python paper/probes.py
```

Pretraining requires a GPU with about 30 GB of VRAM. Each frozen evaluation keeps up to 10 GB of embeddings in host memory (`eval.cache_gb`).

## Dataset

The data used in our experiments can be found in the hugging face repo [orailix/remilp-data](https://huggingface.co/datasets/orailix/remilp-data). All instances are re-distributed under their original licenses. Refer to `DATA_LICENSE.md` for more details.
