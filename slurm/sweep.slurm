#!/bin/bash
# One Optuna worker per array task, all sharing runs/sweeps/<name>/journal.log.
# Array size x --trials is the sweep's budget (4 x 14 = N_TRIALS = 56). The
# random-encoder cells are computed by whichever workers claim them first.
#   mkdir -p logs && sbatch -A <account> slurm/sweep.slurm experiments/sweep_remilp.py
#SBATCH --job-name=sweep
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --output=logs/sweep-%A_%a.log
#SBATCH --error=logs/sweep-%A_%a.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
sweep="${1:?usage: sbatch slurm/sweep.slurm experiments/sweep_<name>.py}"
exec pixi run --as-is remilp sweep "$sweep" \
  --trials 14 --worker "${SLURM_ARRAY_TASK_ID:-0}" --max-concurrent auto
