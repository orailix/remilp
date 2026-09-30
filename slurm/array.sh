#!/usr/bin/env bash
# One SLURM array over the jobs of an experiment file (`remilp launch --slurm`). Usage:
#   N=$(pixi run remilp launch experiments/nodes.py --count eval)
#   sbatch --array=0-$((N-1))%8 slurm/array.sh experiments/nodes.py eval
#SBATCH --job-name=remilp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=20:00:00
#SBATCH --output=logs/slurm-%A_%a.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
exec pixi run --as-is remilp launch "$1" --job "$SLURM_ARRAY_TASK_ID" --phase "${2:-eval}"
