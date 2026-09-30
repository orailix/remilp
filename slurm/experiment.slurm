#!/bin/bash
# One worker of an experiment file on one GPU: pretrains three runs at a time, then
# runs eight cells at once. Workers claim run directories, so any number of copies of
# this job share the experiment, and a copy resubmitted after a failure or a time-out
# finishes what is left:
#   mkdir -p logs && for i in 1 2 3 4; do sbatch -A <account> -J nodes slurm/experiment.slurm experiments/nodes.py; done
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=20:00:00
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
experiment="${1:?usage: sbatch -J <name> slurm/experiment.slurm experiments/<name>.py}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

pixi run --as-is remilp launch "$experiment" --claim 3 --pretrain-only \
  && pixi run --as-is remilp launch "$experiment" --claim 8 \
  || echo "some jobs failed; see the FAILED lines above and resubmit"
