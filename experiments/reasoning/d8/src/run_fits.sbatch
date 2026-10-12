#!/bin/bash
#SBATCH --job-name=d8_mechanism_fits
#SBATCH --partition=long
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --array=0-20%10
#SBATCH --output=experiments/reasoning/d8/logs/fit-%A_%a.out
#SBATCH --error=experiments/reasoning/d8/logs/fit-%A_%a.err
REPO="${TNDM_ROOT:-${SLURM_SUBMIT_DIR:-$(git rev-parse --show-toplevel)}}"  # repository root
set -euo pipefail
WORKTREE=$REPO
cd "$WORKTREE"
export PYTHONPATH="$WORKTREE"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=${PYTHON:-python}
$PY experiments/reasoning/d8/src/fit_budgets.py \
  --repeat "$SLURM_ARRAY_TASK_ID"
