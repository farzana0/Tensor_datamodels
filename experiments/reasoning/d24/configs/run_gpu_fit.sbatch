#!/bin/bash
#SBATCH --job-name=d24_fit
#SBATCH --partition=long
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=experiments/reasoning/d24/gpu/logs/%x-%A_%a.out
#SBATCH --error=experiments/reasoning/d24/gpu/logs/%x-%A_%a.err
# CPU-only job.
# Array index = 3*seed_index + budget_index, seeds (31,43,59), budgets (256,512,1024).
REPO="${TNDM_ROOT:-${SLURM_SUBMIT_DIR:-$(git rev-parse --show-toplevel)}}"  # repository root
set -euo pipefail
cd $REPO/experiments/reasoning/d24
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
SEEDS=(31 43 59); BUDGETS=(256 512 1024)
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID / 3))]}; BUDGET=${BUDGETS[$((SLURM_ARRAY_TASK_ID % 3))]}
if [ -s gpu/fits/seed$SEED/budget_$BUDGET/frozen_result.json ]; then echo "already frozen"; exit 0; fi
${PYTHON:-python} -u gpu/fit_and_freeze.py --seed $SEED --budget $BUDGET
