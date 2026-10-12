#!/bin/bash
# Fit and freeze one learner seed at every budget listed in $BUDGETS (space-separated).
#SBATCH --job-name=d12_fit
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
REPO="${TNDM_ROOT:-${SLURM_SUBMIT_DIR:-$(git rev-parse --show-toplevel)}}"  # repository root
set -euo pipefail
WS=$REPO
HERE=$WS/experiments/reasoning/d12
cd "$WS"
export PYTHONPATH="$WS:$WS/src" PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=${PYTHON:-python}
R15="${TNDM_D12_DEV_SEED31:-$REPO/experiments/reasoning/d12/results/dev_seed31}"  # seed-31 development cells (not shipped; see README)
EXTRA=()
[ "$SEED" = 31 ] && EXTRA=(--fallback-root "$R15" --nonblind)
nvidia-smi --query-gpu=name --format=csv,noheader; echo "host=$(hostname)"
for budget in $BUDGETS; do
  $PY "$HERE/code/fit_d12.py" --seed "$SEED" --new-budget "$budget" --root "$HERE/results" \
    --preregistration-sha256 "$PREREG_SHA" "${EXTRA[@]}"
done
