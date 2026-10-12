#!/bin/bash
# Analysis job. It needs no GPU, but a GPU request is what routes it to `long` while the *-cpu partitions drain.
#SBATCH --job-name=d12_analysis
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
REPO="${TNDM_ROOT:-${SLURM_SUBMIT_DIR:-$(git rev-parse --show-toplevel)}}"  # repository root
set -euo pipefail
WS=$REPO
HERE=$WS/experiments/reasoning/d12
cd "$WS"
export PYTHONPATH="$WS:$WS/src" PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 MPLBACKEND=Agg
PY=${PYTHON:-python}
R15="${TNDM_D12_DEV_SEED31:-$REPO/experiments/reasoning/d12/results/dev_seed31}"  # seed-31 development cells (not shipped; see README)
$PY "$HERE/code/analyze_d12.py" --root "$HERE/results" --out "${OUT:-$HERE/analysis}" --seeds ${SEEDS:-31 43 59} --fallback-root "$R15"
