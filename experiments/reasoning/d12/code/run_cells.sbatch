#!/bin/bash
# Generic d=12 study cell runner.  One array task trains $CHUNK coalitions.
# Required environment (passed with sbatch --export):
#   MASKS_FILE OUT_DIR SEED CHUNK PREREG_SHA  [INIT_SEED DATA_SEED REQUIRE_GPU]
#SBATCH --job-name=d12_d12
#SBATCH --partition=gpu
#SBATCH --requeue
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:50:00
REPO="${TNDM_ROOT:-${SLURM_SUBMIT_DIR:-$(git rev-parse --show-toplevel)}}"  # repository root
set -euo pipefail
WS=$REPO
cd "$WS"
export PYTHONPATH="$WS:$WS/src"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2
export MPLBACKEND=Agg PYTHONUNBUFFERED=1
PY=${PYTHON:-python}
EXTRA=()
[ -n "${INIT_SEED:-}" ] && EXTRA+=(--init-seed "$INIT_SEED")
[ -n "${DATA_SEED:-}" ] && EXTRA+=(--data-seed "$DATA_SEED")
[ -n "${REPLICATE:-}" ] && EXTRA+=(--replicate "$REPLICATE")
RUNNER="$WS/experiments/reasoning/d12/code/query_paths_d12_runner.py"
$PY "$RUNNER" \
  --output-dir "$OUT_DIR" \
  --masks-file "$MASKS_FILE" \
  --task-index "$SLURM_ARRAY_TASK_ID" \
  --chunk "$CHUNK" \
  --seed "$SEED" \
  --require-gpu "${REQUIRE_GPU:-rtx8000}" \
  --preregistration-sha256 "$PREREG_SHA" \
  "${EXTRA[@]}"
