#!/bin/bash
#SBATCH --job-name=d24_pilot
#SBATCH --partition=long
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=24:00:00
#SBATCH --output=experiments/reasoning/d24/pilot/logs/pilot-%A_%a.out
#SBATCH --error=experiments/reasoning/d24/pilot/logs/pilot-%A_%a.err
# CPU-only job (no GPU requested).
REPO="${TNDM_ROOT:-${SLURM_SUBMIT_DIR:-$(git rev-parse --show-toplevel)}}"  # repository root
set -euo pipefail
D24_DIR=$REPO/experiments/reasoning/d24
cd "$D24_DIR"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
PY=${PYTHON:-python}
ARM=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" configs/pilot_arms.txt)
read -r K G D NOISE REPLICATE <<< "$ARM"
OUT="$D24_DIR/pilot/results/arm_k${K}_g${G}_d${D}_c${NOISE}_r${REPLICATE}.json"
if [ -s "$OUT" ]; then
  echo "already complete: $OUT"
  exit 0
fi
"$PY" -u pilot/run_arm.py --k "$K" --g "$G" --d "$D" --noise "$NOISE" --replicate "$REPLICATE" --out "$OUT"
