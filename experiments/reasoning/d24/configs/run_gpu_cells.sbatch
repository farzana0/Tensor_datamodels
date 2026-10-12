#!/bin/bash
#SBATCH --job-name=d24_gpu
#SBATCH --partition=long
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=experiments/reasoning/d24/gpu/logs/%x-%A_%a.out
#SBATCH --error=experiments/reasoning/d24/gpu/logs/%x-%A_%a.err
# Usage: sbatch --array=0-N --export=ALL,MASKS=<file>,SEED=<s>,CHUNK=<cells per task>,WORKERS=<procs> run_gpu_cells.sbatch
# Each task trains CHUNK coalitions, split across WORKERS independent processes on one RTX8000.
# Every coalition is a fresh model with its own common-seed initialisation; outputs are resumable per cell.
REPO="${TNDM_ROOT:-${SLURM_SUBMIT_DIR:-$(git rev-parse --show-toplevel)}}"  # repository root
set -euo pipefail
WT=$REPO
D24_DIR=$WT/experiments/reasoning/d24
cd "$WT"
export PYTHONPATH=$WT:$WT/src PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
PY=${PYTHON:-python}
"$PY" -c "import sys, torch; assert sys.version_info[:2] == (3, 10); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
START=$(( SLURM_ARRAY_TASK_ID * CHUNK ))
PER=$(( (CHUNK + WORKERS - 1) / WORKERS ))
for w in $(seq 0 $((WORKERS - 1))); do
  "$PY" "$D24_DIR/gpu/transformer_and_of_or_d24.py" --output-dir "$D24_DIR/gpu/observed/seed$SEED" \
    --masks-file "$D24_DIR/configs/$MASKS" --start $(( START + w * PER )) --count $(( w == WORKERS - 1 ? CHUNK - w * PER : PER )) \
    --seed "$SEED" > "$D24_DIR/gpu/logs/worker-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_w$w.log" 2>&1 &
done
wait
