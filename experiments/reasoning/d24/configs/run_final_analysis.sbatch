#!/bin/bash
#SBATCH --job-name=d24_final_analysis
#SBATCH --partition=long
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:20:00
#SBATCH --output=experiments/reasoning/d24/gpu/logs/%x-%j.out
#SBATCH --error=experiments/reasoning/d24/gpu/logs/%x-%j.err
REPO="${TNDM_ROOT:-${SLURM_SUBMIT_DIR:-$(git rev-parse --show-toplevel)}}"  # repository root
set -euo pipefail
cd $REPO/experiments/reasoning/d24
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1
PY=${PYTHON:-python}
$PY gpu/verify_cells.py gpu_evaluation_masks.txt 31,43,59
$PY gpu/verify_cells.py gpu_confirmation_masks.txt 31
$PY gpu/analyze.py analyze
$PY gpu/plot_gpu.py
