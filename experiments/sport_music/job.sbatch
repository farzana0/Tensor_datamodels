#!/bin/bash
#SBATCH --job-name=tndm_sm_coalitions
#SBATCH --no-requeue
#SBATCH --gres=gpu:1
#SBATCH --constraint=80gb
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=11:55:00
# One full-parameter Qwen2-7B fine-tune per coalition (about 3 GPU-minutes each on an
# 80 GB A100).  Submit from the repository root.  Requires TNDM_QWEN2_7B (local model
# snapshot) and the reconstructed dataset under experiments/sport_music/dataset/data.
set -euo pipefail
REPO="${TNDM_ROOT:-${SLURM_SUBMIT_DIR:-$(git rev-parse --show-toplevel)}}"
cd "$REPO"
: "${TNDM_QWEN2_7B:?Set TNDM_QWEN2_7B to the local Qwen/Qwen2-7B snapshot directory}"
export PYTHONPATH="$REPO/src:$REPO"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
exec "${PYTHON:-python}" experiments/sport_music/run.py "$@"
