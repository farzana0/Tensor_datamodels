#!/bin/bash
#SBATCH --job-name=tndm_sm_fitcompute
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
# CPU only: refits every datamodel on the saved coalition utilities (Table 2 / Table 13).
set -euo pipefail
REPO="${TNDM_ROOT:-${SLURM_SUBMIT_DIR:-$(git rev-parse --show-toplevel)}}"
cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO" OMP_NUM_THREADS=8
exec "${PYTHON:-python}" experiments/sport_music/fitting_compute.py
