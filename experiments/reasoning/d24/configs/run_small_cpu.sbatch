#!/bin/bash
#SBATCH --job-name=d24_small
#SBATCH --partition=long
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=experiments/reasoning/d24/gpu/logs/%x-%j.out
#SBATCH --error=experiments/reasoning/d24/gpu/logs/%x-%j.err
# Usage: sbatch configs/run_small_cpu.sbatch <script> [args...]; then scontrol update JobId=<id> Partition=long
REPO="${TNDM_ROOT:-${SLURM_SUBMIT_DIR:-$(git rev-parse --show-toplevel)}}"  # repository root
set -euo pipefail
cd $REPO/experiments/reasoning/d24
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1
${PYTHON:-python} -u "$@"
