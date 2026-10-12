#!/bin/bash
#SBATCH --job-name=d24_aggregate
#SBATCH --partition=long
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=experiments/reasoning/d24/pilot/logs/aggregate-%j.out
#SBATCH --error=experiments/reasoning/d24/pilot/logs/aggregate-%j.err
REPO="${TNDM_ROOT:-${SLURM_SUBMIT_DIR:-$(git rev-parse --show-toplevel)}}"  # repository root
set -euo pipefail
cd $REPO/experiments/reasoning/d24
bash run_all.sh aggregate
