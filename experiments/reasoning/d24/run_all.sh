#!/bin/bash
# Reproduce d=24 study.
# Phase 0 (analytic pilot, CPU):
#   bash run_all.sh submit     # calibrate, freeze arm list, submit the Slurm array
#   bash run_all.sh aggregate  # after all 144 arms exist: tables, decision, figures
# Targeted GPU arm (Amendment A), run in order:
#   python pilot/../gpu/freeze_design.py
#   sbatch --job-name=d24_gate --array=0 --export=ALL,MASKS=gpu_gate_masks.txt,SEED=31,CHUNK=45,WORKERS=4 configs/run_gpu_cells.sbatch
#   python gpu/evaluate_gate.py 31                         # must print PASS
#   for s in 31 43 59; do sbatch --array=0-31 --cpus-per-task=2 --mem=8G --time=00:40:00 \
#       --export=ALL,MASKS=gpu_training_masks.txt,SEED=$s,CHUNK=32,WORKERS=1 configs/run_gpu_cells.sbatch; done
#   sbatch --array=0-8 configs/run_gpu_fit.sbatch          # freezes predictions; refuses if evaluation cells exist
#   sbatch configs/run_small_cpu.sbatch gpu/analyze.py confirm-list
#   for s in 31 43 59; do sbatch --array=0-9 --cpus-per-task=2 --mem=8G --time=00:40:00 \
#       --export=ALL,MASKS=gpu_evaluation_masks.txt,SEED=$s,CHUNK=32,WORKERS=1 configs/run_gpu_cells.sbatch; done
#   sbatch --array=0-3 --cpus-per-task=2 --mem=8G --time=00:40:00 \
#       --export=ALL,MASKS=gpu_confirmation_masks.txt,SEED=31,CHUNK=26,WORKERS=1 configs/run_gpu_cells.sbatch
#   sbatch configs/run_final_analysis.sbatch               # verify cells, primary test, figure
#   sbatch configs/run_small_cpu.sbatch gpu/export_cube.py
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} PYTHONNOUSERSITE=1
PY=${PYTHON:-python}
case "${1:-aggregate}" in
  submit)
    "$PY" pilot/calibrate_from_d12.py
    "$PY" pilot/test_kernel_equivalence.py
    "$PY" pilot/make_arm_list.py
    mkdir -p pilot/logs pilot/results
    sbatch --array=0-$(( $(wc -l < configs/pilot_arms.txt) - 1 ))%48 configs/run_pilot.sbatch
    ;;
  aggregate)
    "$PY" pilot/aggregate.py
    "$PY" pilot/plot_pilot.py
    ;;
  *) echo "usage: $0 submit|aggregate" >&2; exit 2 ;;
esac
