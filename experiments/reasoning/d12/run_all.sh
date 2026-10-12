#!/bin/bash
# d=12 study: reproduce the d=12 replication end to end.
#
# Every stage is idempotent: cells that already exist are not retrained, and the fitter
# reuses a frozen artifact when its source hashes are unchanged.  The stages must run in
# this order, because the freeze order is what makes the result pre-registered rather
# than post-hoc.
REPO="${TNDM_ROOT:-${SLURM_SUBMIT_DIR:-$(git rev-parse --show-toplevel)}}"  # repository root
set -euo pipefail
WS=$REPO
HERE="$WS/experiments/reasoning/d12"
CONFIGS="$HERE/configs"
RESULTS="$HERE/results"
CODE="$HERE/code"
PY=${PYTHON:-python}
export PYTHONPATH="$WS:$WS/src"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg
PREREG=${TNDM_PROTOCOL_SHA256:-unrecorded}  # recorded with every cell; set to the hash of your protocol note
CONFIRMATORY_SEEDS="43 59"
ALL_SEEDS="31 43 59"
REPORTED_GPU=rtx8000

submit () {  # submit <name> <masks-file> <out-dir> <seed> <chunk> [extra exports]
  local name=$1 masks=$2 out=$3 seed=$4 chunk=$5; shift 5
  local total; total=$(wc -l < "$masks")
  local tasks=$(( (total + chunk - 1) / chunk - 1 ))
  sbatch --parsable --array="0-${tasks}%16" --job-name="d12_${name}" \
    --output="$HERE/logs/${name}-%A_%a.out" --error="$HERE/logs/${name}-%A_%a.err" \
    --export=ALL,MASKS_FILE="$masks",OUT_DIR="$out",SEED="$seed",CHUNK="$chunk",PREREG_SHA="$PREREG",REQUIRE_GPU="$REPORTED_GPU","$@" \
    "$CODE/run_cells.sbatch"
}

stage=${1:-all}

# 1 -- manifests (no compute)
if [ "$stage" = all ] || [ "$stage" = configs ]; then
  $PY "$CODE/make_configs.py"
fi

# 2 -- learner gate for the confirmatory seeds
if [ "$stage" = all ] || [ "$stage" = gate ]; then
  for seed in $CONFIRMATORY_SEEDS; do
    submit "gate$seed" "$CONFIGS/gate_masks.txt" "$RESULTS/gate/seed$seed" "$seed" 5
  done
  echo "wait for the gate array, then: bash run_all.sh gate-decision"
fi

if [ "$stage" = gate-decision ]; then
  for seed in $CONFIRMATORY_SEEDS; do
    $PY -m experiments.reasoning.common.evaluate_query_paths_d12_gate \
      --input-dir "$RESULTS/gate/seed$seed" --output "$RESULTS/gate/seed$seed/GATE_DECISION.json"
  done
fi

# 3 -- the frozen sparse design, and the leave-one-group-out influence cells
if [ "$stage" = all ] || [ "$stage" = observed ]; then
  for seed in $CONFIRMATORY_SEEDS; do
    submit "obs$seed" "$CONFIGS/new_observed_masks.txt" "$RESULTS/observed/seed$seed" "$seed" 20
  done
  for seed in $ALL_SEEDS; do
    submit "loo$seed" "$CONFIGS/loo_masks.txt" "$RESULTS/loo/seed$seed" "$seed" 4
  done
fi

# 4 -- fit and freeze, every budget, every seed
if [ "$stage" = all ] || [ "$stage" = fit ]; then
  R15="${TNDM_D12_DEV_SEED31:-$REPO/experiments/reasoning/d12/results/dev_seed31}"  # seed-31 development cells (not shipped; see README)
  for seed in $ALL_SEEDS; do
    extra=""
    [ "$seed" = 31 ] && extra="--fallback-root $R15 --nonblind"
    for budget in 32 48 64 96 128 160; do
      $PY "$CODE/fit_d12.py" --seed "$seed" --new-budget "$budget" --root "$RESULTS" \
        --preregistration-sha256 "$PREREG" $extra
    done
  done
fi

# 5 -- freeze the slice nominations, then retrain them
if [ "$stage" = all ] || [ "$stage" = freeze-slices ]; then
  for seed in $ALL_SEEDS; do
    $PY "$CODE/freeze_slices.py" --seed "$seed" --root "$RESULTS" --configs "$CONFIGS"
  done
fi

if [ "$stage" = all ] || [ "$stage" = slices ]; then
  for seed in $ALL_SEEDS; do
    for size in 4 5 6; do
      submit "s${size}_$seed" "$CONFIGS/slice${size}_exec_masks_seed${seed}.txt" \
        "$RESULTS/size${size}_eval/seed$seed" "$seed" 20
    done
  done
fi

# 6 -- the exhaustive size-3 slice and the 14 pair-audit cells, confirmatory seeds only
if [ "$stage" = all ] || [ "$stage" = target ]; then
  for seed in $CONFIRMATORY_SEEDS; do
    submit "tgt$seed" "$CONFIGS/frozen_target_masks.txt" "$RESULTS/target/seed$seed" "$seed" 20
    submit "aud$seed" "$CONFIGS/interaction_audit_masks.txt" "$RESULTS/interaction_audit/seed$seed" "$seed" 7
  done
fi

# 7 -- noise ceiling (independent of stages 2-6)
if [ "$stage" = all ] || [ "$stage" = ceiling ]; then
  for seed in 31 43; do
    submit "cbase$seed" "$CONFIGS/ceiling_masks.txt" "$RESULTS/ceiling/seed$seed/base"  "$seed" 20 REPLICATE=base
    submit "crer$seed"  "$CONFIGS/ceiling_masks.txt" "$RESULTS/ceiling/seed$seed/rerun" "$seed" 20 REPLICATE=rerun
    submit "cin1$seed"  "$CONFIGS/ceiling_masks.txt" "$RESULTS/ceiling/seed$seed/init1" "$seed" 20 REPLICATE=init1,INIT_SEED=$((seed+101)),DATA_SEED=$seed
    submit "cin2$seed"  "$CONFIGS/ceiling_masks.txt" "$RESULTS/ceiling/seed$seed/init2" "$seed" 20 REPLICATE=init2,INIT_SEED=$((seed+102)),DATA_SEED=$seed
    submit "cda1$seed"  "$CONFIGS/ceiling_masks.txt" "$RESULTS/ceiling/seed$seed/data1" "$seed" 20 REPLICATE=data1,INIT_SEED=$seed,DATA_SEED=$((seed+201))
    submit "cda2$seed"  "$CONFIGS/ceiling_masks.txt" "$RESULTS/ceiling/seed$seed/data2" "$seed" 20 REPLICATE=data2,INIT_SEED=$seed,DATA_SEED=$((seed+202))
  done
fi

if [ "$stage" = all ] || [ "$stage" = ceiling-backend ]; then
  # the two deliberately cross-backend arms; these must not use --require-gpu rtx8000
  for seed in 31 43; do
    for gpu in v100 a6000; do
      total=$(wc -l < "$CONFIGS/ceiling_masks.txt")
      sbatch --parsable --array="0-$(( (total + 19) / 20 - 1 ))%4" --job-name="d12_c${gpu}${seed}" \
        --gres="gpu:${gpu}:1" \
        --output="$HERE/logs/c${gpu}${seed}-%A_%a.out" --error="$HERE/logs/c${gpu}${seed}-%A_%a.err" \
        --export=ALL,MASKS_FILE="$CONFIGS/ceiling_masks.txt",OUT_DIR="$RESULTS/ceiling/seed$seed/$gpu",SEED="$seed",CHUNK=20,PREREG_SHA="$PREREG",REQUIRE_GPU="$gpu",REPLICATE="$gpu" \
        "$CODE/run_cells.sbatch"
    done
  done
fi

# 8 -- analysis, tables, figures (light; the size-3 analysis is a batch job)
R15="${TNDM_D12_DEV_SEED31:-$REPO/experiments/reasoning/d12/results/dev_seed31}"  # seed-31 development cells (not shipped; see README)
if [ "$stage" = all ] || [ "$stage" = analyse ]; then
  $PY "$CODE/analyze_ceiling.py" --root "$RESULTS" --configs "$CONFIGS" --out "$HERE/noise_ceiling"
  $PY "$CODE/analyze_slices.py" --root "$RESULTS" --protocols "$HERE" --out "$HERE/analysis" \
    --ceiling "$HERE/noise_ceiling/ceiling.json" --seeds $ALL_SEEDS \
    --confirmatory-seeds $CONFIRMATORY_SEEDS --slices 3 4 5 6 --fallback-root "$R15"
  echo "size-3 analysis: sbatch --export=ALL,SEEDS=\"$ALL_SEEDS\" $CODE/run_analysis.sbatch, then: bash run_all.sh report"
fi
if [ "$stage" = all ] || [ "$stage" = report ]; then
  $PY "$CODE/interaction_contrasts.py" --analysis "$HERE/analysis"
  $PY "$CODE/make_tables_figures.py" --here "$HERE"
  $PY "$CODE/export_cubes.py" --here "$HERE"
fi
