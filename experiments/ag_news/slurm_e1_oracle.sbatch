#!/usr/bin/env bash
#SBATCH --job-name=tndm-e1-d8
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:4
#SBATCH --time=03:00:00
#SBATCH --output=experiments/ag_news/results/data_dependency_attribution/E1_adapter_rank/logs/%x-%A_%a.out
#SBATCH --error=experiments/ag_news/results/data_dependency_attribution/E1_adapter_rank/logs/%x-%A_%a.err

set -euo pipefail
: "${TNDM_REPO:?Set TNDM_REPO to the repository root}"
: "${TNDM_E1_MANIFEST:?Set TNDM_E1_MANIFEST to oracle_manifest.jsonl}"

cd "$TNDM_REPO"
mapfile -t identities < "$TNDM_E1_MANIFEST"
IFS=',' read -ra devices <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
identity_offset="${TNDM_E1_OFFSET:-0}"
if ! [[ "$identity_offset" =~ ^[0-9]+$ ]] || (( identity_offset % 4 != 0 )); then
  echo "TNDM_E1_OFFSET must be a nonnegative multiple of four" >&2
  exit 64
fi
start=$((identity_offset + SLURM_ARRAY_TASK_ID * 4))
pids=()

for offset in 0 1 2 3; do
  position=$((start + offset))
  if (( position >= ${#identities[@]} )); then
    continue
  fi
  parsed="$("${PYTHON:-python}" -c 'import json,sys; x=json.loads(sys.argv[1]); print(x["mask"], x["learner_seed"], x["config"], x["output_root"])' "${identities[$position]}")"
  read -r mask seed config output_root <<< "$parsed"
  device="${devices[$offset]:-$offset}"
  (
    export CUDA_VISIBLE_DEVICES="$device"
    export PYTHONPATH="$TNDM_REPO/src:$TNDM_REPO"
    export HF_HOME="$TNDM_REPO/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    export TOKENIZERS_PARALLELISM=false
    "${PYTHON:-python}" -u experiments/ag_news/run_transformer_coalition.py \
      --mask "$mask" --seed "$seed" --config "$config" \
      --output-root "$output_root" \
      --prepared-data experiments/ag_news/results/raw/transformer/data/ag_news_prepared.json \
      --hf-cache "$TNDM_REPO/hf_cache" --device cuda
  ) &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then status=1; fi
done
exit "$status"
