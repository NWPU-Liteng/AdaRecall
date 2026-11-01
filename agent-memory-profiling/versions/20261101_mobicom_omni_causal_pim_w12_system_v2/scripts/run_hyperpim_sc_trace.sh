#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PRETEST="$(cd "$ROOT/../../.." && pwd)"
PYTHON="${PYTHON:-$ROOT/../../.venvs/flamr-qwen35/bin/python}"
GPUS="${GPUS:-0 1 2 3}"
OUT="${OUT:-$ROOT/artifacts/hyperpim_sc_trace_v1}"
READER="${READER:-/data2/user25263093/.cache/modelscope/hub/models/Qwen/Qwen3___5-4B}"
EMBEDDER="${EMBEDDER:-/data2/user25263093/.cache/modelscope/hub/models/Qwen/Qwen3-Embedding-0___6B}"
RERANKER="${RERANKER:-/data2/user25263093/.cache/modelscope/hub/models/Qwen/Qwen3-Reranker-0___6B}"

export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$ROOT:$ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$OUT/logs"
read -r -a gpu_list <<<"$GPUS"
if (( ${#gpu_list[@]} < 4 )); then
  echo "four GPUs are required" >&2
  exit 2
fi

pids=()
for user in 0 1 2 3; do
  output="$OUT/user_${user}.json"
  if [[ -e "$output" || -e "${output%.json}.partial.json" ]]; then
    echo "refusing to overwrite existing trace output for user $user" >&2
    exit 2
  fi
  CUDA_VISIBLE_DEVICES="${gpu_list[$user]}" "$PYTHON" -u "$ROOT/scripts/run_proof_mainline.py" \
    --prepared "$ROOT/artifacts/main_experiment_20261026/prepared" \
    --image-archive "$PRETEST/MobileMem-main/data/omni/image.zip" \
    --reader "$READER" --embedder "$EMBEDDER" --reranker "$RERANKER" \
    --index "$ROOT/artifacts/main_experiment_20261026/indexes_final/proof_index_user_${user}" \
    --cohort "$ROOT/artifacts/main_experiment_20261026/cohort_200_pilot.json" \
    --user-id "$user" --query-contract --pre-reader-operator-profile \
    --question-ids-file "$OUT/question_ids/user_${user}.json" \
    --input-budget 8192 --total-text-tokens 3600 --max-images 4 \
    --max-new-tokens 512 --planner-time-limit 2 \
    --output "$output" >"$OUT/logs/user_${user}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
exit "$status"
