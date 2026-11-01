#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ARTIFACT="${ARTIFACT:-$ROOT/artifacts/proof_recovery_dev3x64_contract_v2_20261025}"
LOCATOR_ROOT="${LOCATOR_ROOT:-$ROOT/artifacts/proof_recovery_dev3x64_v1_20261025}"
PYTHON="${PYTHON:-$ROOT/../../.venvs/flamr-qwen35/bin/python}"
PREPARED="$ROOT/artifacts/omni_paper_scope_v3_20261021/prepared"
ARCHIVE="$ROOT/../../../MobileMem-main/data/omni/image.zip"
EMBEDDER="${EMBEDDER:-/data2/user25263093/.cache/modelscope/hub/models/Qwen/Qwen3-Embedding-0.6B}"
RERANKER="${RERANKER:-/data2/user25263093/.cache/modelscope/hub/models/Qwen/Qwen3-Reranker-0.6B}"
READER="${READER:-/data2/user25263093/.cache/modelscope/hub/models/Qwen/Qwen3___5-4B}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
CODE="$ARTIFACT/runtime_mainline"
mkdir "$CODE"
cp -a "$ROOT/scripts" "$ROOT/pim_backend" "$CODE/"
"$PYTHON" "$CODE/scripts/register_proof_recovery.py" --root "$ARTIFACT" --prepared "$PREPARED" \
  --reader "$READER" --embedder "$EMBEDDER" --reranker "$RERANKER"

"$PYTHON" "$CODE/scripts/merge_omni_visual_locators.py" --root "$LOCATOR_ROOT" --output "$ARTIFACT/visual_merged"
jobs=()
for user in 0 1 2; do
  CUDA_VISIBLE_DEVICES="$user" "$PYTHON" -u "$CODE/scripts/build_omni_proof_index.py" \
    --prepared "$PREPARED" --locator-root "$ARTIFACT/visual_merged" \
    --model "$EMBEDDER" --version-root "$ROOT" --user-id "$user" \
    --output "$ARTIFACT/proof_index_user_$user" >"$ARTIFACT/proof_index_user_$user.log" 2>&1 &
  jobs+=("$!")
done
for process in "${jobs[@]}"; do wait "$process"; done
jobs=()
for gpu in 0 1 2 3 4 5 6; do
  user=$((gpu % 3))
  shard=$((gpu / 3))
  count=2
  if [ "$user" = 0 ]; then count=3; fi
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u "$CODE/scripts/run_proof_mainline.py" \
    --prepared "$PREPARED" --image-archive "$ARCHIVE" --reader "$READER" \
    --embedder "$EMBEDDER" --reranker "$RERANKER" \
    --index "$ARTIFACT/proof_index_user_$user" --cohort "$ARTIFACT/cohort.json" \
    --user-id "$user" --shard-index "$shard" --shard-count "$count" \
    --output "$ARTIFACT/proof_user_${user}_shard_${shard}.json" \
    --input-budget 8192 --total-text-tokens 3600 --max-images 4 \
    --max-new-tokens 512 --planner-time-limit 2 >"$ARTIFACT/proof_user_${user}_shard_${shard}.log" 2>&1 &
  jobs+=("$!")
done
for process in "${jobs[@]}"; do wait "$process"; done
"$PYTHON" "$CODE/scripts/merge_proof_recovery_runs.py" --root "$ARTIFACT"
CUDA_VISIBLE_DEVICES=3 "$PYTHON" -u "$CODE/scripts/judge_omni_closure.py" \
  --prepared "$PREPARED" --runs "$ARTIFACT/proof_user_0.json" "$ARTIFACT/proof_user_1.json" "$ARTIFACT/proof_user_2.json" \
  --prompt-source "$ROOT/../../../MobileMem-main/omni/eval/eval/question_answering_and_judge_prompts.txt" \
  --output "$ARTIFACT/judge.json" >"$ARTIFACT/judge.log" 2>&1
"$PYTHON" "$CODE/scripts/audit_proof_recovery_closure.py" --root "$ARTIFACT" --prepared "$PREPARED" --cohort "$ARTIFACT/cohort.json"
