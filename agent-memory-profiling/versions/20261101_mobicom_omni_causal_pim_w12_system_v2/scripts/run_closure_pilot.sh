#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PROJECT=$(cd "$ROOT/../../.." && pwd)
PYTHON="$PROJECT/agent-memory-profiling/.venvs/flamr-qwen35/bin/python"
PRETEST="$PROJECT"
PREPARED="$ROOT/artifacts/omni_paper_scope_v3_20261021/prepared"
RUN_ROOT="${RUN_ROOT:-$ROOT/artifacts/provenance_pilot_paper_dev3x4_v2_20261021}"
PER_USER="${PER_USER:-4}"
DISABLE_EDGE_REUSE="${DISABLE_EDGE_REUSE:-1}"
COHORT="${COHORT:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-192}"
RELEVANCE_MODALITY="${RELEVANCE_MODALITY:-text}"
RELEVANCE_BATCH_SIZE="${RELEVANCE_BATCH_SIZE:-1}"
if [ -e "$RUN_ROOT" ]; then
    echo "Fresh RUN_ROOT required: $RUN_ROOT" >&2
    exit 2
fi
mkdir -p "$RUN_ROOT"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
pids=()
for user in 0 1 2; do
    for policy in local_hyperedge provenance_milp; do
        gpu=$user
        if [ "$policy" = provenance_milp ]; then gpu=$((user + 3)); fi
        reuse_args=()
        if [ "$DISABLE_EDGE_REUSE" = "1" ]; then reuse_args+=(--disable-edge-reuse); fi
        if [ -n "$COHORT" ]; then reuse_args+=(--cohort "$COHORT"); fi
        if [ "$policy" = provenance_milp ]; then
            reuse_args+=(--relevance-modality "$RELEVANCE_MODALITY" --relevance-batch-size "$RELEVANCE_BATCH_SIZE")
        fi
        CUDA_VISIBLE_DEVICES=$gpu "$PYTHON" "$ROOT/scripts/run_omni_raw_image_smoke.py" \
            --prepared "$PREPARED" --image-archive "$PRETEST/MobileMem-main/data/omni/image.zip" \
            --output "$RUN_ROOT/${policy}_user_${user}.json" --user-id "$user" --limit "$PER_USER" \
            --evidence-policy "$policy" "${reuse_args[@]}" --candidate-sessions 4 \
            --total-text-tokens 3600 --max-images 2 --max-new-tokens "$MAX_NEW_TOKENS" --screening-run \
            --hardware-identity "RTX3090:${gpu}" \
            --concurrency-contract "six isolated GPU workers; shared CPU/storage; screening only, not isolated final latency" \
            >"$RUN_ROOT/${policy}_user_${user}.log" 2>&1 &
        pids+=("$!")
    done
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit "$status"
