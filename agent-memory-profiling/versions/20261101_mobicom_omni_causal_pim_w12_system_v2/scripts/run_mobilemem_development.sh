#!/usr/bin/env bash
set -euo pipefail
VERSION_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$VERSION_ROOT"
PYTHON="${PYTHON:-$VERSION_ROOT/../../.venvs/flamr-qwen35/bin/python}"
RUN_ROOT="${RUN_ROOT:-artifacts/causal_dev16_v1_20261018}"
read -r -a DEV_GPUS <<< "${GPUS:-0}"
if [[ "${#DEV_GPUS[@]}" != 1 ]]; then
    printf 'Mainline requires one chronological persistent-device stream, not cold shards.\n' >&2
    exit 1
fi
if [[ -e "$RUN_ROOT" ]]; then
    printf 'Refusing to overwrite %s\n' "$RUN_ROOT" >&2
    exit 1
fi
mkdir -p "$RUN_ROOT/logs"
pids=()
for index in "${!DEV_GPUS[@]}"; do
    validation_flag=()
    if [[ "${VALIDATE_PREFIX:-1}" == "1" ]]; then
        validation_flag+=(--validate-prefix)
    fi
    if [[ "${CROSS_QUERY_PREFIX:-0}" == "1" ]]; then
        validation_flag+=(--cross-query-prefix)
    fi
    CUDA_VISIBLE_DEVICES="${DEV_GPUS[$index]}" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
        "$PYTHON" scripts/run_mobilemem_text.py \
        --prepared "${PREPARED:-artifacts/mobilemem_causal_prepare_v1}" \
        --output "$RUN_ROOT/part_$index" --limit "${LIMIT:-16}" --sample-count "${SAMPLE_COUNT:-0}" \
        --shard-count "${#DEV_GPUS[@]}" --shard-index "$index" "${validation_flag[@]}" \
        --reader-protocol "${READER_PROTOCOL:-single}" \
        --prefix-audit-every "${PREFIX_AUDIT_EVERY:-1}" \
        >"$RUN_ROOT/logs/part_$index.log" 2>&1 &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        status=1
    fi
done
exit "$status"
