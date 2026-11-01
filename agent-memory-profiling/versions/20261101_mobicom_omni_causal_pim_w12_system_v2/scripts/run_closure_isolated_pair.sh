#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PRETEST=$(cd "$ROOT/../../.." && pwd)
PYTHON="$PRETEST/agent-memory-profiling/.venvs/flamr-qwen35/bin/python"
RUN_ROOT="${RUN_ROOT:-$ROOT/artifacts/provenance_isolated_pair_dev3x16_v1_20261021}"
COHORT="${COHORT:-$ROOT/artifacts/hash_cohort_dev3x16_v1_20261021.json}"
PER_USER="${PER_USER:-16}"
RELEVANCE_MODALITY="${RELEVANCE_MODALITY:-text}"
QUALITY_SUMMARY="${QUALITY_SUMMARY:-}"
if [ -z "$QUALITY_SUMMARY" ]; then
    echo 'QUALITY_SUMMARY required; do not run final latency on a quality-failing candidate' >&2
    exit 2
fi
"$PYTHON" - "$QUALITY_SUMMARY" "$COHORT" "$RELEVANCE_MODALITY" <<'PY'
import json
from pathlib import Path
import sys

summary = json.loads(Path(sys.argv[1]).read_text())
if not summary.get('complete') or not summary['quality_screen']['no_proxy_score_loss']:
    raise SystemExit('development quality screen failed; no final latency promotion')
if summary.get('cohort_sha256') is None or summary.get('candidate_relevance_modality') != sys.argv[3]:
    raise SystemExit('quality screen lacks matching modality/cohort provenance')
import hashlib
if hashlib.sha256(Path(sys.argv[2]).read_bytes()).hexdigest() != summary['cohort_sha256']:
    raise SystemExit('different cohort than quality screen')
PY
if [ -e "$RUN_ROOT" ]; then
    echo "Fresh RUN_ROOT required" >&2
    exit 2
fi
mkdir -p "$RUN_ROOT"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
for repeat in 0 1; do
    mkdir "$RUN_ROOT/repeat_${repeat}"
    for user in 0 1 2; do
        policies=(local_hyperedge provenance_milp)
        if [ "$repeat" = 1 ]; then policies=(provenance_milp local_hyperedge); fi
        for policy in "${policies[@]}"; do
            relevance_args=()
            if [ "$policy" = provenance_milp ]; then
                relevance_args+=(--relevance-modality "$RELEVANCE_MODALITY")
            fi
            CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$ROOT/scripts/run_omni_raw_image_smoke.py" \
                --prepared "$ROOT/artifacts/omni_paper_scope_v3_20261021/prepared" \
                --image-archive "$PRETEST/MobileMem-main/data/omni/image.zip" \
                --output "$RUN_ROOT/repeat_${repeat}/${policy}_user_${user}.json" \
                --user-id "$user" --limit "$PER_USER" --cohort "$COHORT" \
                --evidence-policy "$policy" --disable-edge-reuse "${relevance_args[@]}" \
                --candidate-sessions 4 --total-text-tokens 3600 --max-images 2 --max-new-tokens 192 \
                --hardware-identity RTX3090:0 \
                --concurrency-contract "one timed serving worker on fixed GPU0; no simultaneous serving/index workers; judge may run GPU6; shared-host telemetry recorded separately" \
                >"$RUN_ROOT/repeat_${repeat}/${policy}_user_${user}.log" 2>&1
        done
    done
done
