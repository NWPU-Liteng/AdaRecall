#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PRETEST=$(cd "$ROOT/../../.." && pwd)
PYTHON="$ROOT/../../.venvs/flamr-qwen35/bin/python"
MODEL_ROOT="$ROOT/../../models"
PREPARED="$ROOT/artifacts/omni_paper_scope_v3_20261021/prepared"
COHORT="$ROOT/artifacts/hash_cohort_dev3x16_v1_20261021.json"
RUN_ROOT="${RUN_ROOT:-$ROOT/artifacts/joint_visual_closure_dev3x16_v1_20261021}"
if [ -e "$RUN_ROOT" ]; then
    echo 'Fresh RUN_ROOT required' >&2
    exit 2
fi
mkdir -p "$RUN_ROOT/runtime" "$RUN_ROOT/smoke"
cp -a "$ROOT/scripts" "$ROOT/pim_backend" "$RUN_ROOT/runtime/"
"$PYTHON" - "$RUN_ROOT" "$COHORT" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

root, cohort = map(Path, sys.argv[1:])
sources = {str(path.relative_to(root / 'runtime')): hashlib.sha256(path.read_bytes()).hexdigest()
           for path in sorted((root / 'runtime').rglob('*')) if path.suffix in ('.py', '.json', '.sh')}
protocol = {'status': 'registered_before_generation', 'sources_sha256': sources,
    'cohort_sha256': hashlib.sha256(cohort.read_bytes()).hexdigest(),
    'reader': 'Qwen3.5-4B', 'judge': 'Qwen3-VL-8B-Instruct',
    'max_images': 4, 'max_new_tokens': 512, 'input_budget': 8192, 'text_budget': 3600,
    'visual_recall_count': 16, 'text_recall_count': 16, 'index_image_max_pixels': 65536,
    'reader_image_max_pixels': 262144, 'reranker_image_max_pixels': 262144,
    'selection': 'two independently scored recall branches; unique union; trained multimodal logits; provenance MILP',
    'boundaries': ['development screening, not sealed-test generalization',
        'different isolated GPUs, shared CPU/storage; no causal latency claim',
        'producer construction separate; prebuilt index lookup is not Reader KV reuse',
        'both policies changed from old 2-image/192-output diagnostic; no cross-run speedup claim',
        'no PIM functional replacement or projected E2E mixed with measured GPU time']}
(root / 'protocol.json').write_text(json.dumps(protocol, indent=2))
PY
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
RUNTIME="$RUN_ROOT/runtime/scripts"
pids=()
for user in 0 1 2; do
    gpu=$((user + 3))
    CUDA_VISIBLE_DEVICES=$gpu "$PYTHON" "$RUNTIME/run_omni_raw_image_smoke.py" \
        --prepared "$PREPARED" --image-archive "$PRETEST/MobileMem-main/data/omni/image.zip" \
        --output "$RUN_ROOT/local_hyperedge_user_${user}.json" --user-id "$user" --limit 16 \
        --cohort "$COHORT" --evidence-policy local_hyperedge --disable-edge-reuse \
        --candidate-sessions 4 --total-text-tokens 3600 --max-images 4 --max-new-tokens 512 \
        --input-budget 8192 --screening-run --hardware-identity "RTX3090:$gpu" \
        --concurrency-contract 'isolated GPUs; shared CPU/storage and producer construction; screening only' \
        >"$RUN_ROOT/local_hyperedge_user_${user}.log" 2>&1 &
    pids+=("$!")
done
for user in 0 1 2; do
    (
        index="$ROOT/artifacts/joint_visual_index_dev_user_${user}_v2_20261021"
        deadline=$((SECONDS + 14400))
        until [ -f "$index/index.json" ]; do
            if [ "$SECONDS" -ge "$deadline" ]; then
                echo "Index construction timeout: $index" >&2
                exit 2
            fi
            sleep 30
        done
        "$PYTHON" - "$index/index.json" <<'PY'
import json
from pathlib import Path
import sys
if not json.loads(Path(sys.argv[1]).read_text()).get('complete'):
    raise SystemExit('incomplete producer index')
PY
        common=(--prepared "$PREPARED" --image-archive "$PRETEST/MobileMem-main/data/omni/image.zip"
            --user-id "$user" --evidence-policy provenance_milp --disable-edge-reuse
            --relevance-modality joint_index --joint-index "$index" --joint-index-image-max-pixels 65536
            --joint-embedding-model "$MODEL_ROOT/Qwen3-VL-Embedding-2B"
            --joint-reranker-model "$MODEL_ROOT/Qwen3-VL-Reranker-2B"
            --relevance-batch-size 2 --joint-recall-count 16 --joint-text-recall-count 16
            --candidate-sessions 4 --total-text-tokens 3600 --max-images 4
            --input-budget 8192 --screening-run --hardware-identity "RTX3090:$user"
            --concurrency-contract 'isolated GPUs; shared CPU/storage; screening only')
        CUDA_VISIBLE_DEVICES=$user "$PYTHON" "$RUNTIME/run_omni_raw_image_smoke.py" \
            "${common[@]}" --output "$RUN_ROOT/smoke/provenance_milp_user_${user}.json" \
            --limit 1 --max-new-tokens 8 >"$RUN_ROOT/smoke/provenance_milp_user_${user}.log" 2>&1
        CUDA_VISIBLE_DEVICES=$user "$PYTHON" "$RUNTIME/run_omni_raw_image_smoke.py" \
            "${common[@]}" --output "$RUN_ROOT/provenance_milp_user_${user}.json" \
            --limit 16 --cohort "$COHORT" --max-new-tokens 512 \
            >"$RUN_ROOT/provenance_milp_user_${user}.log" 2>&1
    ) &
    pids+=("$!")
done
status=0
for process in "${pids[@]}"; do wait "$process" || status=1; done
if [ "$status" != 0 ]; then
    echo 'Generation incomplete; no survivor-only quality evaluation' >&2
    exit "$status"
fi
runs=()
for user in 0 1 2; do
    runs+=("$RUN_ROOT/local_hyperedge_user_${user}.json" "$RUN_ROOT/provenance_milp_user_${user}.json")
done
CUDA_VISIBLE_DEVICES=6 "$PYTHON" "$RUNTIME/judge_omni_closure.py" \
    --prepared "$PREPARED" --runs "${runs[@]}" \
    --prompt-source "$PRETEST/MobileMem-main/omni/eval/eval/question_answering_and_judge_prompts.txt" \
    --output "$RUN_ROOT/qwen3vl8b_judge.json" --max-new-tokens 512 >"$RUN_ROOT/judge.log" 2>&1
"$PYTHON" "$RUNTIME/summarize_closure_pilot.py" --run-root "$RUN_ROOT" \
    --judge "$RUN_ROOT/qwen3vl8b_judge.json" --output "$RUN_ROOT/summary.json" >"$RUN_ROOT/summary.log"
"$PYTHON" "$RUNTIME/diagnose_closure_evidence.py" --prepared "$PREPARED" \
    --runs "${runs[@]}" \
    --judge "$RUN_ROOT/qwen3vl8b_judge.json" --output "$RUN_ROOT/evidence_diagnosis.json"
