#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PRETEST=$(cd "$ROOT/../../.." && pwd)
PYTHON="$ROOT/../../.venvs/flamr-qwen35/bin/python"
MODEL_ROOT="$ROOT/../../models"
PREPARED="$ROOT/artifacts/omni_paper_scope_v3_20261021/prepared"
COHORT="$ROOT/artifacts/hash_cohort_dev3x16_v1_20261021.json"
RUN_ROOT="${RUN_ROOT:-$ROOT/artifacts/joint_visual_closure_dev3x16_v3_sharded_20261021}"
IMAGE_ARCHIVE="$PRETEST/MobileMem-main/data/omni/image.zip"
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
           for path in sorted((root / 'runtime').rglob('*')) if path.is_file() and path.suffix in ('.py', '.json', '.sh')}
protocol = {'status': 'registered_before_generation', 'sources_sha256': sources,
    'cohort_sha256': hashlib.sha256(cohort.read_bytes()).hexdigest(),
    'reader': 'Qwen3.5-4B', 'judge': 'Qwen3-VL-8B-Instruct',
    'max_images': 4, 'max_new_tokens': 512, 'input_budget': 8192, 'text_budget': 3600,
    'visual_recall_count': 16, 'text_recall_count': 16, 'index_image_max_pixels': 65536,
    'reader_image_max_pixels': 262144, 'reranker_image_max_pixels': 262144,
    'index_batch_size': 8, 'index_shards': 3,
    'selection': 'independent joint visual cosine and text reranker recall; unique union; trained multimodal logits; provenance MILP',
    'construction': 'global batch identity, disjoint GPU shards, atomic per-batch checkpoints, exact ordered merge',
    'boundaries': ['development screening, not sealed-test generalization',
        'producer construction cost reported separately from serving cost',
        'prebuilt index lookup is not Reader KV reuse',
        'quality and latency comparison uses the same new 4-image/512-output budget',
        'no PIM functional replacement or projected E2E mixed with measured GPU time',
        'baseline JSONs are copied from the same-budget v1 screening and remain source-labeled']}
(root / 'protocol.json').write_text(json.dumps(protocol, indent=2))
PY
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
RUNTIME="$RUN_ROOT/runtime/scripts"
build_user() {
    local user="$1"; shift
    local pids=() gpu shard output
    for shard in 0 1 2; do
        gpu="$1"; shift
        output="$RUN_ROOT/index_user_${user}_shard_${shard}"
        CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$RUNTIME/build_omni_joint_index_sharded.py" build \
            --prepared "$PREPARED" --image-archive "$IMAGE_ARCHIVE" \
            --model "$MODEL_ROOT/Qwen3-VL-Embedding-2B" --output "$output" \
            --user-id "$user" --device cuda:0 --batch-size 8 --image-max-pixels 65536 \
            --max-input-tokens 8192 --image-only --shard-index "$shard" --shard-count 3 \
            >"$output.log" 2>&1 &
        pids+=("$!")
    done
    local status=0 pid
    for pid in "${pids[@]}"; do wait "$pid" || status=1; done
    if [ "$status" != 0 ]; then
        echo "index construction failed for user $user" >&2
        exit "$status"
    fi
    local shards=()
    for shard in 0 1 2; do shards+=("$RUN_ROOT/index_user_${user}_shard_${shard}"); done
    "$PYTHON" "$RUNTIME/build_omni_joint_index_sharded.py" merge \
        --prepared "$PREPARED" --image-archive "$IMAGE_ARCHIVE" \
        --output "$RUN_ROOT/index_user_${user}" --user-id "$user" \
        --batch-size 8 --image-max-pixels 65536 --max-input-tokens 8192 \
        --image-only --shard-count 3 --shards "${shards[@]}" \
        >"$RUN_ROOT/index_user_${user}.merge.log" 2>&1
}
build_user 0 0 1 2 &
user0_pid=$!
build_user 1 3 4 5 &
user1_pid=$!
index_status=0
wait "$user0_pid" || index_status=1
wait "$user1_pid" || index_status=1
if [ "$index_status" != 0 ]; then exit "$index_status"; fi
build_user 2 0 1 2
generation_pids=()
for user in 0 1 2; do
    index="$RUN_ROOT/index_user_${user}"
    common=(--prepared "$PREPARED" --image-archive "$IMAGE_ARCHIVE" --user-id "$user"
        --evidence-policy provenance_milp --disable-edge-reuse --relevance-modality joint_index
        --joint-index "$index" --joint-index-image-max-pixels 65536
        --joint-embedding-model "$MODEL_ROOT/Qwen3-VL-Embedding-2B"
        --joint-reranker-model "$MODEL_ROOT/Qwen3-VL-Reranker-2B"
        --relevance-batch-size 2 --joint-recall-count 16 --joint-text-recall-count 16
        --candidate-sessions 4 --total-text-tokens 3600 --max-images 4 --input-budget 8192
        --screening-run --hardware-identity "RTX3090:$user"
        --concurrency-contract 'isolated GPU; shared storage; sharded producer construction complete; screening only')
    CUDA_VISIBLE_DEVICES="$user" "$PYTHON" "$RUNTIME/run_omni_raw_image_smoke.py" \
        "${common[@]}" --output "$RUN_ROOT/smoke/provenance_milp_user_${user}.json" \
        --limit 1 --max-new-tokens 8 >"$RUN_ROOT/smoke/provenance_milp_user_${user}.log" 2>&1
    CUDA_VISIBLE_DEVICES="$user" "$PYTHON" "$RUNTIME/run_omni_raw_image_smoke.py" \
        "${common[@]}" --output "$RUN_ROOT/provenance_milp_user_${user}.json" \
        --limit 16 --cohort "$COHORT" --max-new-tokens 512 \
        >"$RUN_ROOT/provenance_milp_user_${user}.log" 2>&1 &
    generation_pids+=("$!")
done
generation_status=0
for pid in "${generation_pids[@]}"; do wait "$pid" || generation_status=1; done
if [ "$generation_status" != 0 ]; then
    echo 'Generation incomplete; no survivor-only evaluation' >&2
    exit "$generation_status"
fi
BASELINE_ROOT="$ROOT/artifacts/joint_visual_closure_dev3x16_v1_20261021"
for user in 0 1 2; do
    cp "$BASELINE_ROOT/local_hyperedge_user_${user}.json" "$RUN_ROOT/local_hyperedge_user_${user}.json"
    cp "$BASELINE_ROOT/local_hyperedge_user_${user}.registration.json" "$RUN_ROOT/local_hyperedge_user_${user}.registration.json"
done
"$PYTHON" - "$RUN_ROOT" "$BASELINE_ROOT" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
root, original = map(Path, sys.argv[1:])
manifest = {}
for user in range(3):
    path = root / f'local_hyperedge_user_{user}.json'
    run = json.loads(path.read_text())
    if not run.get('complete') or len(run['rows']) != 16 or run['max_images'] != 4 or run['max_new_tokens'] != 512:
        raise ValueError('incomplete or incompatible baseline')
    manifest[str(path)] = {'original_path': str(original / path.name),
                          'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
(root / 'baseline_import.json').write_text(json.dumps(manifest, indent=2))
PY
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
    --runs "${runs[@]}" --judge "$RUN_ROOT/qwen3vl8b_judge.json" \
    --output "$RUN_ROOT/evidence_diagnosis.json"
