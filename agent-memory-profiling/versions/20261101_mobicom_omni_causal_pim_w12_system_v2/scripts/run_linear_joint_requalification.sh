#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PRETEST=$(cd "$ROOT/../../.." && pwd)
PYTHON="$ROOT/../../.venvs/flamr-qwen35/bin/python"
MODEL_ROOT="$ROOT/../../models"
PREPARED="$ROOT/artifacts/omni_paper_scope_v3_20261021/prepared"
COHORT="$ROOT/artifacts/hash_cohort_dev3x16_v1_20261021.json"
INDEX_ROOT="$ROOT/artifacts/joint_visual_closure_dev3x16_v3_sharded_20261021"
RUN_ROOT="${RUN_ROOT:-$ROOT/artifacts/joint_linear_requalification_dev3x16_v1_20261022}"
if [ -e "$RUN_ROOT" ]; then
    echo 'Fresh RUN_ROOT required' >&2
    exit 2
fi
mkdir -p "$RUN_ROOT/runtime"
cp -a "$ROOT/scripts" "$ROOT/pim_backend" "$RUN_ROOT/runtime/"
"$PYTHON" - "$RUN_ROOT" "$COHORT" "$INDEX_ROOT" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

root, cohort, index_root = map(Path, sys.argv[1:])
identities = json.loads(cohort.read_text())['question_ids_by_user']
if set(identities) != {'0', '1', '2'} or any(len(items) != 16 for items in identities.values()):
    raise ValueError('original frozen development cohort required')
(root / 'cohort.json').write_bytes(cohort.read_bytes())
sources = {str(path.relative_to(root / 'runtime')): hashlib.sha256(path.read_bytes()).hexdigest()
           for path in sorted((root / 'runtime').rglob('*'))
           if path.is_file() and path.suffix in ('.py', '.json', '.sh')}
indexes = {}
for user in ('0', '1', '2'):
    directory = index_root / f'index_user_{user}'
    if not directory.is_dir():
        raise ValueError('existing registered index required')
    checksums = {}
    for path in sorted(directory.rglob('*')):
        if path.is_file():
            with path.open('rb') as handle:
                checksums[str(path.relative_to(directory))] = hashlib.file_digest(handle, 'sha256').hexdigest()
    if not checksums:
        raise ValueError('empty index')
    indexes[user] = {'path': str(directory), 'files_sha256': checksums}
protocol = {'status': 'registered_before_generation', 'sources_sha256': sources,
    'cohort_sha256': hashlib.sha256(cohort.read_bytes()).hexdigest(),
    'reader': 'Qwen3.5-4B', 'judge': 'Qwen3-VL-8B-Instruct',
    'max_images': 4, 'max_new_tokens': 512, 'input_budget': 8192, 'text_budget': 3600,
    'reader_patch_projection': 'original_conv_unchanged',
    'joint_patch_projection': 'linear', 'joint_patch_projection_role': 'candidate_reranker_only',
    'embedding_index_patch_projection': 'original_conv_unchanged',
    'reranker_batch_size': 2, 'visual_recall_count': 16, 'text_recall_count': 16,
    'index_image_max_pixels': 65536, 'reader_image_max_pixels': 262144,
    'reranker_image_max_pixels': 262144, 'indexes': indexes,
    'producer_construction_source_protocol': str(index_root / 'protocol.json'),
    'producer_construction_source_protocol_sha256': hashlib.sha256((index_root / 'protocol.json').read_bytes()).hexdigest(),
    'gpu_by_policy': {'provenance_milp': {'0': 0, '1': 1, '2': 2},
                      'local_hyperedge': {'0': 3, '1': 4, '2': 5}},
    'evaluation_label_files_read_during_serving': [],
    'profile_reranker': False, 'state_reuse_enabled': False, 'pim_enabled': False,
    'boundaries': ['new BF16 numerical contract requires new quality scores; old scores not inherited',
        'patch implementation repair is not hypergraph, state-reuse, or PIM contribution',
        'both baseline and candidate freshly generated; same unchanged Reader and budgets',
        'isolated GPUs, shared CPU/storage; screening latency only, not causal E2E qualification',
        'existing exact embedding indexes reused; construction remains separately accountable',
        'development-only proxy answer quality; not task execution success or sealed-test generalization'],
    'main_experiment_closed': False, 'actual_pim_end_to_end_speedup': None}
(root / 'protocol.json').write_text(json.dumps(protocol, indent=2))
PY
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu --format=csv > "$RUN_ROOT/hardware.csv"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
RUNTIME="$RUN_ROOT/runtime/scripts"
workers=()
printf 'generation_running\n' > "$RUN_ROOT/status.txt"
for policy in local_hyperedge provenance_milp; do
    for user in 0 1 2; do
        gpu="$user"
        extra=()
        if [ "$policy" = local_hyperedge ]; then
            gpu=$((user + 3))
        else
            extra=(--relevance-modality joint_index --joint-index "$INDEX_ROOT/index_user_${user}"
                --joint-index-image-max-pixels 65536
                --joint-embedding-model "$MODEL_ROOT/Qwen3-VL-Embedding-2B"
                --joint-reranker-model "$MODEL_ROOT/Qwen3-VL-Reranker-2B"
                --relevance-batch-size 2 --joint-recall-count 16 --joint-text-recall-count 16
                --joint-patch-projection linear)
        fi
        CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$RUNTIME/run_omni_raw_image_smoke.py" \
            --prepared "$PREPARED" --image-archive "$PRETEST/MobileMem-main/data/omni/image.zip" \
            --output "$RUN_ROOT/${policy}_user_${user}.json" --user-id "$user" --limit 16 \
            --cohort "$RUN_ROOT/cohort.json" --evidence-policy "$policy" --disable-edge-reuse \
            --candidate-sessions 4 --total-text-tokens 3600 --max-images 4 --max-new-tokens 512 \
            --input-budget 8192 --screening-run --hardware-identity "RTX3090:$gpu" \
            --concurrency-contract 'isolated GPU; shared CPU/storage; new numerical-contract development screen only' \
            "${extra[@]}" > "$RUN_ROOT/${policy}_user_${user}.log" 2>&1 &
        workers+=("$!")
    done
done
status=0
for worker in "${workers[@]}"; do wait "$worker" || status=1; done
if [ "$status" != 0 ]; then
    printf 'generation_failed_no_survivor_evaluation\n' > "$RUN_ROOT/status.txt"
    exit "$status"
fi
runs=()
for user in 0 1 2; do
    runs+=("$RUN_ROOT/local_hyperedge_user_${user}.json" "$RUN_ROOT/provenance_milp_user_${user}.json")
done
printf 'judge_running\n' > "$RUN_ROOT/status.txt"
CUDA_VISIBLE_DEVICES=6 "$PYTHON" "$RUNTIME/judge_omni_closure.py" \
    --prepared "$PREPARED" --runs "${runs[@]}" \
    --prompt-source "$PRETEST/MobileMem-main/omni/eval/eval/question_answering_and_judge_prompts.txt" \
    --output "$RUN_ROOT/qwen3vl8b_judge.json" --max-new-tokens 512 > "$RUN_ROOT/judge.log" 2>&1
"$PYTHON" "$RUNTIME/summarize_closure_pilot.py" --run-root "$RUN_ROOT" \
    --judge "$RUN_ROOT/qwen3vl8b_judge.json" --output "$RUN_ROOT/summary.json" > "$RUN_ROOT/summary.log"
"$PYTHON" "$RUNTIME/audit_linear_joint_requalification.py" --run-root "$RUN_ROOT" \
    --judge "$RUN_ROOT/qwen3vl8b_judge.json" --output "$RUN_ROOT/requalification_audit.json"
printf 'screen_complete_not_main_experiment_closed\n' > "$RUN_ROOT/status.txt"
