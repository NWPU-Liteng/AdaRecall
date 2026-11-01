#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PRETEST=$(cd "$ROOT/../../.." && pwd)
PYTHON="$ROOT/../../.venvs/flamr-qwen35/bin/python"
RUN_ROOT="${RUN_ROOT:-$ROOT/artifacts/joint_visual_cost_profile_dev3x1_v1_20261022}"
INDEX_ROOT="$ROOT/artifacts/joint_visual_closure_dev3x16_v3_sharded_20261021"
COHORT="$ROOT/artifacts/hash_cohort_dev3x16_v1_20261021.json"
RERANKER_BATCH_SIZE="${RERANKER_BATCH_SIZE:-2}"
JOINT_PATCH_PROJECTION="${JOINT_PATCH_PROJECTION:-conv}"
read -r -a USERS <<< "${USERS:-0 1 2}"
read -r -a GPU_IDS <<< "${GPU_IDS:-0 1 2}"
if [ "${#USERS[@]}" -ne "${#GPU_IDS[@]}" ] || [ "${#USERS[@]}" -eq 0 ]; then
    echo 'USERS and GPU_IDS must have matching nonempty values' >&2
    exit 2
fi
if [ -e "$RUN_ROOT/runtime" ] || [ -e "$RUN_ROOT/protocol.json" ]; then
    echo 'Fresh diagnostic run required' >&2
    exit 2
fi
mkdir -p "$RUN_ROOT/runtime"
cp -a "$ROOT/scripts" "$ROOT/pim_backend" "$RUN_ROOT/runtime/"
"$PYTHON" - "$RUN_ROOT" "$COHORT" "$RERANKER_BATCH_SIZE" "${USERS[*]}" "${GPU_IDS[*]}" "$JOINT_PATCH_PROJECTION" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
root, source = map(Path, sys.argv[1:3])
batch_size = int(sys.argv[3])
users, gpu_ids, patch_projection = sys.argv[4].split(), sys.argv[5].split(), sys.argv[6]
if patch_projection not in {'conv', 'linear'}:
    raise ValueError('JOINT_PATCH_PROJECTION must be conv or linear')
cohort = json.loads(source.read_text())
cohort.update(selection='first source-ordered member of frozen development cohort; diagnostic only',
              per_user=1, parent_cohort_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
cohort['question_ids_by_user'] = {user: identities[:1]
                                 for user, identities in cohort['question_ids_by_user'].items()}
cohort.pop('script_sha256', None)
cohort_path = root / 'cohort.json'
cohort_path.write_text(json.dumps(cohort, indent=2))
sources = {str(path.relative_to(root / 'runtime')): hashlib.sha256(path.read_bytes()).hexdigest()
           for path in sorted((root / 'runtime').rglob('*'))
           if path.is_file() and path.suffix in ('.py', '.json', '.sh')}
protocol = {'status': 'registered_before_generation', 'diagnostic_only': True,
    'sources_sha256': sources, 'cohort_sha256': hashlib.sha256(cohort_path.read_bytes()).hexdigest(),
    'evaluation_label_files_read': [], 'reader': 'Qwen3.5-4B', 'judge_invoked': False,
    'max_images': 4, 'max_new_tokens': 8, 'input_budget': 8192, 'text_budget': 3600,
    'profile_reranker': True, 'reranker_batch_size': batch_size,
    'joint_patch_projection': patch_projection,
    'gpu_by_user': dict(zip(users, map(int, gpu_ids))),
    'boundary': 'phase diagnostic with synchronization; no quality or causal E2E comparison'}
(root / 'protocol.json').write_text(json.dumps(protocol, indent=2))
PY
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
RUNTIME="$RUN_ROOT/runtime/scripts"
workers=()
for index in "${!USERS[@]}"; do
    user="${USERS[$index]}"
    gpu="${GPU_IDS[$index]}"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$RUNTIME/run_omni_raw_image_smoke.py" \
        --prepared "$ROOT/artifacts/omni_paper_scope_v3_20261021/prepared" \
        --image-archive "$PRETEST/MobileMem-main/data/omni/image.zip" --user-id "$user" \
        --evidence-policy provenance_milp --disable-edge-reuse --relevance-modality joint_index \
        --joint-index "$INDEX_ROOT/index_user_${user}" --joint-index-image-max-pixels 65536 \
        --joint-embedding-model "$ROOT/../../models/Qwen3-VL-Embedding-2B" \
        --joint-reranker-model "$ROOT/../../models/Qwen3-VL-Reranker-2B" \
        --relevance-batch-size "$RERANKER_BATCH_SIZE" --joint-recall-count 16 --joint-text-recall-count 16 \
        --candidate-sessions 4 --total-text-tokens 3600 --max-images 4 --input-budget 8192 \
        --screening-run --hardware-identity "RTX3090:$gpu" \
        --concurrency-contract 'phase diagnostic only; isolated GPU; shared CPU/storage' \
        --cohort "$RUN_ROOT/cohort.json" --limit 1 --max-new-tokens 8 --profile-reranker \
        --joint-patch-projection "$JOINT_PATCH_PROJECTION" \
        --output "$RUN_ROOT/provenance_milp_user_${user}.json" \
        >"$RUN_ROOT/user_${user}.log" 2>&1 &
    workers+=("$!")
done
status=0
for worker in "${workers[@]}"; do wait "$worker" || status=1; done
exit "$status"
