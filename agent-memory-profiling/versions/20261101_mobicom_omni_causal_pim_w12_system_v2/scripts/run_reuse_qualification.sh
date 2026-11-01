#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON="$ROOT/../../.venvs/flamr-qwen35/bin/python"
PREPARED="$ROOT/artifacts/omni_paper_scope_v3_20261021/prepared"
COHORT="${COHORT:-$ROOT/artifacts/omni_dev3x32_reuse_v1_20261022/cohort.json}"
READER_PATCH_PROJECTION="${READER_PATCH_PROJECTION:-conv}"
SCREEN_POLICIES="${SCREEN_POLICIES:-baseline reuse}"
ARCHIVE="$ROOT/../../../MobileMem-main/data/omni/image.zip"
RUN_ROOT="${RUN_ROOT:-$ROOT/artifacts/omni_dev3x32_reuse_v2_20261022}"
EVIDENCE_POLICY="${EVIDENCE_POLICY:-local_hyperedge}"
RELEVANCE_MODALITY="${RELEVANCE_MODALITY:-text}"
READER_QUANTIZATION="${READER_QUANTIZATION:-bf16}"
W8A16_GROUP_SIZE="${W8A16_GROUP_SIZE:-128}"
W8A16_RESIDUAL_BITS="${W8A16_RESIDUAL_BITS:-0}"
W8A16_FALLBACK_SCOPE="${W8A16_FALLBACK_SCOPE:-none}"
IMAGE_EDGES_PER_SESSION="${IMAGE_EDGES_PER_SESSION:-0}"
QUERY_LOCAL_TOKEN_COST_REUSE="${QUERY_LOCAL_TOKEN_COST_REUSE:-0}"
DEFER_INT8_FINITE_CHECKS="${DEFER_INT8_FINITE_CHECKS:-0}"
PIMNAST_FUNCTIONAL_PROJECTION="${PIMNAST_FUNCTIONAL_PROJECTION:-}"
GPU_OFFSET="${GPU_OFFSET:-0}"
[[ "$GPU_OFFSET" =~ ^[0-9]+$ ]] || exit 2
[[ "$IMAGE_EDGES_PER_SESSION" =~ ^[0-9]+$ ]] || exit 2
[[ "$QUERY_LOCAL_TOKEN_COST_REUSE" == 0 || "$QUERY_LOCAL_TOKEN_COST_REUSE" == 1 ]] || exit 2
[[ "$DEFER_INT8_FINITE_CHECKS" == 0 || "$DEFER_INT8_FINITE_CHECKS" == 1 ]] || exit 2
[[ "$W8A16_GROUP_SIZE" =~ ^(8|16|32|64|128|256)$ ]] || exit 2
[[ "$W8A16_RESIDUAL_BITS" =~ ^(0|2|4)$ ]] || exit 2
[[ "$W8A16_FALLBACK_SCOPE" =~ ^(none|lm_head|output_sensitive)$ ]] || exit 2
if [[ "$READER_QUANTIZATION" != w8a16 && "$W8A16_FALLBACK_SCOPE" != none ]]; then
    echo 'W8A16 fallback requires the W8A16 arm' >&2
    exit 2
fi
if [[ "$READER_QUANTIZATION" != w8a16 && "$W8A16_RESIDUAL_BITS" != 0 ]]; then
    echo 'W8A16 residual correction requires the W8A16 arm' >&2
    exit 2
fi
if [[ "$DEFER_INT8_FINITE_CHECKS" == 1 && "$READER_QUANTIZATION" != w8a8 ]]; then
    echo 'deferred finite-check aggregation requires shared W8A8 primary' >&2
    exit 2
fi
if [[ -n "$PIMNAST_FUNCTIONAL_PROJECTION" ]]; then
    [[ "$READER_QUANTIZATION" == w8a8 && -f "$PIMNAST_FUNCTIONAL_PROJECTION" ]] || exit 2
fi

declare -A AVAILABLE_GPUS=()
declare -A ASSIGNED_GPUS=()
while IFS= read -r index; do
    index="${index//[[:space:]]/}"
    [[ -z "$index" ]] || AVAILABLE_GPUS["$index"]=1
done < <(nvidia-smi --query-gpu=index --format=csv,noheader)
GPU_ASSIGNMENTS=()
for policy in $SCREEN_POLICIES; do
    [[ "$policy" == baseline || "$policy" == reuse || "$policy" == reference || "$policy" == expanded || "$policy" == milp ]] || exit 2
    for user in 0 1 2; do
        gpu=$((user + GPU_OFFSET))
        [[ "$policy" != reuse && "$policy" != expanded && "$policy" != milp ]] || gpu=$((gpu + 3))
        if [[ -z "${AVAILABLE_GPUS[$gpu]:-}" || -n "${ASSIGNED_GPUS[$gpu]:-}" ]]; then
            echo "invalid or duplicate GPU assignment: policy=$policy user=$user gpu=$gpu; GPU_OFFSET is added to policy offsets" >&2
            exit 2
        fi
        ASSIGNED_GPUS["$gpu"]=1
        GPU_ASSIGNMENTS+=("$policy:$user:$gpu")
    done
done
[[ ${#GPU_ASSIGNMENTS[@]} -gt 0 ]] || exit 2

if [[ -e "$RUN_ROOT" ]]; then
    echo "fresh RUN_ROOT required: $RUN_ROOT" >&2
    exit 2
fi
test -f "$ARCHIVE"
test -f "$COHORT"
mkdir -p "$RUN_ROOT/runtime"
cp -a "$ROOT/scripts" "$ROOT/pim_backend" "$RUN_ROOT/runtime/"
cp "$COHORT" "$RUN_ROOT/cohort.json"
if [[ -n "$PIMNAST_FUNCTIONAL_PROJECTION" ]]; then
    cp "$PIMNAST_FUNCTIONAL_PROJECTION" "$RUN_ROOT/pimnast_projection.json"
fi
printf '%s\n' "${GPU_ASSIGNMENTS[@]}" > "$RUN_ROOT/gpu_assignments.txt"
"$PYTHON" - "$RUN_ROOT" "$ARCHIVE" "$COHORT" "$READER_PATCH_PROJECTION" "$SCREEN_POLICIES" "$EVIDENCE_POLICY" "$RELEVANCE_MODALITY" "$READER_QUANTIZATION" "$IMAGE_EDGES_PER_SESSION" "$QUERY_LOCAL_TOKEN_COST_REUSE" "$DEFER_INT8_FINITE_CHECKS" "$W8A16_GROUP_SIZE" "$W8A16_RESIDUAL_BITS" "$W8A16_FALLBACK_SCOPE" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

root, archive, cohort_path = map(Path, sys.argv[1:4])
cohort = json.loads(cohort_path.read_text())
patch_projection, policies, evidence_policy, relevance_modality, quantization, image_quota, token_cost_reuse, defer_finite_checks, w8a16_group_size, w8a16_residual_bits, w8a16_fallback_scope = sys.argv[4:]
policy_list = policies.split()
if policy_list == ['reference', 'expanded']:
    treatment = 'dense session recall top4 versus top8; both enable identical exact producer reuse'
elif policy_list == ['baseline', 'reuse']:
    treatment = 'canonical exact cross-request embedding producer reuse'
elif len(policy_list) == 1 and policy_list[0] in {'reference', 'expanded'}:
    treatment = ('single dense session recall top4 reference arm' if policy_list[0] == 'reference'
                 else 'single dense session recall top8 candidate arm')
elif len(policy_list) == 1 and policy_list[0] == 'reuse':
    treatment = 'single exact producer reuse candidate arm'
elif len(policy_list) == 1 and policy_list[0] == 'baseline':
    treatment = 'single no-reuse reference arm'
elif len(policy_list) == 1 and policy_list[0] == 'milp':
    treatment = 'single provenance MILP evidence-policy package; not single-factor solver attribution'
else:
    treatment = 'registered multi-policy evidence-policy package; interpret only with its declared arm differences'
protocol = {
    'status': 'registered_before_generation',
    'requests_per_policy': sum(map(len, cohort['question_ids_by_user'].values())),
    'users': ['0', '1', '2'], 'cohort_sha256': hashlib.sha256((root / 'cohort.json').read_bytes()).hexdigest(),
    'archive': str(archive.resolve()),
    'only_treatment': treatment,
    'reader_patch_projection': patch_projection, 'policies': policies.split(),
    'reader_quantization': quantization,
    'w8a16_group_size': int(w8a16_group_size) if quantization == 'w8a16' else None,
    'w8a16_residual_bits': (int(w8a16_residual_bits)
                            if quantization == 'w8a16' else None),
    'w8a16_fallback_scope': w8a16_fallback_scope if quantization == 'w8a16' else None,
    'image_candidate_quota_per_session': int(image_quota),
    'query_local_token_cost_reuse': token_cost_reuse == '1',
    'deferred_int8_finite_checks': defer_finite_checks == '1',
    'deferred_int8_finite_checks_contract': (
        'all existing predicates evaluated; one Host confirmation before each cached Reader forward returns; '
        'prefill remains synchronous; no skipped validation or repeated producer arithmetic'
        if defer_finite_checks == '1' else None),
    'evidence_policy': evidence_policy, 'relevance_modality': relevance_modality,
    'per_policy_evidence': {policy: ('provenance_milp' if policy == 'milp' else evidence_policy)
                            for policy in policies.split()},
    'reader_linear_boundary': 'implementation qualification, changes BF16 numerical path; not PIM/reuse/algorithm benefit',
    'sources_sha256': {str(path.relative_to(root / 'runtime')): hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in sorted((root / 'runtime').rglob('*'))
                       if path.is_file() and path.suffix in ('.py', '.json', '.sh')},
    'max_new_tokens': 512, 'max_images': 4, 'image_max_pixels': 262144,
    'cold_state': 'empty store at each user start; requests remain in source order',
    'measurement': 'concurrent shared-host screening; not isolated causal E2E latency',
    'validation_accessed': False, 'holdout_accessed': False,
    'state': 'qualification experiment; no innovation or significant speedup claim',
}
projection = root / 'pimnast_projection.json'
if projection.exists():
    protocol.update(pimnast_functional_projection_sha256=hashlib.sha256(projection.read_bytes()).hexdigest(),
        only_treatment='functional PIMnast bank-layout backend at frozen placement; query-local token costs also enabled if declared; not a causal timing or physical PIM experiment',
        physical_pim_execution=False)
(root / 'protocol.json').write_text(json.dumps(protocol, indent=2))
PY
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu --format=csv > "$RUN_ROOT/hardware.csv"

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false

run_one() {
    local policy=$1
    local user=$2
    local gpu=$3
    local output="$RUN_ROOT/${policy}_user_${user}.json"
    local reuse_args=()
    local sessions=4
    local evidence_policy="$EVIDENCE_POLICY"
    local limit
    limit=$("$PYTHON" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["question_ids_by_user"][sys.argv[2]]))' "$COHORT" "$user")
    if [[ "$policy" == "baseline" ]]; then
        reuse_args+=(--disable-edge-reuse)
    fi
    if [[ "$policy" == "expanded" ]]; then
        sessions=8
    fi
    if [[ "$policy" == "milp" ]]; then
        evidence_policy=provenance_milp
    fi
    if [[ "$QUERY_LOCAL_TOKEN_COST_REUSE" == 1 ]]; then
        reuse_args+=(--query-local-token-cost-reuse)
    fi
    if [[ "$DEFER_INT8_FINITE_CHECKS" == 1 ]]; then
        reuse_args+=(--defer-int8-finite-checks)
    fi
    if [[ -n "$PIMNAST_FUNCTIONAL_PROJECTION" ]]; then
        reuse_args+=(--pimnast-functional-projection "$RUN_ROOT/pimnast_projection.json")
    fi
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$RUN_ROOT/runtime/scripts/run_omni_raw_image_smoke.py" \
        --prepared "$PREPARED" \
        --image-archive "$ARCHIVE" \
        --output "$output" \
        --user-id "$user" \
        --limit "$limit" \
        --cohort "$COHORT" \
        --screening-run \
        --reader-patch-projection "$READER_PATCH_PROJECTION" \
        --reader-quantization "$READER_QUANTIZATION" \
        --w8a16-group-size "$W8A16_GROUP_SIZE" \
        --w8a16-residual-bits "$W8A16_RESIDUAL_BITS" \
        --w8a16-fallback-scope "$W8A16_FALLBACK_SCOPE" \
        --device cuda:0 \
        --candidate-sessions "$sessions" \
        --max-images 4 \
        --max-new-tokens 512 \
        --session-text-tokens 900 \
        --evidence-policy "$evidence_policy" \
        --relevance-modality "$RELEVANCE_MODALITY" \
        --max-hyperedges 64 \
        --image-edges-per-session "$IMAGE_EDGES_PER_SESSION" \
        --image-max-pixels 262144 \
        --edge-cache-bytes $((64 * 1024 * 1024)) \
        --hardware-identity "RTX3090:gpu${gpu}" \
        --concurrency-contract "one worker per isolated GPU; shared CPU/storage" \
        "${reuse_args[@]}" >"$RUN_ROOT/${policy}_user_${user}.log" 2>&1
}

jobs=()
for assignment in "${GPU_ASSIGNMENTS[@]}"; do
    IFS=: read -r policy user gpu <<< "$assignment"
    run_one "$policy" "$user" "$gpu" & jobs+=("$!")
done

status=0
for pid in "${jobs[@]}"; do
    wait "$pid" || status=$?
done
exit "$status"
