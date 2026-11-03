#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: android_bench.sh --model PATH --binary PATH --threads N --manifest PATH [options]

Push a model and a llama.cpp llama-bench binary to an Android device over adb, then run
each unique (prefill_tokens, decode_tokens, n_images) combination from the replay manifest.

Required:
  --model PATH          Local model file supported by the chosen llama.cpp build
  --binary PATH         Local Android-compatible llama-bench executable
  --threads N           Positive benchmark thread count
  --manifest PATH       Output of device/replay_from_trace.py

Options:
  --output-dir PATH     New or empty host directory for raw results
  --serial SERIAL       adb device serial; omitted when exactly one device is attached
  --remote-dir PATH     Remote scratch directory
  --repetitions N       llama-bench repetitions (default: 5)
  --include-image-text-only
                         Also run image-bearing combinations using only their text token
                         lengths; this does NOT replay image preprocessing or the visual
                         encoder
  --help                 Show this help

The script preserves llama-bench stdout/stderr and records an invocation table. It does not
parse benchmark timings, aggregate results, or claim that the model is supported by the binary.
The binary must be self-contained for Android, or its required shared libraries must already be
available to the device process.
EOF
}

MODEL=
BINARY=
THREADS=
MANIFEST=
OUTPUT_DIR=
SERIAL=
REMOTE_DIR=
REPETITIONS=5
INCLUDE_IMAGE_TEXT_ONLY=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            [[ $# -ge 2 ]] || { echo 'missing --model value' >&2; exit 2; }
            MODEL=$2
            shift 2
            ;;
        --binary)
            [[ $# -ge 2 ]] || { echo 'missing --binary value' >&2; exit 2; }
            BINARY=$2
            shift 2
            ;;
        --threads)
            [[ $# -ge 2 ]] || { echo 'missing --threads value' >&2; exit 2; }
            THREADS=$2
            shift 2
            ;;
        --manifest)
            [[ $# -ge 2 ]] || { echo 'missing --manifest value' >&2; exit 2; }
            MANIFEST=$2
            shift 2
            ;;
        --output-dir)
            [[ $# -ge 2 ]] || { echo 'missing --output-dir value' >&2; exit 2; }
            OUTPUT_DIR=$2
            shift 2
            ;;
        --serial)
            [[ $# -ge 2 ]] || { echo 'missing --serial value' >&2; exit 2; }
            SERIAL=$2
            shift 2
            ;;
        --remote-dir)
            [[ $# -ge 2 ]] || { echo 'missing --remote-dir value' >&2; exit 2; }
            REMOTE_DIR=$2
            shift 2
            ;;
        --repetitions)
            [[ $# -ge 2 ]] || { echo 'missing --repetitions value' >&2; exit 2; }
            REPETITIONS=$2
            shift 2
            ;;
        --include-image-text-only)
            INCLUDE_IMAGE_TEXT_ONLY=1
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

[[ -n "$MODEL" ]] || { echo '--model is required' >&2; exit 2; }
[[ -n "$BINARY" ]] || { echo '--binary is required' >&2; exit 2; }
[[ -n "$THREADS" ]] || { echo '--threads is required' >&2; exit 2; }
[[ -n "$MANIFEST" ]] || { echo '--manifest is required' >&2; exit 2; }
[[ -f "$MODEL" ]] || { echo "model file does not exist: $MODEL" >&2; exit 2; }
[[ -f "$BINARY" ]] || { echo "binary file does not exist: $BINARY" >&2; exit 2; }
[[ -f "$MANIFEST" ]] || { echo "manifest file does not exist: $MANIFEST" >&2; exit 2; }
[[ "$THREADS" =~ ^[1-9][0-9]*$ ]] || { echo '--threads must be a positive integer' >&2; exit 2; }
[[ "$REPETITIONS" =~ ^[1-9][0-9]*$ ]] || { echo '--repetitions must be a positive integer' >&2; exit 2; }
command -v adb >/dev/null || { echo 'adb is required' >&2; exit 2; }
command -v jq >/dev/null || { echo 'jq is required' >&2; exit 2; }
command -v sha256sum >/dev/null || { echo 'sha256sum is required' >&2; exit 2; }
jq -e '.schema == "device-replay-manifest-v1"' "$MANIFEST" >/dev/null \
    || { echo "unsupported replay manifest: $MANIFEST" >&2; exit 2; }
jq -e '(.combinations | type) == "array" and (.combinations | length) > 0' "$MANIFEST" >/dev/null \
    || { echo "replay manifest has no combinations: $MANIFEST" >&2; exit 2; }

if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="device_replay_results/replay_$(date -u +%Y%m%dT%H%M%SZ)"
fi
if [[ -e "$OUTPUT_DIR" ]]; then
    [[ -d "$OUTPUT_DIR" ]] || { echo "output path is not a directory: $OUTPUT_DIR" >&2; exit 2; }
    if [[ -n "$(find "$OUTPUT_DIR" -mindepth 1 -print -quit)" ]]; then
        echo "refusing to write into non-empty output directory: $OUTPUT_DIR" >&2
        exit 2
    fi
fi
mkdir -p "$OUTPUT_DIR"

if [[ -z "$REMOTE_DIR" ]]; then
    REMOTE_DIR="/data/local/tmp/omni-replay-$$"
fi
[[ "$REMOTE_DIR" =~ ^/[A-Za-z0-9._/-]+$ ]] \
    || { echo "--remote-dir must be an absolute path without spaces" >&2; exit 2; }

ADB=(adb)
if [[ -n "$SERIAL" ]]; then
    ADB+=(-s "$SERIAL")
fi

REMOTE_MODEL="$REMOTE_DIR/model.gguf"
REMOTE_BINARY="$REMOTE_DIR/llama-bench"
REMOTE_MANIFEST="$REMOTE_DIR/manifest.json"

printf 'action_or_output\n' >"$OUTPUT_DIR/adb_commands.tsv"
record_adb() {
    local action=$1
    shift
    printf '%s\n' "$action" >>"$OUTPUT_DIR/adb_commands.tsv"
    local status=0
    "$@" 2>&1 | tee -a "$OUTPUT_DIR/adb_commands.tsv" || status=$?
    if [[ $status -ne 0 ]]; then
        echo "adb action failed: $action" >&2
        return "$status"
    fi
}

record_adb mkdir "${ADB[@]}" shell mkdir -p "$REMOTE_DIR"
record_adb push_model "${ADB[@]}" push "$MODEL" "$REMOTE_MODEL"
record_adb push_binary "${ADB[@]}" push "$BINARY" "$REMOTE_BINARY"
record_adb push_manifest "${ADB[@]}" push "$MANIFEST" "$REMOTE_MANIFEST"
record_adb chmod "${ADB[@]}" shell chmod 755 "$REMOTE_BINARY"

printf 'index\tprefill_tokens\tdecode_tokens\tn_images\toccurrences\tstatus\tjson\tstderr\n' \
    >"$OUTPUT_DIR/commands.tsv"
run_count=0
ok_count=0
failed_count=0
skipped_count=0
image_text_only_count=0

while IFS=$'\t' read -r prefill_tokens decode_tokens n_images occurrences; do
    run_count=$((run_count + 1))
    printf -v label '%03d' "$run_count"
    stem="bench_${label}_p${prefill_tokens}_n${decode_tokens}_i${n_images}"
    json_path="$OUTPUT_DIR/${stem}.json"
    stderr_path="$OUTPUT_DIR/${stem}.stderr"
    if (( n_images > 0 )) && [[ "$INCLUDE_IMAGE_TEXT_ONLY" != 1 ]]; then
        skipped_count=$((skipped_count + 1))
        printf '%s\t%s\t%s\t%s\t%s\t%s\t-\t-\n' \
            "$run_count" "$prefill_tokens" "$decode_tokens" "$n_images" "$occurrences" \
            'skipped_image_encoder' >>"$OUTPUT_DIR/commands.tsv"
        continue
    fi
    status=ok
    if (( n_images > 0 )); then
        status=ok_image_text_only
    fi
    if "${ADB[@]}" shell "$REMOTE_BINARY" -m "$REMOTE_MODEL" \
        -p "$prefill_tokens" -n "$decode_tokens" -t "$THREADS" \
        -r "$REPETITIONS" -o json >"$json_path" 2>"$stderr_path"; then
        ok_count=$((ok_count + 1))
        if (( n_images > 0 )); then
            image_text_only_count=$((image_text_only_count + 1))
        fi
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$run_count" "$prefill_tokens" "$decode_tokens" "$n_images" "$occurrences" \
            "$status" "$(basename "$json_path")" "$(basename "$stderr_path")" \
            >>"$OUTPUT_DIR/commands.tsv"
    else
        failed_count=$((failed_count + 1))
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$run_count" "$prefill_tokens" "$decode_tokens" "$n_images" "$occurrences" \
            'failed' "$(basename "$json_path")" "$(basename "$stderr_path")" \
            >>"$OUTPUT_DIR/commands.tsv"
    fi
done < <(jq -r '.combinations[] | [.prefill_tokens, .decode_tokens, .n_images, .occurrences] | @tsv' "$MANIFEST")

model_sha256=$(sha256sum "$MODEL" | awk '{print $1}')
binary_sha256=$(sha256sum "$BINARY" | awk '{print $1}')
manifest_sha256=$(sha256sum "$MANIFEST" | awk '{print $1}')
script_sha256=$(sha256sum "$0" | awk '{print $1}')
jq -n \
    --arg model_sha256 "$model_sha256" \
    --arg binary_sha256 "$binary_sha256" \
    --arg manifest_sha256 "$manifest_sha256" \
    --arg script_sha256 "$script_sha256" \
    --arg threads "$THREADS" \
    --arg repetitions "$REPETITIONS" \
    --arg serial "$SERIAL" \
    --arg remote_dir "$REMOTE_DIR" \
    --argjson run_count "$run_count" \
    --argjson ok_count "$ok_count" \
    --argjson failed_count "$failed_count" \
    --argjson skipped_count "$skipped_count" \
    --argjson image_text_only_count "$image_text_only_count" \
    --argjson include_image_text_only "$([ "$INCLUDE_IMAGE_TEXT_ONLY" = 1 ] && echo true || echo false)" \
    '{
        schema: "device-replay-run-v1",
        model_sha256: $model_sha256,
        binary_sha256: $binary_sha256,
        manifest_sha256: $manifest_sha256,
        script_sha256: $script_sha256,
        threads: ($threads | tonumber),
        repetitions: ($repetitions | tonumber),
        adb_serial: (if $serial == "" then null else $serial end),
        remote_dir: $remote_dir,
        include_image_text_only: $include_image_text_only,
        combinations_total: $run_count,
        combinations_ok: $ok_count,
        combinations_failed: $failed_count,
        combinations_skipped_image_encoder: $skipped_count,
        combinations_ok_image_text_only: $image_text_only_count,
        measurements: [],
        boundary: "raw llama-bench files are preserved; image encoder work is not replayed and no aggregate performance claim is made"
    }' >"$OUTPUT_DIR/run_metadata.json"

echo "raw results: $OUTPUT_DIR"
echo "combinations: total=$run_count ok=$ok_count failed=$failed_count skipped_image=$skipped_count"
if (( skipped_count > 0 )); then
    echo 'image-bearing combinations were skipped; see commands.tsv' >&2
fi
if (( failed_count > 0 )); then
    echo 'one or more llama-bench invocations failed; inspect commands.tsv and raw stderr' >&2
    exit 1
fi
