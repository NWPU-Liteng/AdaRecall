#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-0}"
MEMORY_SIZE="${MEMORY_SIZE:-100000}"
REPEAT="${REPEAT:-5}"
CANDIDATE_K="${CANDIDATE_K:-50 100 200}"
TOP_K="${TOP_K:-5 10}"
TIMEOUT_PER_QUERY="${TIMEOUT_PER_QUERY:-120}"
LOCAL_QWEN_DIR="${LOCAL_QWEN_DIR:-../dir}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RETRIEVAL_BACKEND="${RETRIEVAL_BACKEND:-ann_numpy}"

mkdir -p logs results
LOG="logs/run_langgraph_experiment.log"
: > "$LOG"

say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

check_env() {
  "$PYTHON_BIN" - <<'PY'
import torch
import transformers
from packaging.version import Version

tv = Version(transformers.__version__)
if not (Version("4.51.0") <= tv < Version("4.52.0")):
    raise SystemExit(
        f"Incompatible transformers version: {transformers.__version__}. "
        "Use >=4.51.0 and <4.52.0 for this project."
    )
if Version(torch.__version__.split("+")[0]) < Version("2.1.0"):
    raise SystemExit(f"Incompatible torch version: {torch.__version__}. Use >=2.1.0.")
print("runtime_ok", "torch=" + torch.__version__, "transformers=" + transformers.__version__)
PY
}

ensure_ann_index() {
  if [ "$RETRIEVAL_BACKEND" != "ann_numpy" ]; then
    return
  fi

  if [ -f "data/index_ann_${MEMORY_SIZE}.npz" ]; then
    say "ANN index already exists for size=${MEMORY_SIZE}"
    return
  fi

  say "Building ANN index from existing vectors for size=${MEMORY_SIZE}"
  "$PYTHON_BIN" - <<PY
import json
from pathlib import Path

import numpy as np

from src.vector_store import AnnNumpyVectorStore

size = int("${MEMORY_SIZE}")
vector_path = Path("data") / f"index_vectors_{size}.npy"
metadata_path = Path("data") / f"index_metadata_{size}.jsonl"
ann_index_path = Path("data") / f"index_ann_{size}.npz"

if not vector_path.exists() or not metadata_path.exists():
    raise SystemExit(
        f"Missing source index files for size {size}: "
        f"{vector_path} / {metadata_path}"
    )

vectors = np.load(vector_path).astype(np.float32)
metadata = []
with open(metadata_path, "r", encoding="utf-8") as handle:
    for line in handle:
        metadata.append(json.loads(line))

store = AnnNumpyVectorStore(
    vector_path=str(vector_path),
    metadata_path=str(metadata_path),
    ann_index_path=str(ann_index_path),
)
store.build(vectors, metadata)
print(f"ann_index_built size={size} output={ann_index_path}")
PY
}

run_case() {
  local label="$1"
  local execution_mode="$2"
  local memory_rounds="$3"
  local max_new_tokens="$4"

  say "Running case=$label mode=$execution_mode rounds=$memory_rounds max_new_tokens=$max_new_tokens gpu=$GPU_ID"
  export LOCAL_QWEN_CUDA_DEVICE="$GPU_ID"
  export LOCAL_QWEN_DIR

  "$PYTHON_BIN" scripts/run_profile.py \
    --memory-sizes "$MEMORY_SIZE" \
    --candidate-k $CANDIDATE_K \
    --top-k $TOP_K \
    --repeat "$REPEAT" \
    --timeout-per-query "$TIMEOUT_PER_QUERY" \
    --prefer-qdrant 0 \
    --retrieval-backend "$RETRIEVAL_BACKEND" \
    --execution-mode "$execution_mode" \
    --memory-rounds "$memory_rounds" \
    --llm-max-new-tokens "$max_new_tokens" \
    --llm-backend local_qwen \
    --embedding-backend local_qwen 2>&1 | tee -a "$LOG"

  cp results/raw_profile.csv "results/raw_profile_${label}.csv"
  cp results/profile_summary.csv "results/profile_summary_${label}.csv"
}

say "Experiment start"
say "LOCAL_QWEN_DIR=$LOCAL_QWEN_DIR"
say "MEMORY_SIZE=$MEMORY_SIZE REPEAT=$REPEAT CANDIDATE_K=$CANDIDATE_K TOP_K=$TOP_K"
say "RETRIEVAL_BACKEND=$RETRIEVAL_BACKEND"
check_env | tee -a "$LOG"
ensure_ann_index | tee -a "$LOG"

run_case "langgraph_4rounds_24tok" "langgraph" "4" "24"
run_case "langgraph_4rounds_16tok" "langgraph" "4" "16"
run_case "langgraph_4rounds_8tok" "langgraph" "4" "8"

say "Experiment finished"
