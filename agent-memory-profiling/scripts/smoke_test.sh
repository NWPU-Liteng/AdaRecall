#!/usr/bin/env bash
set -u
mkdir -p logs results data
LOG="logs/smoke_test.log"
: > "$LOG"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-agent-memory}"
mkdir -p "$MPLCONFIGDIR"

if [ -z "${CONDA_PREFIX:-}" ] && [ -f .venv/bin/activate ]; then
  source .venv/bin/activate 2>/dev/null || true
fi

set -x
timeout 60s python scripts/generate_memories.py --sizes 100 2>&1 | tee -a "$LOG"
timeout 120s python scripts/build_index.py --input data/synthetic_memories_100.jsonl --prefer-qdrant 1 --embedding-backend auto 2>&1 | tee -a "$LOG"
timeout 120s python scripts/run_profile.py --memory-sizes 100 --candidate-k 20 --top-k 5 --repeat 2 --timeout-per-query 20 --llm-backend auto --embedding-backend auto 2>&1 | tee -a "$LOG"
timeout 120s python scripts/build_index.py --input data/synthetic_memories_100.jsonl --prefer-qdrant 0 --retrieval-backend ann_numpy --embedding-backend auto 2>&1 | tee -a "$LOG"
timeout 120s python scripts/run_profile.py --memory-sizes 100 --candidate-k 20 --top-k 5 --repeat 2 --timeout-per-query 20 --prefer-qdrant 0 --retrieval-backend ann_numpy --llm-backend auto --embedding-backend auto 2>&1 | tee -a "$LOG"
timeout 60s python scripts/analyze_profile.py --input results/raw_profile.csv 2>&1 | tee -a "$LOG"
set +x

echo "Smoke test finished. Check results/raw_profile.csv and results/profile_summary.csv"
