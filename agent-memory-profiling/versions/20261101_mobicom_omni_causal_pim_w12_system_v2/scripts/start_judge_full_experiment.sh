#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${CONFIG:-$ROOT/configs/main_experiment_pilot112_w12.yaml}"
PYTHON="${PYTHON:-$ROOT/../../.venvs/flamr-qwen35/bin/python}"
RUN_ROOT="$($PYTHON - <<'PY' "$CONFIG"
import sys, yaml
config = yaml.safe_load(open(sys.argv[1]))
print(config['paths']['run_root'])
PY
)"
JUDGE_GPU="$($PYTHON - <<'PY' "$CONFIG"
import sys, yaml
config = yaml.safe_load(open(sys.argv[1]))
print(config['execution']['judge_gpu'])
PY
)"
MIN_FREE_GIB="$($PYTHON - <<'PY' "$CONFIG"
import sys, yaml
config = yaml.safe_load(open(sys.argv[1]))
print(config['execution'].get('judge_min_free_gib', 20))
PY
)"

if [ "${1:-}" = "--clear-stop" ] && [ -e "$RUN_ROOT/STOP_JUDGE" ]; then
  mv "$RUN_ROOT/STOP_JUDGE" "$RUN_ROOT/STOP_JUDGE.cleared.$(date +%s)"
fi
if [ -e "$RUN_ROOT/STOP_JUDGE" ]; then
  echo "STOP_JUDGE exists; rerun with --clear-stop after confirming the stop" >&2
  exit 1
fi
if [ -f "$RUN_ROOT/judge.pid" ] && kill -0 "$(cat "$RUN_ROOT/judge.pid")" 2>/dev/null; then
  echo "judge already running: $(cat "$RUN_ROOT/judge.pid")" >&2
  exit 1
fi
if [ -f "$RUN_ROOT/judge.pid" ]; then
  mv "$RUN_ROOT/judge.pid" "$RUN_ROOT/judge.pid.stale.$(date +%s)"
fi

FREE_GIB="$($PYTHON - <<'PY' "$JUDGE_GPU"
import subprocess, sys
gpu = sys.argv[1]
rows = subprocess.check_output([
    'nvidia-smi', '--query-gpu=index,memory.total,memory.used',
    '--format=csv,noheader,nounits'], text=True)
for line in rows.splitlines():
    index, total, used = [item.strip() for item in line.split(',')]
    if index == gpu:
        print(f'{(int(total) - int(used)) / 1024:.1f}')
        break
else:
    print('0')
PY
)"
if ! "$PYTHON" -c "import sys; raise SystemExit(0 if float(sys.argv[1]) >= float(sys.argv[2]) else 1)" \
     "$FREE_GIB" "$MIN_FREE_GIB"; then
  echo "GPU $JUDGE_GPU has ${FREE_GIB} GiB free; ${MIN_FREE_GIB} GiB required" >&2
  exit 2
fi

CUDA_VISIBLE_DEVICES="$JUDGE_GPU" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
OPENBLAS_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false \
  setsid -f "$PYTHON" "$ROOT/scripts/judge_incremental.py" --config "$CONFIG" \
  >> "$RUN_ROOT/judge.log" 2>&1 &
for _ in $(seq 1 30); do
  [ -s "$RUN_ROOT/judge.pid" ] && break
  sleep 1
done
echo "judge started pid=$(cat "$RUN_ROOT/judge.pid") gpu=$JUDGE_GPU"
