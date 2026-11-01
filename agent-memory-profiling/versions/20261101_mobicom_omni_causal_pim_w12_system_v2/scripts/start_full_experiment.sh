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

mkdir -p "$RUN_ROOT"

if [ -f "$RUN_ROOT/orchestrator.pid" ] && kill -0 "$(cat "$RUN_ROOT/orchestrator.pid")" 2>/dev/null; then
  echo "orchestrator already running: $(cat "$RUN_ROOT/orchestrator.pid")" >&2
  exit 1
fi
JUDGE_ALREADY_RUNNING=0
if [ -f "$RUN_ROOT/judge.pid" ] && kill -0 "$(cat "$RUN_ROOT/judge.pid")" 2>/dev/null; then
  JUDGE_ALREADY_RUNNING=1
fi

for name in orchestrator judge; do
  if [ -f "$RUN_ROOT/$name.pid" ] && ! kill -0 "$(cat "$RUN_ROOT/$name.pid")" 2>/dev/null; then
    mv "$RUN_ROOT/$name.pid" "$RUN_ROOT/$name.pid.stale.$(date +%s)"
  fi
done

cd "$ROOT"
"$PYTHON" scripts/run_full_resumable.py --config "$CONFIG" --seed-only \
  >> "$RUN_ROOT/seed.log" 2>&1

OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
TOKENIZERS_PARALLELISM=false \
  setsid -f "$PYTHON" scripts/run_full_resumable.py --config "$CONFIG" \
  >> "$RUN_ROOT/orchestrator.log" 2>&1 &
for _ in $(seq 1 30); do
  [ -s "$RUN_ROOT/orchestrator.pid" ] && break
  sleep 1
done

JUDGE_GPU="$($PYTHON - <<'PY' "$CONFIG"
import sys, yaml
config = yaml.safe_load(open(sys.argv[1]))
print(config['execution']['judge_gpu'])
PY
)"
JUDGE_MIN_FREE_GIB="$($PYTHON - <<'PY' "$CONFIG"
import sys, yaml
config = yaml.safe_load(open(sys.argv[1]))
print(config['execution'].get('judge_min_free_gib', 20))
PY
)"
JUDGE_FREE_GIB="$($PYTHON - <<'PY' "$JUDGE_GPU"
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
if [ "$JUDGE_ALREADY_RUNNING" = "1" ]; then
  JUDGE_MESSAGE="judge already running pid=$(cat "$RUN_ROOT/judge.pid")"
elif [ "${JUDGE_ENABLED:-auto}" != "0" ] \
   && [ ! -e "$RUN_ROOT/STOP_JUDGE" ] \
   && "$PYTHON" -c "import sys; raise SystemExit(0 if float(sys.argv[1]) >= float(sys.argv[2]) else 1)" \
        "$JUDGE_FREE_GIB" "$JUDGE_MIN_FREE_GIB"; then
  CUDA_VISIBLE_DEVICES="$JUDGE_GPU" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  OPENBLAS_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false \
    setsid -f "$PYTHON" scripts/judge_incremental.py --config "$CONFIG" \
    >> "$RUN_ROOT/judge.log" 2>&1 &
  for _ in $(seq 1 30); do
    [ -s "$RUN_ROOT/judge.pid" ] && break
    sleep 1
  done
  JUDGE_MESSAGE="judge started pid=$(cat "$RUN_ROOT/judge.pid")"
else
  JUDGE_MESSAGE="judge skipped; GPU $JUDGE_GPU free=${JUDGE_FREE_GIB} GiB, required=${JUDGE_MIN_FREE_GIB} GiB"
fi

sleep 2
printf 'started orchestrator pid=%s; %s\n' \
  "$(cat "$RUN_ROOT/orchestrator.pid")" "$JUDGE_MESSAGE"
printf 'run_root=%s\n' "$RUN_ROOT"
printf 'monitor=%s/scripts/monitor_full_experiment.sh\n' "$ROOT"
