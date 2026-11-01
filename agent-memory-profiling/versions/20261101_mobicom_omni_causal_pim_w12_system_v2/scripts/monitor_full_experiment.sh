#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${CONFIG:-$ROOT/configs/main_experiment_pilot112_w12.yaml}"
PYTHON="${PYTHON:-$ROOT/../../.venvs/flamr-qwen35/bin/python}"
INTERVAL="${INTERVAL:-30}"

if [ "${1:-}" = "--once" ]; then
  exec "$PYTHON" "$ROOT/scripts/monitor_full_experiment.py" --config "$CONFIG"
fi
exec "$PYTHON" "$ROOT/scripts/monitor_full_experiment.py" \
  --config "$CONFIG" --watch "$INTERVAL"
