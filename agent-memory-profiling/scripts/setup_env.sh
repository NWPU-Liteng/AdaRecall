#!/usr/bin/env bash
set -u
mkdir -p logs
LOG="logs/setup_env.log"
: > "$LOG"

[ -f .env.runtime ] && set -a && source .env.runtime && set +a

if [ -n "${CONDA_PREFIX:-}" ]; then
  echo "Detected active conda environment: $CONDA_PREFIX" | tee -a "$LOG"
  python -m pip install --upgrade pip --timeout 20 --retries 2 2>&1 | tee -a "$LOG" || true
else
  python3 -m venv .venv 2>&1 | tee -a "$LOG" || true
  source .venv/bin/activate
  python -m pip install --upgrade pip --timeout 20 --retries 2 2>&1 | tee -a "$LOG" || true
fi

if [ "${EXPERIMENT_OFFLINE_MODE:-0}" = "1" ]; then
  echo "Offline mode: skip dependency installation" | tee -a "$LOG"
  exit 0
fi

timeout 180s python -m pip install --timeout 20 --retries 2 -r requirements.txt 2>&1 | tee -a "$LOG" || {
  echo "Dependency installation failed. Continue with available packages and fallback mode." | tee -a "$LOG"
  exit 0
}
