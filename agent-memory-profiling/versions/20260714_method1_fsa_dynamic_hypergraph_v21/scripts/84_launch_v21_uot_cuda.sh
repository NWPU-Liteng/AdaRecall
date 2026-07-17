#!/usr/bin/env bash
set -euo pipefail

VERSION_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

DEVICE="cuda:2"
RETRIES="5"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --cuda-init-retries)
      RETRIES="$2"
      shift 2
      ;;
    --)
      shift
      break
      ;;
    *)
      break
      ;;
  esac
done

BOOTSTRAP=$(cat <<'PYTHON'
torch = __import__("torch")
try:
    torch.cuda.init()
except Exception as error:
    print(f"V21_CUDA_CONTEXT_FAILURE {type(error).__name__}: {error}", flush=True)
    raise SystemExit(75)
import sys
version_root = sys.argv[1]
device = sys.argv[2]
device_index = int(device.split(":", 1)[1]) if ":" in device else 0
torch.cuda.set_device(device_index)
torch.empty(1, device=device)
torch.cuda.synchronize(device_index)
sys.path.insert(0, version_root + "/src")
sys.argv = [sys.argv[0], "--device", device, "--cuda-init-retries", "1", *sys.argv[3:]]
from v21.uot_cuda_bootstrap import main
main(context_already_initialized=True, context_attempt=1)
PYTHON
)

for ((attempt = 1; attempt <= RETRIES; attempt += 1)); do
  output=""
  if output=$(conda run --no-capture-output -n agent-memory-profiling \
    python -c "$BOOTSTRAP" "$VERSION_ROOT" "$DEVICE" "$@" 2>&1); then
    printf '%s\n' "$output"
    exit 0
  else
    status=$?
  fi
  printf '%s\n' "$output" >&2
  # conda run may translate the bootstrap's exit 75 into 127. The explicit
  # marker distinguishes a retryable fresh-process CUDA initialization failure
  # from a worker or launcher failure, which must never be rerun implicitly.
  if [[ "$output" != *"V21_CUDA_CONTEXT_FAILURE"* || $attempt -eq $RETRIES ]]; then
    exit "$status"
  fi
  sleep 2
done
