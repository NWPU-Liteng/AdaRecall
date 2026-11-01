#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/../../.venvs/flamr-qwen35/bin/python}"
W12_CONFIG="$ROOT/configs/main_experiment_pilot112_w12.yaml"
HOST_CONFIG="$ROOT/configs/main_experiment_pilot112_host_control.yaml"
W12_ROOT="$ROOT/results/pilot112_w12_v1"
HOST_ROOT="$ROOT/results/pilot112_host_control_v1"
ANALYSIS="$ROOT/artifacts/analysis/pilot112_w12_v1"
BASELINE="$ROOT/../20261030_mobicom_omni_causal_pim_v3_fullresume/results/main_run_full_resume_v1"
PIM_REFERENCE="$ROOT/artifacts/pim_backend_w12_reference_20261030/closure_manifest.json"
STATUS="$ROOT/results/pilot112_closure_chain_status.json"
LOG="$ROOT/results/pilot112_closure_chain.log"
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-43200}"

mkdir -p "$ANALYSIS" "$(dirname "$STATUS")"

write_status() {
  "$PYTHON" - "$STATUS" "$1" "$2" <<'PY'
import json, sys
from datetime import datetime
from pathlib import Path
path, stage, detail = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
value = {
    "schema": "mobilemem-w12-pilot-closure-chain-v1",
    "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    "stage": stage,
    "detail": detail,
}
temporary = path.with_suffix(".tmp")
temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
temporary.replace(path)
PY
}

counts() {
  "$PYTHON" - "$1" "$2" <<'PY'
import json, sys
from pathlib import Path
root, expected = Path(sys.argv[1]), int(sys.argv[2])
reader = 0
for path in root.glob("user_*_shard_0.json"):
    try:
        document = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        continue
    if document.get("complete"):
        reader += len(document.get("rows") or [])
judge = len(list((root / "judge_rows").glob("*/*.json")))
print(reader, judge, int(reader >= expected and judge >= expected))
PY
}

wait_for_run() {
  local root="$1"
  local expected="$2"
  local deadline=$((SECONDS + MAX_WAIT_SECONDS))
  while [ "$SECONDS" -lt "$deadline" ]; do
    read -r reader judge done < <(counts "$root" "$expected")
    write_status "$3" "reader=$reader/$expected judge=$judge/$expected"
    if [ "$done" = "1" ]; then
      return 0
    fi
    if [ -f "$root/orchestrator_status.json" ]; then
      if "$PYTHON" - "$root/orchestrator_status.json" <<'PY'
import json, sys
value = json.load(open(sys.argv[1]))
raise SystemExit(0 if value.get("failed_users") else 1)
PY
      then
        write_status "$3" "failed_users_present"
        return 2
      fi
    fi
    sleep 60
  done
  write_status "$3" "timeout"
  return 3
}

{
  echo "[$(date -Is)] waiting for W12 pilot"
  wait_for_run "$W12_ROOT" 112 "waiting_w12_pilot"
  echo "[$(date -Is)] W12 pilot complete; running W12 audit"
  write_status "auditing_w12" "W12 pilot complete"
  "$PYTHON" "$ROOT/scripts/analyze_w12_system_run.py" \
    --config "$W12_CONFIG" \
    --baseline-reader-root "$BASELINE" \
    --baseline-judge-root "$BASELINE/judge_rows" \
    --pim-reference "$PIM_REFERENCE" \
    --output "$ANALYSIS/final_vs_old_host_baseline.json" \
    --bootstrap-samples 20000 --bootstrap-seed 20261031

  sleep 60
  echo "[$(date -Is)] launching same-algorithm Host control"
  write_status "host_control" "launching Host BF16 control"
  CONFIG="$HOST_CONFIG" bash "$ROOT/scripts/start_full_experiment.sh"
  wait_for_run "$HOST_ROOT" 112 "waiting_host_control"
  echo "[$(date -Is)] Host control complete; running paired audit"
  write_status "auditing_host_control" "Host control complete"
  "$PYTHON" "$ROOT/scripts/analyze_w12_system_run.py" \
    --config "$W12_CONFIG" \
    --baseline-reader-root "$HOST_ROOT" \
    --baseline-judge-root "$HOST_ROOT/judge_rows" \
    --pim-reference "$PIM_REFERENCE" \
    --output "$ANALYSIS/final_vs_same_algorithm_host.json" \
    --bootstrap-samples 20000 --bootstrap-seed 20261031
  write_status "complete" "W12 pilot and same-algorithm Host control audited"
  echo "[$(date -Is)] closure chain complete"
} >> "$LOG" 2>&1
