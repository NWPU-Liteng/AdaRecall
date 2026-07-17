#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


VERSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VERSION_ROOT / "src"))

from v21.io import load_config, write_json_new  # noqa: E402


WORKER = VERSION_ROOT / "scripts" / "112_build_v21_sparse_semantic_index.py"


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch immutable CPU v21 sparse semantic-index OOF units.")
    parser.add_argument("--unit-start", type=int, required=True)
    parser.add_argument("--unit-stop", type=int, required=True)
    parser.add_argument("--run-id", default="v21_sparse_semantic_index_r1_20260714")
    parser.add_argument("--attempt", default="attempt_001")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    args = parser.parse_args()
    config = load_config("v21_sparse_joint.json")
    if args.run_id != config["index_run_id"]:
        raise ValueError("run ID must match sparse-index preregistration")
    units = [(int(seed), fold) for seed in config["fixed_seeds"] for fold in range(int(config["outer_folds"]))]
    if not 1 <= args.unit_start <= args.unit_stop <= len(units):
        raise ValueError("unit interval must be within 1..25")
    run_dir = VERSION_ROOT / "artifacts" / "v21_sparse_semantic_index" / args.run_id
    log_dir = run_dir / "launcher_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for unit in range(args.unit_start, args.unit_stop + 1):
        seed, fold = units[unit - 1]
        attempt_dir = run_dir / f"seed_{seed}" / f"fold_{fold}" / "attempts" / args.attempt
        manifest = attempt_dir / "v21_sparse_semantic_index_manifest.json"
        if attempt_dir.exists():
            results.append({"unit": unit, "seed": seed, "fold": fold, "status": "skipped_completed" if manifest.is_file() else "existing_attempt_requires_new_attempt"})
            continue
        command = [sys.executable, str(WORKER), "--seed", str(seed), "--fold", str(fold), "--run-id", args.run_id, "--attempt", args.attempt]
        try:
            completed = subprocess.run(command, cwd=VERSION_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=args.timeout_seconds, check=False)
            output, exit_code = completed.stdout or "", completed.returncode
            status = "completed" if exit_code == 0 and manifest.is_file() else "failed"
        except subprocess.TimeoutExpired as error:
            output, exit_code, status = str(error.stdout or ""), None, "timed_out"
        log = log_dir / f"v21_sparse_index_unit_{unit:02d}_seed_{seed}_fold_{fold}_{args.attempt}_{stamp()}.log"
        with log.open("x", encoding="utf-8") as handle:
            handle.write(output)
        results.append({"unit": unit, "seed": seed, "fold": fold, "status": status, "exit_code": exit_code, "log": str(log.relative_to(VERSION_ROOT))})
    output = log_dir / f"v21_sparse_index_launcher_summary_{args.attempt}_{stamp()}_units_{args.unit_start}_{args.unit_stop}.json"
    write_json_new(output, {"schema_version": "v21-sparse-index-launcher-1", "run_id": args.run_id, "attempt": args.attempt, "results": results, "calibration_read": False, "formal_test_read": False})
    print(json.dumps({"completed": sum(item["status"] == "completed" for item in results), "results": results}, sort_keys=True))


if __name__ == "__main__":
    main()
