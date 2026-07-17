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


WORKER = VERSION_ROOT / "scripts" / "87_train_v21_projection_uot_oof.py"


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch independent v21 Phase 1 seed/fold OOF units.")
    parser.add_argument("--unit-start", type=int, required=True)
    parser.add_argument("--unit-stop", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--run-id", default="v21_phase1_oof_r1_20260714")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    args = parser.parse_args()
    phase = load_config("v21_phase1_oof.json")
    units = [(seed, fold) for seed in phase["fixed_seeds"] for fold in range(int(phase["outer_folds"]))]
    if not 1 <= args.unit_start <= args.unit_stop <= len(units):
        raise ValueError("unit interval must be within 1..25")
    run_dir = VERSION_ROOT / "artifacts" / "v21_phase1" / args.run_id
    log_dir = run_dir / "launcher_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for unit_index in range(args.unit_start, args.unit_stop + 1):
        seed, fold = units[unit_index - 1]
        unit_dir = run_dir / f"seed_{seed}" / f"fold_{fold}"
        manifest = unit_dir / "v21_unit_manifest.json"
        if manifest.is_file():
            if not args.resume:
                results.append({"unit": unit_index, "seed": seed, "fold": fold, "status": "existing_requires_resume"})
                continue
            results.append({"unit": unit_index, "seed": seed, "fold": fold, "status": "skipped_completed"})
            continue
        if unit_dir.exists():
            results.append({"unit": unit_index, "seed": seed, "fold": fold, "status": "incomplete_existing_unit_requires_new_run_id"})
            continue
        command = [
            sys.executable, str(WORKER), "--seed", str(seed), "--fold", str(fold),
            "--device", args.device, "--run-id", args.run_id,
        ]
        try:
            completed = subprocess.run(
                command, cwd=VERSION_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, timeout=args.timeout_seconds, check=False,
            )
            output = completed.stdout or ""
            status = "completed" if completed.returncode == 0 and manifest.is_file() else "failed"
            exit_code = completed.returncode
        except subprocess.TimeoutExpired as error:
            output = str(error.stdout or "")
            status = "timed_out"
            exit_code = None
        log_path = log_dir / f"v21_phase1_unit_{unit_index:02d}_seed_{seed}_fold_{fold}_{args.device.replace(':', '_')}.log"
        with log_path.open("x", encoding="utf-8") as handle:
            handle.write(output)
        record = {
            "unit": unit_index, "seed": seed, "fold": fold, "status": status,
            "exit_code": exit_code, "log": str(log_path.relative_to(VERSION_ROOT)),
        }
        if status != "completed":
            failure = log_dir / f"v21_phase1_unit_{unit_index:02d}_failure_{now()}.json"
            write_json_new(failure, {**record, "run_id": args.run_id, "calibration_read": False, "formal_test_read": False})
            record["failure_record"] = str(failure.relative_to(VERSION_ROOT))
        results.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
    summary = {
        "schema_version": "v21-phase1-launcher-summary-1",
        "run_id": args.run_id,
        "unit_interval": [args.unit_start, args.unit_stop],
        "device": args.device,
        "results": results,
        "calibration_read": False,
        "formal_test_read": False,
    }
    summary_path = log_dir / (
        f"v21_phase1_launcher_summary_{now()}_units_{args.unit_start:02d}_{args.unit_stop:02d}_"
        f"{args.device.replace(':', '_')}.json"
    )
    write_json_new(summary_path, summary)
    if any(row["status"] not in {"completed", "skipped_completed"} for row in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
