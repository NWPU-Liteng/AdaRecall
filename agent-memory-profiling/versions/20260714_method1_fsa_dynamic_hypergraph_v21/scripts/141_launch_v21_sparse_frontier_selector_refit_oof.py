#!/usr/bin/env python3
"""Launch immutable sparse-frontier selector refit OOF units."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


VERSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VERSION_ROOT / "src"))

from v21.io import load_config, read_json, write_json_new  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="v21_sparse_frontier_selector_refit_oof_r1.json")
    parser.add_argument("--run-id", default="v21_sparse_frontier_selector_refit_oof_r1_20260715")
    parser.add_argument("--attempt", default="attempt_001")
    parser.add_argument("--devices", default="cuda:0,cuda:2,cuda:3,cuda:6")
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.run_id != config["run_id"]:
        raise ValueError("run ID must match config")
    devices = [value.strip() for value in args.devices.split(",") if value.strip()]
    jobs = [(int(seed), fold) for seed in config["fixed_seeds"] for fold in range(int(config["outer_folds"]))]
    root = VERSION_ROOT / "artifacts/v21_sparse_frontier_selector_refit" / args.run_id
    log_dir = root / "launcher_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    def execute(index: int, seed: int, fold: int) -> dict:
        unit = root / f"seed_{seed}" / f"fold_{fold}" / "attempts" / args.attempt
        manifest_path = unit / "v21_sparse_frontier_selector_refit_manifest.json"
        if manifest_path.is_file():
            manifest = read_json(manifest_path)
            if manifest.get("status") == "completed":
                return {"seed": seed, "fold": fold, "status": "already_completed", "device": "existing"}
            raise ValueError(f"existing unit is incomplete: {manifest_path}")
        device = devices[index % len(devices)]
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        log_path = log_dir / f"unit_seed_{seed}_fold_{fold}_{args.attempt}_{timestamp}.log"
        command = [
            sys.executable, "-u", str(VERSION_ROOT / "scripts/140_train_v21_sparse_frontier_selector_refit_oof.py"),
            "--config", args.config, "--run-id", args.run_id, "--seed", str(seed), "--fold", str(fold),
            "--attempt", args.attempt, "--device", device,
        ]
        result = subprocess.run(command, cwd=VERSION_ROOT, text=True, capture_output=True)
        with log_path.open("x", encoding="utf-8") as handle:
            handle.write(result.stdout)
            if result.stderr:
                handle.write("\n[stderr]\n" + result.stderr)
        if result.returncode:
            raise RuntimeError(f"unit failed seed={seed} fold={fold}; see {log_path}")
        return {"seed": seed, "fold": fold, "status": "completed", "device": device, "log": str(log_path.relative_to(VERSION_ROOT))}

    results, failures = [], []
    with ThreadPoolExecutor(max_workers=min(args.max_workers, len(devices))) as executor:
        futures = {executor.submit(execute, index, seed, fold): (seed, fold) for index, (seed, fold) in enumerate(jobs)}
        for future in as_completed(futures):
            seed, fold = futures[future]
            try:
                row = future.result()
                results.append(row)
            except Exception as error:
                row = {"seed": seed, "fold": fold, "status": "failed", "error": str(error)}
                failures.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    summary_path = log_dir / f"launcher_summary_{args.attempt}_{timestamp}.json"
    write_json_new(summary_path, {
        "schema_version": "v21-sparse-frontier-refit-launcher-1",
        "status": "completed" if not failures else "failed",
        "run_id": args.run_id,
        "attempt": args.attempt,
        "results": sorted(results, key=lambda row: (row["seed"], row["fold"])),
        "failures": failures,
        "calibration_read": False,
        "formal_test_read": False,
    })
    if failures:
        raise RuntimeError(f"{len(failures)} units failed; see {summary_path}")
    print(json.dumps({"status": "completed", "units": len(results), "summary": str(summary_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
