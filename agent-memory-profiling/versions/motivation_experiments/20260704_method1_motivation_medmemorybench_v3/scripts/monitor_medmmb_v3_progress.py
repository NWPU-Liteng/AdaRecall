#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


ARTIFACTS = [
    ("M1 invocations", "aggregate/M1_history_scale_invocations.jsonl", "jsonl"),
    ("M1 summary", "aggregate/M1_history_scale_summary.csv", "csv"),
    ("M2 invocations", "aggregate/M2_frontier_budget_invocations.jsonl", "jsonl"),
    ("M2 occupancy", "aggregate/M2_frontier_occupancy.csv", "csv"),
    ("M2 summary", "aggregate/M2_frontier_budget_summary.csv", "csv"),
    ("M3 scattering", "aggregate/M3_tile_scattering.csv", "csv"),
    ("O1 smoke trace", "aggregate/O1_smoke_stage_trace.jsonl", "jsonl"),
    ("O1 pilot trace", "aggregate/O1_pilot_stage_trace.jsonl", "jsonl"),
    ("O1 formal trace", "aggregate/O1_formal_stage_trace.jsonl", "jsonl"),
    ("O1 pilot same/cross", "aggregate/O1_pilot_same_cross_stage_stats.csv", "csv"),
    ("O1 pilot shuffle null", "aggregate/O1_pilot_stage_shuffle_null.csv", "csv"),
    ("O1 same/cross", "aggregate/O1_same_cross_stage_stats.csv", "csv"),
    ("O1 shuffle null", "aggregate/O1_stage_shuffle_null.csv", "csv"),
    ("validation", "reports/VALIDATION.md", "text"),
]


EXPECTED_FORMAL_ROWS = {
    "M2 invocations": 240 * 4,
    "O1 formal trace": 120 * 4,
}

EXPECTED_FORMAL_QUERIES = {
    "M1 invocations": 240,
    "M2 invocations": 240,
    "M3 scattering": 240,
    "O1 formal trace": 120,
}

PROGRESS_FILES = [
    "runs/progress/M1_formal_progress.json",
    "runs/progress/M2_formal_progress.json",
    "runs/progress/O1_formal_progress.json",
    "runs/progress/O1_pilot_progress.json",
    "runs/progress/M1_smoke_progress.json",
    "runs/progress/M2_smoke_progress.json",
    "runs/progress/O1_smoke_progress.json",
]


def _run(cmd: list[str]) -> str:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return out.strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def _line_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return sum(1 for _ in handle)


def _csv_data_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        rows = sum(1 for _ in reader)
    return max(0, rows - 1)


def _mtime(path: Path) -> str:
    if not path.exists():
        return "-"
    return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")


def _size(path: Path) -> str:
    if not path.exists():
        return "-"
    value = path.stat().st_size
    units = ["B", "KB", "MB", "GB"]
    size = float(value)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f}{unit}"
        size /= 1024.0
    return str(value)


def _mode_counts_jsonl(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not path.exists():
        return counts
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except Exception:
                continue
            mode = str(row.get("mode") or row.get("run_id", "").split("_")[1:2] or "unknown")
            counts[mode] = counts.get(mode, 0) + 1
    return counts


def _jsonl_query_progress(path: Path, *, mode: str = "formal") -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    queries: set[str] = set()
    rows = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except Exception:
                continue
            row_mode = str(row.get("mode", ""))
            run_id = str(row.get("run_id", ""))
            if row_mode != mode and f"_{mode}_" not in run_id:
                continue
            rows += 1
            if row.get("query_id") is not None:
                queries.add(str(row["query_id"]))
    return len(queries), rows


def _csv_query_progress(path: Path, *, mode: str = "formal") -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    queries: set[str] = set()
    rows = 0
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row_mode = str(row.get("mode", ""))
            run_id = str(row.get("run_id", ""))
            if row_mode and row_mode != mode and f"_{mode}_" not in run_id:
                continue
            rows += 1
            if row.get("query_id"):
                queries.add(str(row["query_id"]))
    return len(queries), rows


def _pct(done: int, total: int) -> str:
    if total <= 0:
        return "-"
    return f"{100.0 * done / total:5.1f}%"


def _active_experiment_processes() -> str:
    ps = _run(["ps", "-eo", "pid,etime,pcpu,pmem,args"])
    lines = []
    needles = [
        "run_m1_history_scale.py",
        "run_m2_frontier_budget.py",
        "run_m3_tile_scattering.py",
        "run_o1_stage_trace.py",
        "compute_o1_null_models.py",
        "make_motivation_figures.py",
        "validate_medmemorybench_motivation.py",
    ]
    for line in ps.splitlines():
        if any(needle in line for needle in needles):
            lines.append(line)
    return "\n".join(lines) if lines else "no matching experiment process"


def _gpu_status() -> str:
    query = "index,name,memory.used,memory.total,utilization.gpu,temperature.gpu"
    return _run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"])


def _gpu_processes() -> str:
    query = "pid,gpu_uuid,used_memory,process_name"
    return _run(["nvidia-smi", f"--query-compute-apps={query}", "--format=csv,noheader"])


def _print_artifacts() -> None:
    print("Artifacts")
    print("---------")
    for label, rel, kind in ARTIFACTS:
        path = ROOT / rel
        if kind == "jsonl":
            rows = _line_count(path)
            expected = EXPECTED_FORMAL_ROWS.get(label)
            suffix = f" / expected formal {expected} ({_pct(rows, expected)} raw rows)" if expected else ""
            modes = _mode_counts_jsonl(path)
            mode_text = f" modes={modes}" if modes else ""
            q_expected = EXPECTED_FORMAL_QUERIES.get(label)
            q_done, formal_rows = _jsonl_query_progress(path)
            q_text = f" formal_queries={q_done}/{q_expected} ({_pct(q_done, q_expected)}) formal_rows={formal_rows}" if q_expected else ""
            print(f"{label:18} rows={rows}{suffix:>42} size={_size(path):>8} mtime={_mtime(path)}{q_text}{mode_text}")
        elif kind == "csv":
            rows = _csv_data_rows(path)
            expected = EXPECTED_FORMAL_ROWS.get(label)
            suffix = f" / expected formal {expected} ({_pct(rows, expected)} raw rows)" if expected else ""
            q_expected = EXPECTED_FORMAL_QUERIES.get(label)
            q_done, formal_rows = _csv_query_progress(path)
            q_text = f" formal_queries={q_done}/{q_expected} ({_pct(q_done, q_expected)}) formal_rows={formal_rows}" if q_expected else ""
            print(f"{label:18} rows={rows}{suffix:>42} size={_size(path):>8} mtime={_mtime(path)}{q_text}")
        else:
            exists = "present" if path.exists() else "missing"
            print(f"{label:18} {exists:>8} size={_size(path):>8} mtime={_mtime(path)}")


def _print_heartbeats() -> None:
    print("Progress heartbeats")
    print("-------------------")
    any_seen = False
    for rel in PROGRESS_FILES:
        path = ROOT / rel
        if not path.exists():
            continue
        any_seen = True
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"{rel}: unreadable: {exc}")
            continue
        total_queries = int(payload.get("total_queries") or 0)
        completed_queries = int(payload.get("completed_queries") or 0)
        total_units = int(payload.get("total_units") or 0)
        completed_units = int(payload.get("completed_units") or 0)
        print(
            f"{rel}: queries={completed_queries}/{total_queries} ({_pct(completed_queries, total_queries)}), "
            f"units={completed_units}/{total_units} ({_pct(completed_units, total_units)}), "
            f"current={payload.get('current_query_id', '-')}, updated={payload.get('updated_at', '-')}"
        )
    if not any_seen:
        print("no heartbeat files yet; current long-running jobs started before heartbeat support will only update final artifacts")


def print_status() -> None:
    print("=" * 88)
    print(f"MedMemoryBench v3 progress @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"root={ROOT}")
    print()
    print("Active experiment processes")
    print("---------------------------")
    print(_active_experiment_processes())
    print()
    print("GPU status")
    print("----------")
    print(_gpu_status())
    print()
    print("GPU compute processes")
    print("---------------------")
    print(_gpu_processes())
    print()
    _print_artifacts()
    print()
    _print_heartbeats()
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor MedMemoryBench v3 experiment progress.")
    parser.add_argument("--watch", action="store_true", help="Refresh repeatedly.")
    parser.add_argument("--interval", type=float, default=60.0, help="Refresh interval in seconds for --watch.")
    args = parser.parse_args()

    if args.watch:
        while True:
            os.system("clear")
            print_status()
            time.sleep(max(1.0, args.interval))
    else:
        print_status()


if __name__ == "__main__":
    main()
