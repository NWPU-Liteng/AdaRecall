#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    return vals[min(len(vals) - 1, int(round((len(vals) - 1) * p)))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version-dir", type=Path, required=True)
    args = parser.parse_args()
    rows_by_key: dict[tuple, list[dict]] = defaultdict(list)
    for trace_path in (args.version_dir / "runs" / "selection_sweep").glob("*_selection_sweep.jsonl"):
        with trace_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                key = (row["protocol"], row["task"], row["setting"], row["R_requested"], row["B_requested"])
                rows_by_key[key].append(row)

    summary_path = args.version_dir / "reports" / "candidate_frontier_sweep_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "protocol",
                "task",
                "setting",
                "R",
                "B",
                "events",
                "median_N",
                "p50_selection_ms",
                "p95_selection_ms",
                "p99_selection_ms",
                "p50_rerank_ms",
                "p95_rerank_ms",
                "avg_payload_bytes_for_rerank",
            ],
        )
        writer.writeheader()
        for key, rows in sorted(rows_by_key.items()):
            protocol, task, setting, r, b = key
            selection = [float(row["t_total_selection_ms"]) for row in rows]
            rerank = [float(row["t_rerank_ms"]) for row in rows]
            payload = [float(row["payload_bytes_for_rerank"]) for row in rows]
            writer.writerow(
                {
                    "protocol": protocol,
                    "task": task,
                    "setting": setting,
                    "R": r,
                    "B": b,
                    "events": len(rows),
                    "median_N": median([int(row["N"]) for row in rows]),
                    "p50_selection_ms": pct(selection, 0.50),
                    "p95_selection_ms": pct(selection, 0.95),
                    "p99_selection_ms": pct(selection, 0.99),
                    "p50_rerank_ms": pct(rerank, 0.50),
                    "p95_rerank_ms": pct(rerank, 0.95),
                    "avg_payload_bytes_for_rerank": sum(payload) / len(payload) if payload else 0.0,
                }
            )

    report_path = args.version_dir / "reports" / "candidate_frontier_sweep_report.md"
    report_path.write_text(
        "# Candidate Frontier Sweep Report\n\n"
        f"Summary CSV: `{summary_path}`\n\n"
        "This report summarizes the V5.1 selection sweep traces. Full Qwen task-quality "
        "curves are intentionally not inferred from selection-only runs.\n",
        encoding="utf-8",
    )
    print(json.dumps({"summary": str(summary_path), "report": str(report_path)}, indent=2))


if __name__ == "__main__":
    main()
