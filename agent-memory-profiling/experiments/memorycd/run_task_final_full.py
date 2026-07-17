#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version-dir", type=Path, required=True)
    parser.add_argument("--protocol", choices=["official_single", "official_cross", "causal_cross"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--embedding-cache-dir", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", default=["rating_prediction", "item_ranking"])
    parser.add_argument("--meta-dir", type=Path, default=Path("data/public_datasets/memorycd/meta"))
    parser.add_argument("--reranker-batch-size", type=int, default=16)
    args = parser.parse_args()

    pareto_path = args.version_dir / "reports" / "pareto_budget_report.json"
    payload = json.loads(pareto_path.read_text(encoding="utf-8"))
    selected = [row for row in payload["groups"] if row["protocol"] == args.protocol and row["task"] in set(args.tasks)]
    if not selected:
        raise SystemExit(f"No Pareto rows found in {pareto_path} for {args.protocol} and tasks {args.tasks}")

    by_task: dict[str, dict] = defaultdict(dict)
    for row in selected:
        by_task[row["task"]][row["setting"]] = row

    args.output_dir.mkdir(parents=True, exist_ok=True)
    command_log = []
    for task, setting_rows in sorted(by_task.items()):
        selected_methods = {"recency_k8", "dense_k8"}
        for group in setting_rows.values():
            if group.get("reference_high_budget"):
                selected_methods.add(group["reference_high_budget"]["method"])
            if group.get("pareto_min_budget"):
                selected_methods.add(group["pareto_min_budget"]["method"])
        cmd = [
            sys.executable,
            "experiments/memorycd/run_task_quality_curve.py",
            "--version-dir",
            str(args.version_dir),
            "--protocol",
            args.protocol,
            "--output-dir",
            str(args.output_dir),
            "--gpu",
            str(args.gpu),
            "--embedding-cache-dir",
            str(args.embedding_cache_dir),
            "--meta-dir",
            str(args.meta_dir),
            "--tasks",
            task,
            "--methods",
            *sorted(selected_methods),
            "--max-events-per-stratum",
            "969",
            "--reranker-batch-size",
            str(args.reranker_batch_size),
        ]
        command_log.append({"task": task, "methods": sorted(selected_methods), "command": cmd, "note": "Full run uses the same task runner with full stratum cap but only the four formal methods."})
        env = os.environ.copy()
        env["LOCAL_QWEN_CUDA_DEVICE"] = str(args.gpu)
        subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True, env=env)

    manifest_path = args.output_dir / f"{args.protocol}_task_final_full_manifest.json"
    manifest_path.write_text(json.dumps({"protocol": args.protocol, "commands": command_log}, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "tasks": sorted(by_task)}, indent=2))


if __name__ == "__main__":
    main()
