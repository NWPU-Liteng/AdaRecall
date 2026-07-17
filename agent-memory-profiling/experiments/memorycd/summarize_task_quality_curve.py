#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.memorycd.audit import iter_jsonl
from src.memorycd.task_runtime import (
    SimpleItemRankingEvaluator,
    SimpleTextEvaluator,
    bootstrap_confidence_interval,
    derive_delta,
    summarize_latency,
    write_json,
)


FORMAL_TASKS = {
    "rating_prediction",
    "item_ranking",
    "review_generation",
    "review_summarization",
}


def _formal_trace_paths(input_dir: Path) -> list[Path]:
    formal_paths: list[Path] = []
    for trace_path in sorted(input_dir.glob("*_task_curve_subset.jsonl")):
        stem = trace_path.stem
        if not any(task in stem for task in FORMAL_TASKS):
            continue
        formal_paths.append(trace_path)
    return formal_paths


def _metric_name(task: str) -> str:
    if task == "rating_prediction":
        return "rmse"
    if task == "item_ranking":
        return "ndcg@5"
    if task in {"review_generation", "review_summarization"}:
        return "rougeL"
    raise ValueError(task)


def _metric_value(task: str, rows: list[dict]) -> tuple[float, list[float]]:
    if task == "rating_prediction":
        errors = [float(row["metrics"]["error"]) for row in rows]
        return math.sqrt(sum(e * e for e in errors) / len(errors)), [abs(e) for e in errors]
    if task == "item_ranking":
        vals = [float(row["metrics"]["ndcg@5"]) for row in rows]
        return sum(vals) / len(vals), vals
    vals = [float(row["metrics"].get("rougeL_f", row["metrics"].get("rougeL", 0.0))) for row in rows]
    return sum(vals) / len(vals), vals


def _attach_metrics(task: str, prediction: object, label: dict) -> dict[str, float]:
    if task == "rating_prediction":
        error = abs(float(prediction) - float(label["gold_rating"]))
        return {"error": error}
    if task == "item_ranking":
        evaluator = SimpleItemRankingEvaluator()
        gt = str(label["positive_parent_asin"])
        ranked = list(prediction)
        return {
            "ndcg@1": evaluator.ndcg_at_k(ranked, gt, 1),
            "ndcg@3": evaluator.ndcg_at_k(ranked, gt, 3),
            "ndcg@5": evaluator.ndcg_at_k(ranked, gt, 5),
            "recall@1": evaluator.recall_at_k(ranked, gt, 1),
            "recall@3": evaluator.recall_at_k(ranked, gt, 3),
            "recall@5": evaluator.recall_at_k(ranked, gt, 5),
        }
    evaluator = SimpleTextEvaluator()
    if task == "review_generation":
        return evaluator.add_prediction(str(prediction), str(label.get("gold_review_text") or ""))
    if task == "review_summarization":
        return evaluator.add_prediction(str(prediction), str(label.get("gold_review_title") or ""))
    raise ValueError(task)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    args = parser.parse_args()

    input_dir = args.version_dir / "runs" / "task_curve_subset"
    rows_by_key: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    labels_by_protocol: dict[str, dict[str, dict]] = {}
    for trace_path in _formal_trace_paths(input_dir):
        for line in trace_path.open("r", encoding="utf-8"):
            row = json.loads(line)
            protocol = row["protocol"]
            if protocol not in labels_by_protocol:
                label_path = args.version_dir / "manifests" / f"memorycd_{protocol}" / "evaluation_labels.jsonl"
                labels_by_protocol[protocol] = {label_row["event_id"]: label_row for label_row in iter_jsonl(label_path)}
            label = labels_by_protocol[row["protocol"]][row["event_id"]]
            row["metrics"] = _attach_metrics(row["task"], row["prediction"], label)
            key = (row["protocol"], row["task"], row["setting"], row["method"])
            rows_by_key[key].append(row)

    summary_csv = args.version_dir / "reports" / "task_quality_curve_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "protocol",
                "task",
                "setting",
                "method",
                "events",
                "metric_name",
                "metric_value",
                "ci_low",
                "ci_high",
                "p50_selection_ms",
                "p95_selection_ms",
                "p99_selection_ms",
                "p50_task_ms",
                "p95_task_ms",
                "p99_task_ms",
                "parse_failure_rate",
            ],
        )
        writer.writeheader()
        for key, rows in sorted(rows_by_key.items()):
            protocol, task, setting, method = key
            eval_rows = [row for row in rows if not row.get("warmup_event")]
            if not eval_rows:
                continue
            metric_value, bootstrap_source = _metric_value(task, eval_rows)
            ci_low, ci_high = bootstrap_confidence_interval(
                bootstrap_source if task != "rating_prediction" else [v * v for v in bootstrap_source],
                num_samples=args.bootstrap_samples,
            )
            if task == "rating_prediction":
                ci_low = math.sqrt(max(ci_low, 0.0))
                ci_high = math.sqrt(max(ci_high, 0.0))
            selection_stats = summarize_latency([row["t_total_selection_ms"] for row in eval_rows])
            task_stats = summarize_latency([row["t_total_task_ms"] for row in eval_rows])
            parse_failures = sum(1 for row in eval_rows if row["backend_mode"] == "local_qwen" and (row["prediction"] == "" or row["prediction"] == []))
            writer.writerow(
                {
                    "protocol": protocol,
                    "task": task,
                    "setting": setting,
                    "method": method,
                    "events": len(eval_rows),
                    "metric_name": _metric_name(task),
                    "metric_value": metric_value,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "p50_selection_ms": selection_stats["p50"],
                    "p95_selection_ms": selection_stats["p95"],
                    "p99_selection_ms": selection_stats["p99"],
                    "p50_task_ms": task_stats["p50"],
                    "p95_task_ms": task_stats["p95"],
                    "p99_task_ms": task_stats["p99"],
                    "parse_failure_rate": parse_failures / len(eval_rows),
                }
            )

    rows = list(csv.DictReader(summary_csv.open("r", encoding="utf-8")))
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["protocol"], row["task"], row["setting"])].append(row)

    pareto_payload: dict[str, list[dict]] = {"groups": []}
    for key, group_rows in sorted(grouped.items()):
        protocol, task, setting = key
        metric_key = _metric_name(task)
        reverse = task != "rating_prediction"
        reference = sorted(group_rows, key=lambda row: float(row["metric_value"]), reverse=reverse)[0]
        delta = derive_delta(task, float(reference["metric_value"]))
        if task == "rating_prediction":
            threshold = float(reference["metric_value"]) + delta
            eligible = [row for row in group_rows if float(row["ci_high"]) <= threshold]
            eligible = sorted(eligible, key=lambda row: (int("0" if row["method"] in {"recency_k8", "dense_k8"} else row["method"].split("_b")[1].split("_")[0]), float(row["p50_selection_ms"])))
        else:
            threshold = float(reference["metric_value"]) - delta
            eligible = [row for row in group_rows if float(row["ci_low"]) >= threshold]
            eligible = sorted(eligible, key=lambda row: (int("0" if row["method"] in {"recency_k8", "dense_k8"} else row["method"].split("_b")[1].split("_")[0]), float(row["p50_selection_ms"])))
        pareto = eligible[0] if eligible else None
        pareto_payload["groups"].append(
            {
                "protocol": protocol,
                "task": task,
                "setting": setting,
                "metric_name": metric_key,
                "reference_high_budget": reference,
                "delta": delta,
                "pareto_min_budget": pareto,
            }
        )

    pareto_path = args.version_dir / "reports" / "pareto_budget_report.json"
    write_json(pareto_path, pareto_payload)

    report_md = args.version_dir / "reports" / "task_quality_curve_report.md"
    lines = [
        "# Task Quality Curve Report",
        "",
        f"Summary CSV: `{summary_csv}`",
        f"Pareto JSON: `{pareto_path}`",
        "",
        "This report summarizes the V5.1 fixed-subset Qwen task-quality curves with bootstrap confidence intervals.",
    ]
    report_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"summary_csv": str(summary_csv), "pareto_json": str(pareto_path)}, indent=2))


if __name__ == "__main__":
    main()
