#!/usr/bin/env python3
"""Measure candidate-frontier versus selector quality headroom on train OOF."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


VERSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VERSION_ROOT / "src"))

from v21.io import iter_jsonl, load_config, sha256_file, write_csv_new, write_json_new  # noqa: E402


def group_recall(selected: set[str], groups: tuple[frozenset[str], ...]) -> tuple[float, float]:
    covered = [bool(selected.intersection(group)) for group in groups]
    return float(np.mean(covered)), float(all(covered))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-revision", default="r1_attempt001")
    args = parser.parse_args()
    data = load_config("v21_data.json")
    sparse = load_config("v21_sparse_joint_tuned_oof_r2.json")
    labels = {
        str(row["query_id"]): tuple(frozenset(map(str, group["acceptable_memory_ids"])) for group in row["evidence_groups"])
        for row in iter_jsonl(data["split"]["train_query_plan"])
    }
    selected_metrics_path = VERSION_ROOT / "aggregate/v21_method1_core_kpi_query_metrics_r1_attempt001.csv"
    selected: dict[tuple[int, str], tuple[float, float]] = {}
    with selected_metrics_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if str(row["method"]) == "sparse_joint_semantic_bundle_hypergraph" and int(row["budget"]) == 32:
                selected[(int(row["seed"]), str(row["query_id"]))] = (float(row["group_recall_at_k"]), float(row["exact_task_success"]))
    rows = []
    for seed in map(int, sparse["fixed_seeds"]):
        for fold in range(int(sparse["outer_folds"])):
            path = VERSION_ROOT / "artifacts/v21_fixed_sparse_frontiers/v21_fixed_sparse_frontiers_r1_20260715" / f"seed_{seed}" / f"fold_{fold}" / "attempts/attempt_001/v21_fixed_sparse_frontiers.jsonl"
            for row in iter_jsonl(path):
                query_id = str(row["query_id"])
                frontier = set(map(str, row["frontier_memory_ids"]))
                protected = set(map(str, row["protected_ann_memory_ids"]))
                frontier_recall, frontier_all = group_recall(frontier, labels[query_id])
                protected_recall, protected_all = group_recall(protected, labels[query_id])
                selected_recall, selected_all = selected[(seed, query_id)]
                relevant = set().union(*labels[query_id])
                rows.append({
                    "seed": seed, "fold": fold, "query_id": query_id,
                    "protected_group_recall": protected_recall,
                    "selected_B32_group_recall": selected_recall,
                    "frontier96_group_recall_upper_bound": frontier_recall,
                    "protected_all_groups": protected_all,
                    "selected_B32_all_groups": selected_all,
                    "frontier96_all_groups_upper_bound": frontier_all,
                    "selector_group_recall_headroom_inside_frontier": frontier_recall - selected_recall,
                    "candidate_group_recall_gap_to_perfect": 1.0 - frontier_recall,
                    "frontier96_member_recall": len(frontier.intersection(relevant)) / max(len(relevant), 1),
                })
    if len(rows) != 1355 * len(sparse["fixed_seeds"]):
        raise ValueError("frontier headroom rows are incomplete")
    metrics_path = VERSION_ROOT / "aggregate" / f"v21_sparse_frontier_quality_headroom_metrics_{args.output_revision}.csv"
    write_csv_new(metrics_path, rows)
    fields = [field for field in rows[0] if field not in {"seed", "fold", "query_id"}]
    pooled = {f"{field}_mean": float(np.mean([row[field] for row in rows])) for field in fields}
    summary = {
        "schema_version": "v21-sparse-frontier-quality-headroom-1",
        "status": "completed_gold_diagnostic_not_deployable",
        "rows": len(rows),
        "pooled": pooled,
        "interpretation": "frontier96 upper bound uses heldout evidence labels for diagnosis only; it is never an inference input",
        "calibration_read": False,
        "formal_test_read": False,
        "artifacts": {"metrics": str(metrics_path.relative_to(VERSION_ROOT))},
        "hashes": {"selected_metrics": sha256_file(selected_metrics_path), "metrics": sha256_file(metrics_path)},
    }
    summary_path = VERSION_ROOT / "aggregate" / f"v21_sparse_frontier_quality_headroom_{args.output_revision}.json"
    write_json_new(summary_path, summary)
    print(json.dumps({"status": summary["status"], "pooled": pooled}, sort_keys=True))


if __name__ == "__main__":
    main()
