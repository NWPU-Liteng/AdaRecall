#!/usr/bin/env python3
"""Close Method 1 core KPIs from frozen tuned-r2 OOF outputs.

This is a descriptive post-closure read of train OOF artifacts. It never reads
calibration or formal-test inputs and it does not introduce an effect threshold.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


VERSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VERSION_ROOT / "src"))

from v21.io import iter_jsonl, load_config, read_json, sha256_file, write_csv_new, write_json_new  # noqa: E402


def key_for(row: dict) -> tuple[int, int, str, str, int]:
    return int(row["seed"]), int(row["fold"]), str(row["query_id"]), str(row["method"]), int(row["budget"])


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def exact_dominance(primary: dict, comparator: dict) -> tuple[bool, bool]:
    quality = ("group_recall_at_k_mean", "evidence_member_recall_at_k_mean", "exact_task_success_rate")
    cost = ("query_e2e_ms_mean", "payload_bytes_mean", "logical_tile_activations_mean")

    def dominates(left: dict, right: dict) -> bool:
        weak = all(float(left[field]) >= float(right[field]) for field in quality)
        weak = weak and all(float(left[field]) <= float(right[field]) for field in cost)
        strict = any(float(left[field]) > float(right[field]) for field in quality)
        strict = strict or any(float(left[field]) < float(right[field]) for field in cost)
        return weak and strict

    return dominates(primary, comparator), dominates(comparator, primary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="v21_method1_core_kpi_closure_r1.json")
    parser.add_argument("--output-revision", default="r1_attempt001")
    args = parser.parse_args()
    config = load_config(args.config)
    if config.get("minimum_effect_threshold") is not None or config.get("threshold_based_method_failure") is not False:
        raise ValueError("core KPI closure must not use an effect threshold")
    if config.get("calibration_read") is not False or config.get("formal_test_read") is not False:
        raise ValueError("locked split access is forbidden")

    data = load_config("v21_data.json")
    labels: dict[str, tuple[tuple[str, ...], ...]] = {}
    for row in iter_jsonl(data["split"]["train_query_plan"]):
        query_id = str(row["query_id"])
        groups = tuple(tuple(dict.fromkeys(map(str, group["acceptable_memory_ids"]))) for group in row["evidence_groups"])
        if query_id in labels or not groups or any(not group for group in groups):
            raise ValueError(f"invalid or duplicate evidence groups for {query_id}")
        labels[query_id] = groups
    if len(labels) != 1355:
        raise ValueError("frozen train query count changed")

    source_summary = read_json(config["source_summary"])
    metrics_path = VERSION_ROOT / config["source_metrics"]
    if sha256_file(metrics_path) != source_summary["hashes"]["combined_metrics"]:
        raise ValueError("source combined metrics hash mismatch")
    point = config["deployment_point"]
    budgets = frozenset(map(int, point["budgets"]))
    methods = frozenset([config["primary_method"], *config["comparators"]])

    source_metrics: dict[tuple[int, int, str, str, int], dict[str, float]] = {}
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if str(row["method"]) not in methods or int(row["budget"]) not in budgets:
                continue
            if (int(row["activated_prototypes"]), int(row["members_per_semantic_edge"]), int(row["semantic_bundle_beam"])) != (
                int(point["activated_prototypes"]), int(point["members_per_semantic_edge"]), int(point["semantic_bundle_beam"]),
            ):
                continue
            key = key_for(row)
            if key in source_metrics:
                raise ValueError(f"duplicate source metric key: {key}")
            source_metrics[key] = {
                "group_recall_at_k": float(row["group_coverage"]),
                "exact_task_success": float(row["all_groups_covered"]),
                "query_e2e_ms": float(row["query_e2e_ms"]),
                "payload_bytes": float(row["payload_bytes"]),
                "logical_tile_activations": float(row["total_unique_tile_activations"]),
                "metadata_bytes": float(row["metadata_bytes"]),
                "frontier_candidate_count": float(row["frontier_candidate_count"]),
            }

    records: list[dict] = []
    seeds = list(map(int, load_config("v21_sparse_joint_tuned_oof_r2.json")["fixed_seeds"]))
    folds = int(load_config("v21_sparse_joint_tuned_oof_r2.json")["outer_folds"])
    for seed in seeds:
        for fold in range(folds):
            unit = VERSION_ROOT / "artifacts" / "v21_sparse_joint" / config["source_run_id"] / f"seed_{seed}" / f"fold_{fold}" / "attempts" / config["source_attempt"]
            manifest = read_json(unit / "v21_sparse_joint_unit_manifest.json")
            if manifest.get("calibration_read") is not False or manifest.get("formal_test_read") is not False:
                raise ValueError("source unit touched a locked split")
            selection_path = unit / "v21_sparse_joint_selections.jsonl"
            digest = hashlib.sha256()
            with selection_path.open("rb") as handle:
                for line in handle:
                    digest.update(line)
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if str(row["method"]) not in methods or int(row["budget"]) not in budgets:
                        continue
                    if (int(row["activated_prototypes"]), int(row["members_per_semantic_edge"]), int(row["semantic_bundle_beam"])) != (
                        int(point["activated_prototypes"]), int(point["members_per_semantic_edge"]), int(point["semantic_bundle_beam"]),
                    ):
                        continue
                    key = key_for(row)
                    metric = source_metrics.pop(key, None)
                    if metric is None:
                        raise ValueError(f"selection has no matching source metric: {key}")
                    selected = tuple(map(str, row["selected_memory_ids"]))
                    if len(selected) != int(row["budget"]) or len(set(selected)) != len(selected):
                        raise ValueError(f"invalid top-k selection: {key}")
                    groups = labels[str(row["query_id"])]
                    relevant = frozenset(memory_id for group in groups for memory_id in group)
                    hits = len(set(selected).intersection(relevant))
                    derived_group_recall = float(np.mean([bool(set(selected).intersection(group)) for group in groups]))
                    if not np.isclose(derived_group_recall, metric["group_recall_at_k"], atol=1e-12):
                        raise ValueError(f"group recall does not reproduce source metric: {key}")
                    records.append({
                        "seed": seed, "fold": fold, "query_id": row["query_id"], "method": row["method"], "budget": row["budget"],
                        "selected_count": len(selected), "relevant_member_count": len(relevant), "evidence_group_count": len(groups),
                        "evidence_member_hits": hits,
                        "evidence_member_precision_at_k": hits / len(selected),
                        "evidence_member_recall_at_k": hits / len(relevant),
                        **metric,
                    })
            if digest.hexdigest() != manifest["hashes"]["selections"]:
                raise ValueError(f"selection hash mismatch: seed={seed} fold={fold}")
    if source_metrics:
        raise ValueError(f"unmatched source metrics remain: {len(source_metrics)}")
    expected = len(labels) * len(seeds) * len(budgets) * len(methods)
    if len(records) != expected:
        raise ValueError(f"expected {expected} KPI rows, found {len(records)}")

    rows_path = VERSION_ROOT / "aggregate" / f"v21_method1_core_kpi_query_metrics_{args.output_revision}.csv"
    write_csv_new(rows_path, records)
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in records:
        grouped[(str(row["method"]), int(row["budget"]))].append(row)
    pooled: list[dict] = []
    pooled_by_key: dict[tuple[str, int], dict] = {}
    for (method, budget), items in sorted(grouped.items()):
        output = {
            "method": method,
            "budget": budget,
            "rows": len(items),
            "group_recall_at_k_mean": float(np.mean([row["group_recall_at_k"] for row in items])),
            "evidence_member_recall_at_k_mean": float(np.mean([row["evidence_member_recall_at_k"] for row in items])),
            "evidence_member_precision_at_k_mean": float(np.mean([row["evidence_member_precision_at_k"] for row in items])),
            "exact_task_success_rate": float(np.mean([row["exact_task_success"] for row in items])),
            "query_e2e_ms_mean": float(np.mean([row["query_e2e_ms"] for row in items])),
            "query_e2e_ms_p50": percentile([row["query_e2e_ms"] for row in items], 50),
            "query_e2e_ms_p95": percentile([row["query_e2e_ms"] for row in items], 95),
            "payload_bytes_mean": float(np.mean([row["payload_bytes"] for row in items])),
            "logical_tile_activations_mean": float(np.mean([row["logical_tile_activations"] for row in items])),
            "metadata_bytes_mean": float(np.mean([row["metadata_bytes"] for row in items])),
            "frontier_candidate_count_mean": float(np.mean([row["frontier_candidate_count"] for row in items])),
        }
        pooled.append(output)
        pooled_by_key[(method, budget)] = output
    pooled_path = VERSION_ROOT / "aggregate" / f"v21_method1_core_kpis_{args.output_revision}.csv"
    write_csv_new(pooled_path, pooled)

    comparisons: list[dict] = []
    delta_fields = (
        "group_recall_at_k_mean", "evidence_member_recall_at_k_mean", "evidence_member_precision_at_k_mean",
        "exact_task_success_rate", "query_e2e_ms_mean", "payload_bytes_mean", "logical_tile_activations_mean",
    )
    for budget in sorted(budgets):
        primary = pooled_by_key[(config["primary_method"], budget)]
        for comparator_name in config["comparators"]:
            comparator = pooled_by_key[(comparator_name, budget)]
            primary_dominates, comparator_dominates = exact_dominance(primary, comparator)
            comparisons.append({
                "budget": budget,
                "primary_method": config["primary_method"],
                "comparator": comparator_name,
                **{f"delta_{field}": float(primary[field]) - float(comparator[field]) for field in delta_fields},
                "primary_exactly_pareto_dominates": primary_dominates,
                "comparator_exactly_pareto_dominates": comparator_dominates,
            })
    comparison_path = VERSION_ROOT / "aggregate" / f"v21_method1_core_kpi_comparisons_{args.output_revision}.csv"
    write_csv_new(comparison_path, comparisons)

    summary = {
        "schema_version": "v21-method1-core-kpi-closure-1",
        "status": "completed_descriptive_oof_no_effect_threshold",
        "revision_id": config["revision_id"],
        "source_run_id": config["source_run_id"],
        "source_attempt": config["source_attempt"],
        "deployment_point": point,
        "query_metric_rows": len(records),
        "pooled": {f"{row['method']}|B={row['budget']}": row for row in pooled},
        "comparisons": comparisons,
        "metric_semantics": config["core_quality_metrics"],
        "agent_handoff": config["agent_handoff"],
        "decision_policy": config["decision_policy"],
        "minimum_effect_threshold": None,
        "threshold_based_method_failure": False,
        "dynamic_update_evaluated": False,
        "calibration_read": False,
        "formal_test_read": False,
        "artifacts": {
            "query_metrics": str(rows_path.relative_to(VERSION_ROOT)),
            "pooled_kpis": str(pooled_path.relative_to(VERSION_ROOT)),
            "comparisons": str(comparison_path.relative_to(VERSION_ROOT)),
        },
        "hashes": {
            "config": sha256_file(VERSION_ROOT / "configs" / args.config),
            "source_summary": sha256_file(VERSION_ROOT / config["source_summary"]),
            "source_metrics": sha256_file(metrics_path),
            "query_metrics": sha256_file(rows_path),
            "pooled_kpis": sha256_file(pooled_path),
            "comparisons": sha256_file(comparison_path),
        },
    }
    summary_path = VERSION_ROOT / "aggregate" / f"v21_method1_core_kpi_closure_{args.output_revision}.json"
    write_json_new(summary_path, summary)

    budget = max(budgets)
    report_path = VERSION_ROOT / "reports" / f"METHOD1_V21_CORE_KPI_CLOSURE_{args.output_revision.upper()}.md"
    primary = pooled_by_key[(config["primary_method"], budget)]
    lines = [
        "# Method 1 v21 Core KPI Closure",
        "",
        "This is a descriptive train-OOF closure over the immutable tuned-r2 outputs. No calibration or formal-test input was read, and no minimum-effect threshold is used.",
        "",
        "## Metric semantics",
        "",
        "- `group recall@K` is the fraction of required evidence groups hit by at least one acceptable memory. It is the task-aligned retrieval recall and reproduces the prior `group_coverage` metric.",
        "- `evidence-member recall@K` is conventional set recall over the union of all acceptable memories. It penalizes failure to retrieve alternatives that are not simultaneously required, so it must not replace group recall.",
        "- `exact task success` is the fraction of queries covering every evidence group. It is a retrieval-set accuracy proxy, not final Agent answer accuracy.",
        "- `logical Tiles` are host-side logical layout counts, not PIM hardware measurements.",
        "",
        f"## Fixed deployment point: 16 activated prototypes, 32 members, beam 32, B={budget}",
        "",
        "| method | group recall | member recall | member precision | exact success | e2e mean | e2e p95 | payload bytes | logical Tiles |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in [config["primary_method"], *config["comparators"]]:
        row = pooled_by_key[(method, budget)]
        lines.append(
            f"| {method} | {row['group_recall_at_k_mean']:.6f} | {row['evidence_member_recall_at_k_mean']:.6f} | "
            f"{row['evidence_member_precision_at_k_mean']:.6f} | {row['exact_task_success_rate']:.6f} | "
            f"{row['query_e2e_ms_mean']:.3f} ms | {row['query_e2e_ms_p95']:.3f} ms | "
            f"{row['payload_bytes_mean']:.1f} | {row['logical_tile_activations_mean']:.3f} |"
        )
    lines.extend([
        "",
        "## Closure interpretation",
        "",
        "The output reports raw paired-method deltas and exact Pareto dominance only. A small positive, zero, or negative effect remains a measured result; it is not converted into a method-valid/method-invalid gate.",
        "",
        f"The retrieval handoff is the selected B-memory set. `rerank_executed=false`, so no downstream reranker or final Agent answer-accuracy result is claimed. The primary B={budget} values are group recall "
        f"`{primary['group_recall_at_k_mean']:.6f}`, exact task success `{primary['exact_task_success_rate']:.6f}`, mean e2e `{primary['query_e2e_ms_mean']:.3f} ms`, and logical Tiles `{primary['logical_tile_activations_mean']:.3f}`.",
        "",
        "Dynamic content-update mechanics are implemented separately as the audited static overlay fallback. They do not turn this OOF result into a causal streaming evaluation.",
    ])
    with report_path.open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(json.dumps({"status": summary["status"], "rows": len(records), "summary": str(summary_path), "report": str(report_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
