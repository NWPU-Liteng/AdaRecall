#!/usr/bin/env python3
"""Aggregate the complete sparse-frontier selector refit OOF experiment."""
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

from v21.io import load_config, read_json, sha256_file, write_csv_new, write_json_new  # noqa: E402


FIELDS = ("group_recall", "all_groups_covered", "evidence_member_recall", "evidence_member_precision", "selected_payload_tile_activations", "payload_bytes", "selector_ms")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="v21_sparse_frontier_selector_refit_oof_r1.json")
    parser.add_argument("--run-id", default="v21_sparse_frontier_selector_refit_oof_r1_20260715")
    parser.add_argument("--attempt", default="attempt_001")
    parser.add_argument("--output-revision", default="r1_attempt001")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.run_id != config["run_id"] or config.get("minimum_effect_threshold") is not None:
        raise ValueError("run mismatch or forbidden threshold")
    root = VERSION_ROOT / "artifacts/v21_sparse_frontier_selector_refit" / args.run_id
    rows: list[dict] = []
    manifests: list[dict] = []
    for seed in map(int, config["fixed_seeds"]):
        for fold in range(int(config["outer_folds"])):
            unit = root / f"seed_{seed}" / f"fold_{fold}" / "attempts" / args.attempt
            manifest_path = unit / "v21_sparse_frontier_selector_refit_manifest.json"
            metrics_path = unit / "v21_sparse_frontier_selector_refit_metrics.csv"
            manifest = read_json(manifest_path)
            if manifest.get("status") != "completed" or manifest.get("target_fold_excluded") is not True:
                raise ValueError(f"invalid unit {manifest_path}")
            if manifest.get("calibration_read") is not False or manifest.get("formal_test_read") is not False:
                raise ValueError(f"locked split access in {manifest_path}")
            if sha256_file(metrics_path) != manifest["hashes"]["metrics"]:
                raise ValueError(f"metric hash mismatch in {manifest_path}")
            with metrics_path.open(newline="", encoding="utf-8") as handle:
                unit_rows = list(csv.DictReader(handle))
            rows.extend(unit_rows)
            manifests.append(manifest)
    expected = 1355 * len(config["fixed_seeds"]) * 3 * 3
    if len(rows) != expected or len(manifests) != 25:
        raise ValueError(f"expected 25 units and {expected} metric rows")
    combined_path = VERSION_ROOT / "aggregate" / f"v21_sparse_frontier_selector_refit_metrics_{args.output_revision}.csv"
    write_csv_new(combined_path, rows)
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["method"]), int(row["budget"]))].append(row)
    pooled: list[dict] = []
    pooled_map: dict[tuple[str, int], dict] = {}
    for (method, budget), items in sorted(grouped.items()):
        output = {"method": method, "budget": budget, "rows": len(items), **{f"{field}_mean": float(np.mean([float(row[field]) for row in items])) for field in FIELDS}}
        if method == "sparse_frontier_refit_mlp":
            values = [float(row["selector_ms"]) for row in items]
            output["selector_ms_p50"] = float(np.percentile(values, 50))
            output["selector_ms_p95"] = float(np.percentile(values, 95))
        pooled.append(output)
        pooled_map[(method, budget)] = output
    pooled_path = VERSION_ROOT / "aggregate" / f"v21_sparse_frontier_selector_refit_pooled_{args.output_revision}.csv"
    write_csv_new(pooled_path, pooled)
    comparisons = []
    for budget in map(int, config["budgets"]):
        refit = pooled_map[("sparse_frontier_refit_mlp", budget)]
        for comparator in ("ann_flat", "existing_qhds"):
            baseline = pooled_map[(comparator, budget)]
            comparisons.append({
                "budget": budget,
                "comparator": comparator,
                **{f"delta_{field}": refit[f"{field}_mean"] - baseline[f"{field}_mean"] for field in FIELDS if field != "selector_ms"},
            })
    comparison_path = VERSION_ROOT / "aggregate" / f"v21_sparse_frontier_selector_refit_comparisons_{args.output_revision}.csv"
    write_csv_new(comparison_path, comparisons)
    tile_summary_path = VERSION_ROOT / "aggregate/v21_persona_sharded_hyper_index_summary_r2_attempt001.json"
    tile_summary = read_json(tile_summary_path)
    summary = {
        "schema_version": "v21-sparse-frontier-selector-refit-summary-1",
        "status": "completed_25_of_25_oof",
        "run_id": args.run_id,
        "attempt": args.attempt,
        "units": 25,
        "rows": len(rows),
        "pooled": {f"{row['method']}|B={row['budget']}": row for row in pooled},
        "comparisons": comparisons,
        "training": {
            "loss_first_mean": float(np.mean([manifest["training_loss_first"] for manifest in manifests])),
            "loss_last_mean": float(np.mean([manifest["training_loss_last"] for manifest in manifests])),
            "actual_sparse_frontier_training": True,
            "target_fold_excluded_all_units": True,
        },
        "persona_sharded_tile_result": {
            "corrected_total_logical_tiles": tile_summary["pooled"]["persona_sharded_total_logical_tile_activations_mean"],
            "relative_reduction_vs_global": tile_summary["relative_effect_persona_sharded_minus_global"]["total_logical_tile_activations"],
            "lookup_ms": tile_summary["pooled"]["persona_sharded_lookup_ms_mean"],
        },
        "minimum_effect_threshold": None,
        "calibration_read": False,
        "formal_test_read": False,
        "artifacts": {"combined_metrics": str(combined_path.relative_to(VERSION_ROOT)), "pooled": str(pooled_path.relative_to(VERSION_ROOT)), "comparisons": str(comparison_path.relative_to(VERSION_ROOT))},
        "hashes": {"config": sha256_file(VERSION_ROOT / "configs" / args.config), "combined_metrics": sha256_file(combined_path), "pooled": sha256_file(pooled_path), "comparisons": sha256_file(comparison_path), "tile_summary": sha256_file(tile_summary_path)},
    }
    summary_path = VERSION_ROOT / "aggregate" / f"v21_sparse_frontier_selector_refit_summary_{args.output_revision}.json"
    write_json_new(summary_path, summary)
    report_path = VERSION_ROOT / "reports" / f"METHOD1_V21_SPARSE_FRONTIER_SELECTOR_REFIT_{args.output_revision.upper()}.md"
    b32 = {method: pooled_map[(method, 32)] for method in ("ann_flat", "existing_qhds", "sparse_frontier_refit_mlp")}
    report = f"""# Method 1 v21 Sparse-Frontier Selector Refit OOF

All 25 OOF units train the fixed candidate utility MLP on actual cap-96 outer-train sparse frontiers. Target folds, calibration, and formal test are excluded.

## B=32

| selector | group recall | member recall | exact task success | payload Tiles | payload bytes |
|---|---:|---:|---:|---:|---:|
| same-frontier cosine | {b32['ann_flat']['group_recall_mean']:.6f} | {b32['ann_flat']['evidence_member_recall_mean']:.6f} | {b32['ann_flat']['all_groups_covered_mean']:.6f} | {b32['ann_flat']['selected_payload_tile_activations_mean']:.3f} | {b32['ann_flat']['payload_bytes_mean']:.1f} |
| existing QHDS | {b32['existing_qhds']['group_recall_mean']:.6f} | {b32['existing_qhds']['evidence_member_recall_mean']:.6f} | {b32['existing_qhds']['all_groups_covered_mean']:.6f} | {b32['existing_qhds']['selected_payload_tile_activations_mean']:.3f} | {b32['existing_qhds']['payload_bytes_mean']:.1f} |
| sparse-frontier refit MLP | {b32['sparse_frontier_refit_mlp']['group_recall_mean']:.6f} | {b32['sparse_frontier_refit_mlp']['evidence_member_recall_mean']:.6f} | {b32['sparse_frontier_refit_mlp']['all_groups_covered_mean']:.6f} | {b32['sparse_frontier_refit_mlp']['selected_payload_tile_activations_mean']:.3f} | {b32['sparse_frontier_refit_mlp']['payload_bytes_mean']:.1f} |

No minimum-effect threshold is applied. The result is retained whether the refit effect is positive, zero, or negative.
"""
    with report_path.open("x", encoding="utf-8") as handle:
        handle.write(report)
    print(json.dumps({"status": summary["status"], "units": 25, "summary": str(summary_path), "report": str(report_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
