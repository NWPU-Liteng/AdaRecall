#!/usr/bin/env python3
"""Aggregate the immutable structure-aware selector OOF revision."""
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
    parser.add_argument("--config", default="v21_structure_aware_selector_oof_r1.json")
    parser.add_argument("--run-id", default="v21_structure_aware_selector_oof_r1_20260715")
    parser.add_argument("--attempt", default="attempt_003")
    parser.add_argument("--output-revision", default="r1_attempt003")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.run_id != config["run_id"] or config.get("minimum_effect_threshold") is not None:
        raise ValueError("run mismatch or forbidden threshold")
    root = VERSION_ROOT / "artifacts/v21_structure_aware_selector" / args.run_id
    rows, manifests = [], []
    total_queries = 0
    for seed in map(int, config["fixed_seeds"]):
        for fold in range(int(config["outer_folds"])):
            unit = root / f"seed_{seed}" / f"fold_{fold}" / "attempts" / args.attempt
            manifest_path = unit / "v21_structure_aware_selector_manifest.json"
            metrics_path = unit / "v21_structure_aware_selector_metrics.csv"
            manifest = read_json(manifest_path)
            if manifest.get("status") != "completed" or manifest.get("target_fold_excluded") is not True:
                raise ValueError(f"invalid selector unit {manifest_path}")
            if manifest.get("calibration_read") is not False or manifest.get("formal_test_read") is not False or manifest.get("minimum_effect_threshold") is not None:
                raise ValueError(f"locked split or threshold in {manifest_path}")
            if sha256_file(metrics_path) != manifest["hashes"]["metrics"]:
                raise ValueError(f"metric hash mismatch {metrics_path}")
            with metrics_path.open(newline="", encoding="utf-8") as handle:
                unit_rows = list(csv.DictReader(handle))
            unit_queries = int(manifest["outer_heldout_queries"])
            if len(unit_rows) != unit_queries * 3 * 3:
                raise ValueError(f"unit row count mismatch {metrics_path}")
            rows.extend(unit_rows)
            manifests.append(manifest)
            total_queries += unit_queries
    if len(manifests) != 25 or total_queries != 1355 * len(config["fixed_seeds"]) or len(rows) != total_queries * 3 * 3:
        raise ValueError("selector OOF is not complete")
    combined_path = VERSION_ROOT / "aggregate" / f"v21_structure_aware_selector_metrics_{args.output_revision}.csv"
    write_csv_new(combined_path, rows)
    grouped = defaultdict(list)
    for row in rows:
        grouped[(str(row["method"]), int(row["budget"]))].append(row)
    pooled, pooled_map = [], {}
    for key, items in sorted(grouped.items()):
        method, budget = key
        output = {"method": method, "budget": budget, "rows": len(items), **{f"{field}_mean": float(np.mean([float(row[field]) for row in items])) for field in FIELDS}}
        if method == "structure_aware_selector":
            selector_values = [float(row["selector_ms"]) for row in items]
            output["selector_ms_p50"] = float(np.percentile(selector_values, 50))
            output["selector_ms_p95"] = float(np.percentile(selector_values, 95))
        pooled.append(output)
        pooled_map[key] = output
    pooled_path = VERSION_ROOT / "aggregate" / f"v21_structure_aware_selector_pooled_{args.output_revision}.csv"
    write_csv_new(pooled_path, pooled)
    comparisons = []
    for budget in map(int, config["budgets"]):
        selector = pooled_map[("structure_aware_selector", budget)]
        for comparator in ("same_frontier_cosine", "existing_qhds"):
            baseline = pooled_map[(comparator, budget)]
            comparisons.append({"budget": budget, "comparator": comparator, **{f"delta_{field}": selector[f"{field}_mean"] - baseline[f"{field}_mean"] for field in FIELDS if field != "selector_ms"}})
    comparison_path = VERSION_ROOT / "aggregate" / f"v21_structure_aware_selector_comparisons_{args.output_revision}.csv"
    write_csv_new(comparison_path, comparisons)
    summary = {
        "schema_version": "v21-structure-aware-selector-summary-1", "status": "completed_25_of_25_oof", "run_id": args.run_id, "attempt": args.attempt, "units": 25, "rows": len(rows),
        "architecture": "stateful hypergraph marginal selector with separate unary and structural networks",
        "structural_features": config["structural_feature_order"], "pooled": {f"{method}|B={budget}": value for (method, budget), value in pooled_map.items()}, "comparisons": comparisons,
        "training": {"training_examples_mean": float(np.mean([manifest["training_examples"] for manifest in manifests])), "loss_first_mean": float(np.mean([manifest["training_loss_first"] for manifest in manifests])), "loss_last_mean": float(np.mean([manifest["training_loss_last"] for manifest in manifests])), "target_fold_excluded_all_units": True},
        "minimum_effect_threshold": None, "calibration_read": False, "formal_test_read": False,
        "artifacts": {"combined_metrics": str(combined_path.relative_to(VERSION_ROOT)), "pooled": str(pooled_path.relative_to(VERSION_ROOT)), "comparisons": str(comparison_path.relative_to(VERSION_ROOT))},
        "hashes": {"config": sha256_file(VERSION_ROOT / "configs" / args.config), "combined_metrics": sha256_file(combined_path), "pooled": sha256_file(pooled_path), "comparisons": sha256_file(comparison_path)},
    }
    summary_path = VERSION_ROOT / "aggregate" / f"v21_structure_aware_selector_summary_{args.output_revision}.json"
    write_json_new(summary_path, summary)
    b32 = {method: pooled_map[(method, 32)] for method in ("same_frontier_cosine", "existing_qhds", "structure_aware_selector")}
    report = f"""# Method 1 v21 Structure-Aware Selector OOF

All 25 units train a stateful selector on outer-train retained semantic-bundle provenance. The model has separate unary and structural marginal networks; structural state includes bundle completion/progress, prototype coverage novelty, and embedding diversity. Calibration and formal test were not read.

| selector | group recall | member recall | exact task success | payload Tiles | payload bytes | selector ms |
|---|---:|---:|---:|---:|---:|---:|
| same-frontier cosine | {b32['same_frontier_cosine']['group_recall_mean']:.6f} | {b32['same_frontier_cosine']['evidence_member_recall_mean']:.6f} | {b32['same_frontier_cosine']['all_groups_covered_mean']:.6f} | {b32['same_frontier_cosine']['selected_payload_tile_activations_mean']:.3f} | {b32['same_frontier_cosine']['payload_bytes_mean']:.1f} | {b32['same_frontier_cosine']['selector_ms_mean']:.4f} |
| existing QHDS | {b32['existing_qhds']['group_recall_mean']:.6f} | {b32['existing_qhds']['evidence_member_recall_mean']:.6f} | {b32['existing_qhds']['all_groups_covered_mean']:.6f} | {b32['existing_qhds']['selected_payload_tile_activations_mean']:.3f} | {b32['existing_qhds']['payload_bytes_mean']:.1f} | {b32['existing_qhds']['selector_ms_mean']:.4f} |
| structure-aware selector | {b32['structure_aware_selector']['group_recall_mean']:.6f} | {b32['structure_aware_selector']['evidence_member_recall_mean']:.6f} | {b32['structure_aware_selector']['all_groups_covered_mean']:.6f} | {b32['structure_aware_selector']['selected_payload_tile_activations_mean']:.3f} | {b32['structure_aware_selector']['payload_bytes_mean']:.1f} | {b32['structure_aware_selector']['selector_ms_mean']:.4f} |

No minimum-effect threshold is applied. This is an OOF selector revision; it does not read calibration or formal-test data and does not claim final Agent answer quality.
"""
    report_path = VERSION_ROOT / "reports" / f"METHOD1_V21_STRUCTURE_AWARE_SELECTOR_{args.output_revision.upper()}.md"
    with report_path.open("x", encoding="utf-8") as handle:
        handle.write(report)
    print(json.dumps({"status": summary["status"], "rows": len(rows), "summary": str(summary_path), "report": str(report_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
