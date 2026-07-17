#!/usr/bin/env python3
"""Compare matched single-grid global and persona-sharded e2e controls."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


VERSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VERSION_ROOT / "src"))

from v21.io import read_json, sha256_file, write_csv_new, write_json_new  # noqa: E402


STAGES = (
    "ann_seed_search_ms", "latent_aspect_predict_ms", "semantic_edge_lookup_ms",
    "evidence_bundle_build_ms", "bundle_hyperedge_score_ms", "submodular_select_ms",
    "candidate_union_ms", "selection_total_ms", "query_e2e_ms",
)
GLOBAL_RUN = "v21_sparse_joint_global_e2e_control_oof_r1_20260715"
SHARDED_RUN = "v21_sparse_joint_sharded_e2e_oof_r1_20260715"
ACTIVE_PROTOTYPES = 16
MEMBERS_PER_EDGE = 32
BUNDLE_BEAM = 32


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", default="attempt_001")
    parser.add_argument("--output-revision", default="r1_attempt001")
    args = parser.parse_args()
    records = []
    manifests = []
    for seed in range(20260712, 20260717):
        for fold in range(5):
            unit_rows = {}
            selection_hashes = {}
            for label, run_id in (("global", GLOBAL_RUN), ("persona_sharded", SHARDED_RUN)):
                unit = VERSION_ROOT / "artifacts/v21_sparse_joint" / run_id / f"seed_{seed}" / f"fold_{fold}" / "attempts" / args.attempt
                manifest_path = unit / "v21_sparse_joint_unit_manifest.json"
                metrics_path = unit / "v21_sparse_joint_query_metrics.csv"
                selections_path = unit / "v21_sparse_joint_selections.jsonl"
                manifest = read_json(manifest_path)
                if manifest.get("status") != "completed" or manifest.get("calibration_read") is not False or manifest.get("formal_test_read") is not False:
                    raise ValueError(f"invalid unit {manifest_path}")
                if sha256_file(metrics_path) != manifest["hashes"]["metrics"] or sha256_file(selections_path) != manifest["hashes"]["selections"]:
                    raise ValueError(f"unit hash mismatch {manifest_path}")
                with metrics_path.open(newline="", encoding="utf-8") as handle:
                    rows = [
                        row for row in csv.DictReader(handle)
                        if str(row["method"]) == "sparse_joint_semantic_bundle_hypergraph"
                        and int(row["activated_prototypes"]) == ACTIVE_PROTOTYPES
                        and int(row["members_per_semantic_edge"]) == MEMBERS_PER_EDGE
                        and int(row["semantic_bundle_beam"]) == BUNDLE_BEAM
                        and int(row["budget"]) == 32
                    ]
                unit_rows[label] = {str(row["query_id"]): row for row in rows}
                selection_hashes[label] = sha256_file(selections_path)
                manifests.append(manifest)
            if selection_hashes["global"] != selection_hashes["persona_sharded"]:
                raise ValueError(f"global/sharded selections differ seed={seed} fold={fold}")
            if set(unit_rows["global"]) != set(unit_rows["persona_sharded"]):
                raise ValueError(f"global/sharded query sets differ seed={seed} fold={fold}")
            for query_id in sorted(unit_rows["global"]):
                global_row = unit_rows["global"][query_id]
                shard_row = unit_rows["persona_sharded"][query_id]
                if global_row["selected_memory_ids"] if "selected_memory_ids" in global_row else False:
                    raise RuntimeError("metrics unexpectedly contain selection payload")
                records.append({
                    "seed": seed, "fold": fold, "query_id": query_id,
                    **{f"global_{field}": float(global_row[field]) for field in STAGES},
                    **{f"persona_sharded_{field}": float(shard_row[field]) for field in STAGES},
                    **{f"delta_{field}": float(shard_row[field]) - float(global_row[field]) for field in STAGES},
                    "group_coverage": float(global_row["group_coverage"]),
                    "all_groups_covered": float(global_row["all_groups_covered"]),
                })
    if len(records) != 6775 or len(manifests) != 50:
        raise ValueError("matched e2e controls are incomplete")
    metrics_path = VERSION_ROOT / "aggregate" / f"v21_sparse_joint_e2e_control_paired_metrics_{args.output_revision}.csv"
    write_csv_new(metrics_path, records)
    pooled = {}
    for field in STAGES:
        global_values = np.asarray([row[f"global_{field}"] for row in records], dtype=np.float64)
        shard_values = np.asarray([row[f"persona_sharded_{field}"] for row in records], dtype=np.float64)
        delta = shard_values - global_values
        pooled[field] = {
            "global_mean": float(global_values.mean()),
            "persona_sharded_mean": float(shard_values.mean()),
            "delta_mean": float(delta.mean()),
            "delta_p50": float(np.percentile(delta, 50)),
            "delta_p95": float(np.percentile(delta, 95)),
        }
    tile_path = VERSION_ROOT / "aggregate/v21_persona_sharded_hyper_index_summary_r2_attempt001.json"
    tile = read_json(tile_path)
    summary = {
        "schema_version": "v21-sparse-joint-e2e-control-comparison-1",
        "status": "completed_25_of_25_matched_single_grid_controls",
        "rows": len(records),
        "selections_identical_all_units": True,
        "quality_unchanged": {"group_coverage": float(np.mean([row["group_coverage"] for row in records])), "all_groups_covered": float(np.mean([row["all_groups_covered"] for row in records]))},
        "stage_latency": pooled,
        "corrected_tile_result": {
            "global_total_logical_tiles": tile["pooled"]["global_total_logical_tile_activations_mean"],
            "persona_sharded_total_logical_tiles": tile["pooled"]["persona_sharded_total_logical_tile_activations_mean"],
            "relative_effect": tile["relative_effect_persona_sharded_minus_global"]["total_logical_tile_activations"],
        },
        "minimum_effect_threshold": None,
        "calibration_read": False,
        "formal_test_read": False,
        "artifacts": {"paired_metrics": str(metrics_path.relative_to(VERSION_ROOT))},
        "hashes": {"paired_metrics": sha256_file(metrics_path), "tile_summary": sha256_file(tile_path)},
    }
    summary_path = VERSION_ROOT / "aggregate" / f"v21_sparse_joint_e2e_control_summary_{args.output_revision}.json"
    write_json_new(summary_path, summary)
    report_path = VERSION_ROOT / "reports" / f"METHOD1_V21_SPARSE_JOINT_E2E_CONTROL_{args.output_revision.upper()}.md"
    report = f"""# Method 1 v21 Matched Single-Grid E2E Control

Global and persona-sharded rank indices were evaluated with the same single grid, reused selector weights, and identical selections in all 25 OOF units. Calibration and formal test were not read.

| stage | global mean | persona-sharded mean | paired delta |
|---|---:|---:|---:|
""" + "\n".join(
        f"| {field} | {values['global_mean']:.3f} ms | {values['persona_sharded_mean']:.3f} ms | {values['delta_mean']:+.3f} ms |"
        for field, values in pooled.items()
    ) + f"""

Corrected logical Tiles change from `{summary['corrected_tile_result']['global_total_logical_tiles']:.3f}` to `{summary['corrected_tile_result']['persona_sharded_total_logical_tiles']:.3f}` (`{summary['corrected_tile_result']['relative_effect'] * 100:+.2f}%`). No minimum-effect threshold is applied.
"""
    with report_path.open("x", encoding="utf-8") as handle:
        handle.write(report)
    print(json.dumps({"status": summary["status"], "rows": len(records), "summary": str(summary_path), "report": str(report_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
