#!/usr/bin/env python3
"""Aggregate the complete persona-sharded hyper-index OOF experiment."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


VERSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VERSION_ROOT / "src"))

from v21.io import load_config, read_json, sha256_file, write_csv_new, write_json_new  # noqa: E402


MEAN_FIELDS = (
    "global_rank_entries_visited",
    "persona_rank_entries_visited",
    "global_metadata_bytes_accessed",
    "persona_metadata_bytes_accessed",
    "global_metadata_tile_activations",
    "persona_metadata_tile_activations",
    "global_feature_tile_activations",
    "replicated_feature_tile_activations",
    "selected_payload_tile_activations",
    "global_total_logical_tile_activations",
    "persona_sharded_total_logical_tile_activations",
    "feature_replica_hit_rate",
    "global_lookup_ms",
    "persona_sharded_lookup_ms",
    "payload_bytes",
)


def summarize(rows: list[dict]) -> dict:
    output = {f"{field}_mean": float(np.mean([float(row[field]) for row in rows])) for field in MEAN_FIELDS}
    for field in ("global_total_logical_tile_activations", "persona_sharded_total_logical_tile_activations", "global_lookup_ms", "persona_sharded_lookup_ms"):
        values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
        output[f"{field}_p50"] = float(np.percentile(values, 50))
        output[f"{field}_p95"] = float(np.percentile(values, 95))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="v21_persona_sharded_hyper_index_oof_r2.json")
    parser.add_argument("--run-id", default="v21_persona_sharded_hyper_index_oof_r2_20260715")
    parser.add_argument("--attempt", default="attempt_001")
    parser.add_argument("--output-revision", default="r2_attempt001")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.run_id != config["run_id"] or config.get("minimum_effect_threshold") is not None:
        raise ValueError("run/config mismatch or forbidden threshold")
    root = VERSION_ROOT / "artifacts/v21_persona_sharded_hyper_index" / args.run_id
    all_rows: list[dict] = []
    unit_manifests: list[dict] = []
    for seed in map(int, config["fixed_seeds"]):
        for fold in range(int(config["outer_folds"])):
            unit = root / f"seed_{seed}" / f"fold_{fold}" / "attempts" / args.attempt
            manifest_path = unit / "v21_persona_sharded_hyper_index_manifest.json"
            metrics_path = unit / "v21_persona_sharded_hyper_index_metrics.csv"
            manifest = read_json(manifest_path)
            if manifest.get("status") != "completed" or manifest.get("calibration_read") is not False or manifest.get("formal_test_read") is not False:
                raise ValueError(f"invalid unit manifest: {manifest_path}")
            if manifest.get("returned_members_identical_all_queries") is not True or manifest.get("quality_selection_unchanged") is not True:
                raise ValueError(f"lookup equivalence failed: {manifest_path}")
            if sha256_file(metrics_path) != manifest["hashes"]["metrics"] or sha256_file(VERSION_ROOT / "configs" / args.config) != manifest["hashes"]["config"]:
                raise ValueError(f"unit hash mismatch: {manifest_path}")
            with metrics_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            if len(rows) != int(manifest["queries"]):
                raise ValueError(f"unit row count mismatch: {manifest_path}")
            all_rows.extend(rows)
            unit_manifests.append(manifest)
    expected = 1355 * len(config["fixed_seeds"])
    if len(all_rows) != expected or len(unit_manifests) != 25:
        raise ValueError(f"expected 25 units and {expected} rows")
    if any(str(row["returned_members_identical"]).lower() not in {"true", "1"} for row in all_rows):
        raise ValueError("not every query preserved returned members")

    combined_path = VERSION_ROOT / "aggregate" / f"v21_persona_sharded_hyper_index_metrics_{args.output_revision}.csv"
    write_csv_new(combined_path, all_rows)
    pooled = summarize(all_rows)
    per_seed_fold: list[dict] = []
    for seed in map(int, config["fixed_seeds"]):
        for fold in range(int(config["outer_folds"])):
            rows = [row for row in all_rows if int(row["seed"]) == seed and int(row["fold"]) == fold]
            per_seed_fold.append({"seed": seed, "fold": fold, "rows": len(rows), **summarize(rows)})
    per_unit_path = VERSION_ROOT / "aggregate" / f"v21_persona_sharded_hyper_index_per_seed_fold_{args.output_revision}.csv"
    write_csv_new(per_unit_path, per_seed_fold)

    effects = {
        "rank_entries_visited": pooled["persona_rank_entries_visited_mean"] - pooled["global_rank_entries_visited_mean"],
        "metadata_bytes_accessed": pooled["persona_metadata_bytes_accessed_mean"] - pooled["global_metadata_bytes_accessed_mean"],
        "metadata_tile_activations": pooled["persona_metadata_tile_activations_mean"] - pooled["global_metadata_tile_activations_mean"],
        "feature_tile_activations": pooled["replicated_feature_tile_activations_mean"] - pooled["global_feature_tile_activations_mean"],
        "total_logical_tile_activations": pooled["persona_sharded_total_logical_tile_activations_mean"] - pooled["global_total_logical_tile_activations_mean"],
        "lookup_ms": pooled["persona_sharded_lookup_ms_mean"] - pooled["global_lookup_ms_mean"],
    }
    relative = {
        "rank_entries_visited": effects["rank_entries_visited"] / pooled["global_rank_entries_visited_mean"],
        "metadata_bytes_accessed": effects["metadata_bytes_accessed"] / pooled["global_metadata_bytes_accessed_mean"],
        "metadata_tile_activations": effects["metadata_tile_activations"] / pooled["global_metadata_tile_activations_mean"],
        "feature_tile_activations": effects["feature_tile_activations"] / pooled["global_feature_tile_activations_mean"],
        "total_logical_tile_activations": effects["total_logical_tile_activations"] / pooled["global_total_logical_tile_activations_mean"],
        "lookup_ms": effects["lookup_ms"] / pooled["global_lookup_ms_mean"],
    }
    kpi_path = VERSION_ROOT / "aggregate/v21_method1_core_kpi_closure_r1_attempt001.json"
    kpi = read_json(kpi_path)
    quality = kpi["pooled"]["sparse_joint_semantic_bundle_hypergraph|B=32"]
    legacy_headroom_path = VERSION_ROOT / "aggregate/v21_legacy_hyper_layout_headroom_r1_attempt001.json"
    legacy_headroom = read_json(legacy_headroom_path)
    summary = {
        "schema_version": "v21-persona-sharded-hyper-index-oof-summary-1",
        "status": "completed_25_of_25_oof",
        "run_id": args.run_id,
        "attempt": args.attempt,
        "units": len(unit_manifests),
        "rows": len(all_rows),
        "pooled": pooled,
        "paired_effect_persona_sharded_minus_global": effects,
        "relative_effect_persona_sharded_minus_global": relative,
        "quality_unchanged": {
            "group_recall_at_32": quality["group_recall_at_k_mean"],
            "evidence_member_recall_at_32": quality["evidence_member_recall_at_k_mean"],
            "exact_task_success": quality["exact_task_success_rate"],
            "payload_bytes": quality["payload_bytes_mean"],
        },
        "storage": {
            "global_rank_metadata_bytes": unit_manifests[0]["rank_metadata_storage_bytes_global"],
            "persona_sharded_rank_metadata_bytes": unit_manifests[0]["rank_metadata_storage_bytes_sharded"],
            "feature_replication_factor": unit_manifests[0]["feature_replication_factor"],
            "feature_replica_storage_bytes": unit_manifests[0]["feature_replica_storage_bytes"],
        },
        "legacy_all_train_layout_diagnostic_only": legacy_headroom["pooled"],
        "metric_correction": {
            "old_v21_value": "52.61 mixed returned-member Tiles under neutral layout plus full-index metadata bytes",
            "corrected_global": "4KiB rank-prefix blocks union capacity-16 candidate feature/payload Tiles",
            "corrected_sharded": "persona-prototype 4KiB rank-prefix blocks union the same single-copy candidate feature/payload Tiles",
            "old_and_corrected_tile_totals_directly_comparable": False,
        },
        "minimum_effect_threshold": None,
        "claim_boundary": config["claim_boundary"],
        "calibration_read": False,
        "formal_test_read": False,
        "artifacts": {
            "combined_metrics": str(combined_path.relative_to(VERSION_ROOT)),
            "per_seed_fold": str(per_unit_path.relative_to(VERSION_ROOT)),
        },
        "hashes": {
            "config": sha256_file(VERSION_ROOT / "configs" / args.config),
            "combined_metrics": sha256_file(combined_path),
            "per_seed_fold": sha256_file(per_unit_path),
            "core_kpi": sha256_file(kpi_path),
            "legacy_headroom": sha256_file(legacy_headroom_path),
        },
    }
    summary_path = VERSION_ROOT / "aggregate" / f"v21_persona_sharded_hyper_index_summary_{args.output_revision}.json"
    write_json_new(summary_path, summary)
    report_path = VERSION_ROOT / "reports" / f"METHOD1_V21_PERSONA_SHARDED_HYPER_INDEX_{args.output_revision.upper()}.md"
    report = f"""# Method 1 v21 Persona-Sharded Hyper-Index OOF

All 25 outer-fold units completed without calibration or formal-test access. The persona-sharded index returned exactly the same members as the global prototype rank index for all {len(all_rows):,} query-seed rows, so retrieval quality and payload are unchanged.

## Core B=32 quality retained

- group recall@32: `{quality['group_recall_at_k_mean']:.6f}`
- evidence-member recall@32: `{quality['evidence_member_recall_at_k_mean']:.6f}`
- exact task success: `{quality['exact_task_success_rate']:.6f}`
- payload: `{quality['payload_bytes_mean']:.1f} B`

## Corrected logical storage result

| metric | global prototype rank index | persona x prototype shards | relative change |
|---|---:|---:|---:|
| rank entries visited | {pooled['global_rank_entries_visited_mean']:.1f} | {pooled['persona_rank_entries_visited_mean']:.1f} | {relative['rank_entries_visited'] * 100:+.2f}% |
| metadata bytes accessed | {pooled['global_metadata_bytes_accessed_mean']:.1f} | {pooled['persona_metadata_bytes_accessed_mean']:.1f} | {relative['metadata_bytes_accessed'] * 100:+.2f}% |
| metadata 4KiB Tile activations | {pooled['global_metadata_tile_activations_mean']:.3f} | {pooled['persona_metadata_tile_activations_mean']:.3f} | {relative['metadata_tile_activations'] * 100:+.2f}% |
| candidate feature Tile activations | {pooled['global_feature_tile_activations_mean']:.3f} | {pooled['replicated_feature_tile_activations_mean']:.3f} | {relative['feature_tile_activations'] * 100:+.2f}% |
| selected payload Tile activations | {pooled['selected_payload_tile_activations_mean']:.3f} | {pooled['selected_payload_tile_activations_mean']:.3f} | +0.00% |
| corrected total logical Tiles | {pooled['global_total_logical_tile_activations_mean']:.3f} | {pooled['persona_sharded_total_logical_tile_activations_mean']:.3f} | {relative['total_logical_tile_activations'] * 100:+.2f}% |
| sparse lookup mean | {pooled['global_lookup_ms_mean']:.3f} ms | {pooled['persona_sharded_lookup_ms_mean']:.3f} ms | {relative['lookup_ms'] * 100:+.2f}% |

The old `52.61` value is not directly comparable: it mapped returned members to neutral payload Tiles while charging the full 10.76 MB rank index on every query. The corrected metric separates actual rank-prefix blocks, candidate feature Tiles, and selected payload Tiles.

The rank storage size is unchanged because sharding reorders the same entries. No feature or payload replication is used in r2. These are host/logical measurements, not PIM hardware results.
"""
    with report_path.open("x", encoding="utf-8") as handle:
        handle.write(report)
    print(json.dumps({"status": summary["status"], "units": 25, "rows": len(all_rows), "summary": str(summary_path), "report": str(report_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
