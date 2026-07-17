#!/usr/bin/env python3
"""Diagnostic-only Tile headroom from a prior all-train hyper layout.

The legacy layout is not OOF-safe for v21 and is never a deployable result.
It is read only to determine whether its layout family can organize the fixed
v21 sparse frontier better than the neutral source-order layout.
"""
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


LEGACY_LAYOUT = VERSION_ROOT.parent / "20260711_method1_qa_basis_formal_v1/artifacts/keypoint_basis_v10_protected_hyper_layout_candidate/schedule_layout_contract/memory_to_tile.csv"


def load_tile(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {str(row["memory_id"]): str(row["tile_id"]) for row in csv.DictReader(handle)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-revision", default="r1_attempt001")
    args = parser.parse_args()
    config = load_config("v21_sparse_joint_tuned_oof_r2.json")
    neutral_path = VERSION_ROOT / "artifacts/v21_effect_closure/v21_neutral_memory_to_tile.csv"
    layouts = {"neutral_source_order": load_tile(neutral_path), "legacy_all_train_hyper_diagnostic": load_tile(LEGACY_LAYOUT)}
    if len(layouts["neutral_source_order"]) != 42034 or set(layouts["neutral_source_order"]) != set(layouts["legacy_all_train_hyper_diagnostic"]):
        raise ValueError("legacy and v21 layouts do not cover the same memory corpus")
    rows: list[dict] = []
    for seed in map(int, config["fixed_seeds"]):
        for fold in range(int(config["outer_folds"])):
            path = VERSION_ROOT / "artifacts/v21_sparse_joint" / config["run_id"] / f"seed_{seed}" / f"fold_{fold}" / "attempts/attempt_001/v21_sparse_joint_selections.jsonl"
            for row in iter_jsonl(path):
                if not (
                    str(row["method"]) == "sparse_joint_semantic_bundle_hypergraph"
                    and int(row["activated_prototypes"]) == 16
                    and int(row["members_per_semantic_edge"]) == 32
                    and int(row["semantic_bundle_beam"]) == 32
                    and int(row["budget"]) in {8, 16, 32}
                ):
                    continue
                selected = tuple(map(str, row["selected_memory_ids"]))
                frontier = tuple(map(str, row["frontier_memory_ids"]))
                for name, tile in layouts.items():
                    rows.append({
                        "seed": seed, "fold": fold, "query_id": row["query_id"], "budget": row["budget"], "layout": name,
                        "selected_payload_tile_activations": len({tile[memory_id] for memory_id in selected}),
                        "frontier_feature_tile_activations": len({tile[memory_id] for memory_id in frontier}),
                    })
    expected = 1355 * len(config["fixed_seeds"]) * 3 * len(layouts)
    if len(rows) != expected:
        raise ValueError(f"expected {expected} rows, found {len(rows)}")
    metrics_path = VERSION_ROOT / "aggregate" / f"v21_legacy_hyper_layout_headroom_metrics_{args.output_revision}.csv"
    write_csv_new(metrics_path, rows)
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["layout"]), int(row["budget"]))].append(row)
    pooled = {
        f"{layout}|B={budget}": {
            "rows": len(items),
            "selected_payload_tile_activations_mean": float(np.mean([row["selected_payload_tile_activations"] for row in items])),
            "frontier_feature_tile_activations_mean": float(np.mean([row["frontier_feature_tile_activations"] for row in items])),
        }
        for (layout, budget), items in sorted(grouped.items())
    }
    summary = {
        "schema_version": "v21-legacy-hyper-layout-headroom-1",
        "status": "completed_diagnostic_not_oof_safe_not_deployable",
        "pooled": pooled,
        "legacy_layout_training_scope": "all v19 train queries; overlaps every v21 OOF target fold",
        "permitted_use": "layout-family headroom diagnosis only",
        "calibration_read": False,
        "formal_test_read": False,
        "artifacts": {"metrics": str(metrics_path.relative_to(VERSION_ROOT))},
        "hashes": {
            "legacy_layout": sha256_file(LEGACY_LAYOUT),
            "neutral_layout": sha256_file(neutral_path),
            "metrics": sha256_file(metrics_path),
        },
    }
    summary_path = VERSION_ROOT / "aggregate" / f"v21_legacy_hyper_layout_headroom_{args.output_revision}.json"
    write_json_new(summary_path, summary)
    print(json.dumps({"status": summary["status"], "B32": {key: value for key, value in pooled.items() if key.endswith("B=32")}}, sort_keys=True))


if __name__ == "__main__":
    main()
