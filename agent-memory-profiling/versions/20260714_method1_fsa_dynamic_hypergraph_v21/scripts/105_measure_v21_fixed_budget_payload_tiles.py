#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


VERSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VERSION_ROOT / "src"))

from v21.io import iter_jsonl, load_config, sha256_file, write_csv_new, write_json_new  # noqa: E402


def main() -> None:
    data = load_config("v21_data.json")
    qhds = load_config("v21_qhds.json")
    layout_path = VERSION_ROOT / "artifacts" / "v21_effect_closure" / "v21_neutral_memory_to_tile.csv"
    with layout_path.open(newline="") as handle:
        layout_rows = list(csv.DictReader(handle))
    tile = {row["memory_id"]: row["tile_id"] for row in layout_rows}
    payload = {row["memory_id"]: int(row["payload_bytes"]) for row in layout_rows}
    groups = {str(row["query_id"]): [set(map(str, group["acceptable_memory_ids"])) for group in row["evidence_groups"]] for row in iter_jsonl(data["split"]["train_query_plan"])}
    output = []
    run_dir = VERSION_ROOT / "artifacts" / "v21_qhds" / qhds["run_id"]
    for seed in qhds["fixed_seeds"]:
        for fold in range(5):
            selection_path = run_dir / f"seed_{seed}" / f"fold_{fold}" / "v21_qhds_selections.jsonl"
            for row in iter_jsonl(selection_path):
                selected = list(map(str, row["selected_memory_ids"]))
                selected_tiles = {tile[memory_id] for memory_id in selected}
                evidence_ids = set().union(*groups[row["query_id"]])
                effective_tiles = {tile[memory_id] for memory_id in selected if memory_id in evidence_ids}
                output.append({
                    "query_id": row["query_id"], "seed": int(seed), "fold": fold,
                    "method": row["method"], "budget": int(row["budget"]),
                    "candidate_count": len(selected),
                    "payload_bytes": sum(payload[memory_id] for memory_id in selected),
                    "payload_tile_activations": len(selected_tiles),
                    "effective_evidence_tiles": len(effective_tiles),
                    "redundant_tile_activations": len(selected_tiles.difference(effective_tiles)),
                })
    output_csv = VERSION_ROOT / "aggregate" / "v21_fixed_budget_payload_tile_metrics.csv"
    write_csv_new(output_csv, output)
    summary = {}
    for budget in qhds["budgets"]:
        summary[str(budget)] = {}
        for method in sorted({row["method"] for row in output}):
            rows = [row for row in output if row["method"] == method and row["budget"] == budget]
            summary[str(budget)][method] = {
                key: float(np.mean([row[key] for row in rows]))
                for key in ("payload_bytes", "payload_tile_activations", "effective_evidence_tiles", "redundant_tile_activations")
            }
    result = {
        "schema_version": "v21-fixed-budget-payload-tile-result-1",
        "status": "payload_tile_baseline_completed_metadata_and_latency_pending",
        "rows": len(output), "summary": summary,
        "layout_policy": "neutral_source_order_fixed_for_all_methods",
        "tile_capacity": 16,
        "claim_boundary": "logical payload Tile counts, not PIM timing",
        "hashes": {"metrics": sha256_file(output_csv), "layout": sha256_file(layout_path)},
        "calibration_read": False, "formal_test_read": False
    }
    write_json_new(VERSION_ROOT / "aggregate" / "v21_fixed_budget_payload_tile_summary.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__": main()
