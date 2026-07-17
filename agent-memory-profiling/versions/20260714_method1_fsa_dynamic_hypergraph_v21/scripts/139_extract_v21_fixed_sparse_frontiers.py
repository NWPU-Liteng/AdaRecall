#!/usr/bin/env python3
"""Extract immutable fixed-point frontiers once for sparse-selector refitting."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


VERSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VERSION_ROOT / "src"))

from v21.io import iter_jsonl, load_config, read_json, sha256_file, write_json_new  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-config", default="v21_sparse_joint_tuned_oof_r2.json")
    parser.add_argument("--run-id", default="v21_fixed_sparse_frontiers_r1_20260715")
    parser.add_argument("--attempt", default="attempt_001")
    args = parser.parse_args()
    config = load_config(args.source_config)
    root = VERSION_ROOT / "artifacts/v21_fixed_sparse_frontiers" / args.run_id
    completed = 0
    for seed in map(int, config["fixed_seeds"]):
        for fold in range(int(config["outer_folds"])):
            unit = root / f"seed_{seed}" / f"fold_{fold}" / "attempts" / args.attempt
            unit.mkdir(parents=True, exist_ok=False)
            source = VERSION_ROOT / "artifacts/v21_sparse_joint" / config["run_id"] / f"seed_{seed}" / f"fold_{fold}" / "attempts/attempt_001"
            source_manifest = read_json(source / "v21_sparse_joint_unit_manifest.json")
            if source_manifest.get("calibration_read") is not False or source_manifest.get("formal_test_read") is not False:
                raise ValueError("source frontier touched a locked split")
            source_path = source / "v21_sparse_joint_selections.jsonl"
            output_path = unit / "v21_fixed_sparse_frontiers.jsonl"
            rows = []
            for row in iter_jsonl(source_path):
                if (
                    str(row["method"]) == "sparse_joint_semantic_bundle_hypergraph"
                    and int(row["activated_prototypes"]) == 16
                    and int(row["members_per_semantic_edge"]) == 32
                    and int(row["semantic_bundle_beam"]) == 32
                    and int(row["budget"]) == 32
                ):
                    rows.append({
                        "query_id": str(row["query_id"]),
                        "seed": seed,
                        "fold": fold,
                        "frontier_memory_ids": list(map(str, row["frontier_memory_ids"])),
                        "protected_ann_memory_ids": list(map(str, row["frontier_memory_ids"][:5])),
                    })
            expected = int(source_manifest["outer_heldout_queries"])
            if len(rows) != expected or any(len(row["frontier_memory_ids"]) != 96 for row in rows):
                raise ValueError(f"fixed frontier extraction mismatch seed={seed} fold={fold}")
            with output_path.open("x", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
            manifest = {
                "schema_version": "v21-fixed-sparse-frontier-unit-1",
                "status": "completed",
                "run_id": args.run_id,
                "attempt": args.attempt,
                "seed": seed,
                "fold": fold,
                "queries": len(rows),
                "frontier_size": 96,
                "protected_ann_seeds": 5,
                "calibration_read": False,
                "formal_test_read": False,
                "artifacts": {"frontiers": str(output_path.relative_to(VERSION_ROOT))},
                "hashes": {
                    "source_manifest": sha256_file(source / "v21_sparse_joint_unit_manifest.json"),
                    "source_selections": sha256_file(source_path),
                    "frontiers": sha256_file(output_path),
                },
            }
            write_json_new(unit / "v21_fixed_sparse_frontier_manifest.json", manifest)
            completed += 1
            print(json.dumps({"seed": seed, "fold": fold, "queries": len(rows), "status": "completed"}), flush=True)
    print(json.dumps({"status": "completed", "units": completed}, sort_keys=True))


if __name__ == "__main__":
    main()
