#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


VERSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VERSION_ROOT / "src"))

from v21.io import load_config, read_json, sha256_file, write_json_new  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate immutable v21 sparse semantic indexes.")
    parser.add_argument("--run-id", default="v21_sparse_semantic_index_r1_20260714")
    parser.add_argument("--attempt", default="attempt_001")
    args = parser.parse_args()
    config = load_config("v21_sparse_joint.json")
    if args.run_id != config["index_run_id"]:
        raise ValueError("run ID must match sparse-index preregistration")
    root = VERSION_ROOT / "artifacts" / "v21_sparse_semantic_index" / args.run_id
    manifests = []
    for seed in config["fixed_seeds"]:
        for fold in range(int(config["outer_folds"])):
            path = root / f"seed_{seed}" / f"fold_{fold}" / "attempts" / args.attempt / "v21_sparse_semantic_index_manifest.json"
            if not path.is_file():
                raise FileNotFoundError(f"missing sparse index unit: {path}")
            manifest = read_json(path)
            if manifest["status"] != "completed" or manifest["attempt"] != args.attempt:
                raise ValueError(f"invalid sparse index manifest: {path}")
            manifests.append(manifest)
    metadata = [int(item["metadata_bytes"]) for item in manifests]
    shapes = {tuple(item["rank_index_shape"]) for item in manifests}
    incidence_hashes = {item["hashes"]["incidence_array"] for item in manifests}
    if len(shapes) != 1 or len(incidence_hashes) != len(manifests):
        raise ValueError("sparse index shapes must agree and each OOF unit must retain its own incidence provenance")
    result = {
        "schema_version": "v21-sparse-semantic-index-summary-1", "status": "completed", "revision_id": config["revision_id"],
        "run_id": args.run_id, "attempt": args.attempt, "units": len(manifests), "rank_index_shape": list(next(iter(shapes))),
        "metadata_bytes_per_unit": {"mean": float(np.mean(metadata)), "min": int(min(metadata)), "max": int(max(metadata)), "total": int(sum(metadata))},
        "online_contract": config["online_lookup"], "bundle_component_policy": config["bundle_component_policy"],
        "source_incidence_hash_count": len(incidence_hashes),
        "index_hashes": {f"seed_{item['seed']}_fold_{item['fold']}": item["hashes"]["rank_index"] for item in manifests},
        "hashes": {"config": sha256_file(VERSION_ROOT / "configs" / "v21_sparse_joint.json")},
        "calibration_read": False, "formal_test_read": False,
    }
    output = VERSION_ROOT / "aggregate" / "v21_sparse_semantic_index_summary_r1.json"
    write_json_new(output, result)
    report = "# Method 1 v21 Sparse Semantic Index r1\n\n"
    report += "All 25 seed/fold UOT incidence matrices have immutable prototype-to-memory rank indexes. Index construction is offline only. Online lookup may inspect only rank-list prefixes until it finds the configured number of eligible members, and must report every visited incidence entry.\n\n"
    report += f"- Rank-index shape per unit: `{result['rank_index_shape']}`\n- Metadata bytes per unit: `{int(np.mean(metadata))}`\n- Total immutable rank metadata bytes across 25 OOF units: `{sum(metadata)}`\n- No calibration or formal-test input was read.\n\n"
    report += "The retained learned bundle-locality scorer is not an approved primary component. It may only appear in a diagnostic ablation; the sparse joint primary candidate path starts from independent semantic-edge member ranking and must still face the budgeted pairwise factor graph.\n"
    report_path = VERSION_ROOT / "reports" / "METHOD1_V21_SPARSE_SEMANTIC_INDEX_R1.md"
    with report_path.open("x", encoding="utf-8") as handle:
        handle.write(report)
    print(json.dumps({"status": result["status"], "units": result["units"], "metadata_total": sum(metadata)}, sort_keys=True))


if __name__ == "__main__":
    main()
