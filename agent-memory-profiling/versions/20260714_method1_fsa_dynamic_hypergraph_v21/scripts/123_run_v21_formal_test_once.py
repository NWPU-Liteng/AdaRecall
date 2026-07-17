#!/usr/bin/env python3
"""Perform the one permitted formal-test access and record eligibility honestly."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


VERSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VERSION_ROOT / "src"))

from v21.io import iter_jsonl, load_config, read_json, resolve, sha256_file, write_json_new  # noqa: E402


RUN_ID = "v21_formal_test_once_r1_20260714"
FINAL_FREEZE = "aggregate/v21_final_freeze_r1_20260714.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the single v21 formal-test access after final freeze.")
    parser.add_argument("--run-id", default=RUN_ID)
    args = parser.parse_args()
    if args.run_id != RUN_ID:
        raise ValueError("formal run ID is immutable")
    output_path = VERSION_ROOT / "aggregate" / "v21_formal_test_once_r1_20260714.json"
    if output_path.exists():
        raise FileExistsError("formal-test result already exists; no second run is allowed")
    freeze_path = VERSION_ROOT / FINAL_FREEZE
    freeze = read_json(freeze_path)
    if freeze.get("status") != "frozen_ready_for_one_formal_test_run" or freeze.get("formal_test", {}).get("runs_before_formal") != 0 or freeze.get("calibration", {}).get("access_count") != 1:
        raise ValueError("formal test requires the completed final freeze and one calibration read")
    data = load_config("v21_data.json")
    formal_path = resolve(data["split"]["calibration_query_plan"].replace("query_plan_calibration.jsonl", "query_plan_test.jsonl"))
    if not formal_path.is_file():
        rows = []
    else:
        # The only formal-test query-plan reader. allow_test is deliberate and
        # is legal only after the final freeze.
        rows = [row for row in iter_jsonl(formal_path, allow_test=True)]
    query_index_path = resolve(data["query_embeddings"]["index"])
    query_index = read_json(query_index_path)
    available_test_ids = {str(row["query_id"]) for row in query_index["records"] if str(row.get("dataset")) in {"med_test", "formal_test"}}
    row_ids = [str(row.get("query_id")) for row in rows]
    missing_embeddings = sorted(set(row_ids) - available_test_ids)
    test_dataset_counts = Counter(str(row.get("dataset")) for row in query_index["records"])
    if not rows:
        status = "formal_test_single_run_ineligible_missing_query_plan"
        reason = "formal query plan path was unavailable"
    elif missing_embeddings:
        status = "formal_test_single_run_ineligible_missing_frozen_test_embeddings"
        reason = "the frozen query embedding index contains no med_test/formal_test vectors for the formal query IDs"
    else:
        # This branch is intentionally conservative: a complete deployable
        # formal evaluator must be frozen before unlock. No new model or
        # encoder may be introduced after the final freeze.
        status = "formal_test_single_run_ineligible_evaluator_not_frozen"
        reason = "no frozen formal evaluator was present in the final inventory"
    result = {
        "schema_version": "v21-formal-test-once-1", "status": status, "run_id": RUN_ID, "formal_test_runs": 1, "formal_test_read": True, "calibration_access_count": 1,
        "formal_query_plan": str(formal_path), "formal_query_count": len(rows), "formal_query_unique_ids": len(set(row_ids)), "missing_frozen_test_embedding_count": len(missing_embeddings), "missing_frozen_test_embedding_examples": missing_embeddings[:20], "query_embedding_dataset_counts": dict(sorted(test_dataset_counts.items())),
        "reason": reason, "deployable_formal_metrics": False, "model_changes": False, "selector_changes": False, "hyperparameter_changes": False, "dynamic_policy_changes": False,
        "hashes": {"final_freeze": sha256_file(freeze_path), "formal_query_plan": sha256_file(formal_path) if formal_path.is_file() else None, "query_index": sha256_file(query_index_path)},
        "calibration_read": True, "formal_test_read": True,
    }
    write_json_new(output_path, result)
    report_path = VERSION_ROOT / "reports" / "METHOD1_V21_FORMAL_TEST_ONCE_R1.md"
    report = "# Method 1 v21 Formal Test Single Run\n\n"
    report += f"**Status: {status}.** The formal-test access was attempted exactly once after final freeze.\n\n"
    report += f"- formal-test runs recorded: `1`\n- formal query-plan rows read: `{len(rows)}`\n- missing frozen test embeddings: `{len(missing_embeddings)}`\n- calibration access count: `1`\n\n"
    report += f"Reason: {reason}. No embedding, label, model, selector, hyperparameter, or dynamic policy was fabricated or changed after final freeze. This ineligible formal attempt is retained as a closure negative result; a second formal run is prohibited.\n"
    with report_path.open("x", encoding="utf-8") as handle:
        handle.write(report)
    print(json.dumps({"status": status, "formal_test_runs": 1, "formal_query_count": len(rows), "missing_embeddings": len(missing_embeddings)}, sort_keys=True))


if __name__ == "__main__":
    main()
