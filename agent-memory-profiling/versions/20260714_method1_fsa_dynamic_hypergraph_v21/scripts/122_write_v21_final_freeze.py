#!/usr/bin/env python3
"""Freeze the post-calibration Method 1 policy before the single formal run."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


VERSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VERSION_ROOT / "src"))

from v21.io import read_json, sha256_file, write_json_new  # noqa: E402


RUN_ID = "v21_final_freeze_r1_20260714"


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the immutable post-calibration final freeze.")
    parser.add_argument("--run-id", default=RUN_ID)
    args = parser.parse_args()
    if args.run_id != RUN_ID:
        raise ValueError("final freeze run ID is immutable")
    output_path = VERSION_ROOT / "aggregate" / "v21_final_freeze_r1_20260714.json"
    if output_path.exists():
        raise FileExistsError("final freeze already exists")
    pre_path = VERSION_ROOT / "aggregate" / "v21_pre_calibration_freeze_r1_20260714.json"
    calibration_path = VERSION_ROOT / "aggregate" / "v21_calibration_crc_ltt_r1_20260714.json"
    calibration_metrics_path = VERSION_ROOT / "aggregate" / "v21_calibration_crc_ltt_metrics_r1_20260714.csv"
    gate_path = VERSION_ROOT / "aggregate" / "v21_gate_c_decision_r1_20260714.json"
    dynamic_path = VERSION_ROOT / "aggregate" / "v21_dynamic_feasibility_audit_r1_20260714.json"
    pre = read_json(pre_path)
    calibration = read_json(calibration_path)
    gate = read_json(gate_path)
    dynamic = read_json(dynamic_path)
    if pre.get("status") != "frozen_ready_for_single_calibration_read" or calibration.get("status") != "completed_single_calibration_read" or calibration.get("calibration_access_count") != 1:
        raise ValueError("pre-freeze/calibration sequence is incomplete")
    if calibration.get("formal_test_runs") != 0 or calibration.get("formal_test_read") is not False:
        raise ValueError("formal-test lock is not intact")
    if gate.get("decision") != "PASS" or dynamic.get("decision") != "dynamic_update_not_eligible_static_fsa_qehg_fallback":
        raise ValueError("final freeze requires Gate C PASS and static fallback decision")
    unchanged = []
    changed = []
    for item in pre["inventory"]:
        path = VERSION_ROOT / item["path"] if not str(item["path"]).startswith("/") else Path(item["path"])
        current = sha256_file(path)
        (unchanged if current == item["sha256"] else changed).append({"path": item["path"], "expected": item["sha256"], "current": current})
    if changed:
        raise ValueError(f"pre-calibration source artifacts changed: {changed[:3]}")
    selected_budget = int(calibration["crc_selected_budget"])
    selected_policy = {"grid": calibration["fixed_policy"]["grid"], "budget": selected_budget, "protected_ann_seeds": calibration["fixed_policy"]["protected_ann_seeds"], "model_seed": calibration["fixed_policy"]["model_seed"], "selector": calibration["fixed_policy"]["selector"], "dynamic_policy": "static_fsa_qehg_fallback"}
    hashes = {
        "pre_calibration_freeze": sha256_file(pre_path), "calibration_summary": sha256_file(calibration_path), "calibration_metrics": sha256_file(calibration_metrics_path), "gate_c": sha256_file(gate_path), "dynamic_audit": sha256_file(dynamic_path),
        "calibration_runner": sha256_file(VERSION_ROOT / "scripts" / "121_run_v21_calibration_crc_ltt.py"), "final_freeze_writer": sha256_file(VERSION_ROOT / "scripts" / "122_write_v21_final_freeze.py"),
    }
    final = {
        "schema_version": "v21-final-freeze-1", "status": "frozen_ready_for_one_formal_test_run", "run_id": RUN_ID,
        "policy": selected_policy, "gate_c": {"decision": gate["decision"], "artifact": str(gate_path.relative_to(VERSION_ROOT))}, "dynamic": {"decision": dynamic["decision"], "artifact": str(dynamic_path.relative_to(VERSION_ROOT))},
        "calibration": {"access_count": 1, "artifact": str(calibration_path.relative_to(VERSION_ROOT)), "crc_selected_budget": selected_budget, "crc_pass": calibration["budgets"][str(selected_budget)]["crc_pass"], "ltt_pass_after_holm": calibration["budgets"][str(selected_budget)]["ltt_pass_after_holm"], "model_changes": False, "selector_changes": False, "hyperparameter_changes": False, "dynamic_policy_changes": False},
        "source_inventory": {"pre_calibration_inventory_count": pre["inventory_count"], "pre_calibration_inventory_hash": sha256_file(pre_path), "all_pre_calibration_sources_unchanged": True},
        "formal_test": {"locked_until_this_manifest": False, "runs_before_formal": 0, "read_before_formal": False, "single_run_only": True},
        "closure_order": ["pre_calibration_freeze", "single_calibration_crc_ltt", "final_freeze", "single_formal_test", "closure_manifest"],
        "hashes": hashes, "calibration_read": True, "formal_test_read": False, "formal_test_runs": 0,
    }
    write_json_new(output_path, final)
    report_path = VERSION_ROOT / "reports" / "METHOD1_V21_FINAL_FREEZE_R1.md"
    report = "# Method 1 v21 Final Freeze\n\n"
    report += "The Method 1 policy is frozen after the single calibration CRC/LTT read. All pre-calibration source hashes remain unchanged; the calibration result did not pass the declared risk target, so B=32 is retained as the fallback budget.\n\n"
    report += f"- fixed grid: `{selected_policy['grid']}`\n- frozen budget: `{selected_budget}`\n- CRC pass: `{calibration['budgets'][str(selected_budget)]['crc_pass']}`\n- LTT after Holm: `{calibration['budgets'][str(selected_budget)]['ltt_pass_after_holm']}`\n- dynamic policy: `static_fsa_qehg_fallback`\n- calibration access count: `1`\n- formal-test runs: `0`\n\n"
    report += "No model, prototype, selector, hyperparameter, split, payload layout, or dynamic policy may change after this freeze. The formal test is permitted exactly once and will be recorded even if the frozen corpus cannot provide a legal test embedding.\n"
    with report_path.open("x", encoding="utf-8") as handle:
        handle.write(report)
    print(json.dumps({"status": final["status"], "budget": selected_budget, "calibration_access_count": 1, "formal_test_runs": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
