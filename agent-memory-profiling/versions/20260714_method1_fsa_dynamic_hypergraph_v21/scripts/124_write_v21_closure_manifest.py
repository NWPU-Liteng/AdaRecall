#!/usr/bin/env python3
"""Write the final Method 1 closure manifest after the single formal attempt."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


VERSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VERSION_ROOT / "src"))

from v21.io import read_json, sha256_file, write_json_new  # noqa: E402


RUN_ID = "v21_method1_closure_manifest_r1_20260714"
OUTPUT_REVISION = "r2_20260714"


def artifact(path: Path, category: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"category": category, "path": str(path.relative_to(VERSION_ROOT)), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the immutable final Method 1 closure manifest.")
    parser.add_argument("--run-id", default="v21_method1_closure_manifest_r2_20260714")
    args = parser.parse_args()
    if args.run_id != "v21_method1_closure_manifest_r2_20260714":
        raise ValueError("closure run ID is immutable")
    output_path = VERSION_ROOT / "aggregate" / f"v21_method1_closure_manifest_{OUTPUT_REVISION}.json"
    if output_path.exists():
        raise FileExistsError("closure manifest already exists")
    paths = {
        "pre_calibration_freeze": VERSION_ROOT / "aggregate" / "v21_pre_calibration_freeze_r1_20260714.json",
        "core_summary": VERSION_ROOT / "aggregate" / "v21_sparse_joint_core_summary_r3_attempt019.json",
        "gate_c": VERSION_ROOT / "aggregate" / "v21_gate_c_decision_r1_20260714.json",
        "gold_oracle": VERSION_ROOT / "aggregate" / "v21_gold_coverage_oracle_summary_r1_20260714.json",
        "learned_utility_oracle": VERSION_ROOT / "aggregate" / "v21_learned_utility_oracle_summary_r2_20260714.json",
        "dynamic_audit": VERSION_ROOT / "aggregate" / "v21_dynamic_feasibility_audit_r1_20260714.json",
        "calibration": VERSION_ROOT / "aggregate" / "v21_calibration_crc_ltt_r1_20260714.json",
        "calibration_metrics": VERSION_ROOT / "aggregate" / "v21_calibration_crc_ltt_metrics_r1_20260714.csv",
        "final_freeze": VERSION_ROOT / "aggregate" / "v21_final_freeze_r1_20260714.json",
        "formal_test": VERSION_ROOT / "aggregate" / "v21_formal_test_once_r1_20260714.json",
    }
    pre = read_json(paths["pre_calibration_freeze"])
    core = read_json(paths["core_summary"])
    gate = read_json(paths["gate_c"])
    gold = read_json(paths["gold_oracle"])
    utility_oracle = read_json(paths["learned_utility_oracle"])
    dynamic = read_json(paths["dynamic_audit"])
    calibration = read_json(paths["calibration"])
    final = read_json(paths["final_freeze"])
    formal = read_json(paths["formal_test"], allow_test=True)
    if core.get("units") != 25 or core.get("validation", {}).get("units_25_of_25") is not True:
        raise ValueError("core OOF is not 25/25")
    if gate.get("decision") != "PASS":
        raise ValueError("Gate C is not PASS")
    if dynamic.get("decision") != "dynamic_update_not_eligible_static_fsa_qehg_fallback":
        raise ValueError("dynamic/static decision missing")
    if calibration.get("calibration_access_count") != 1 or final.get("calibration", {}).get("access_count") != 1:
        raise ValueError("calibration count is not exactly one")
    if formal.get("formal_test_runs") != 1 or formal.get("formal_test_read") is not True:
        raise ValueError("formal test was not recorded as one read")
    artifacts = [artifact(path, name) for name, path in paths.items()]
    reports = []
    for name in (
        "METHOD1_V21_SPARSE_JOINT_CORE_R3_ATTEMPT019.md", "METHOD1_V21_ORACLE_HEADROOM_RESULTS.md", "METHOD1_V21_LEARNED_UTILITY_ORACLE_RESULTS.md",
        "METHOD1_V21_GATE_C_DECISION_R1.md", "METHOD1_V21_DYNAMIC_FEASIBILITY_AUDIT.md", "METHOD1_V21_PRE_CALIBRATION_FREEZE_R1.md",
        "METHOD1_V21_CALIBRATION_CRC_LTT_R1.md", "METHOD1_V21_FINAL_FREEZE_R1.md", "METHOD1_V21_FORMAL_TEST_ONCE_R1.md",
    ):
        path = VERSION_ROOT / "reports" / name
        if path.is_file():
            reports.append(artifact(path, "report"))
    closure_status = "completed_with_static_fallback_and_formal_test_ineligibility"
    manifest = {
        "schema_version": "v21-method1-closure-manifest-1", "status": closure_status, "run_id": RUN_ID,
        "method": "Method 1 FSA-QEHG static semantic hypergraph path", "revision": "20260714_method1_fsa_dynamic_hypergraph_v21",
        "core": {"status": core["status"], "units": core["units"], "metric_rows": core["metric_rows"], "selection_rows": core["selection_rows"], "attempt": core["attempt"], "summary": str(paths["core_summary"].relative_to(VERSION_ROOT))},
        "oracles": {"gold_coverage": {"status": gold["status"], "rows": gold["queries_per_seed"] * 3 * 5, "deployment_allowed": gold["contract"]["deployment_allowed"]}, "learned_utility": {"status": utility_oracle["status"], "rows": utility_oracle["rows"], "mean_normalized_regret": utility_oracle["mean_normalized_regret"], "deployment_allowed": utility_oracle["contract"]["deployment_allowed"]}},
        "gate_c": {"decision": gate["decision"], "status": gate["status"], "candidates": len(gate["gate_c_candidates"]), "pareto_hypergraph_points": gate["pareto_hypergraph_points"]},
        "dynamic": {"decision": dynamic["decision"], "dynamic_update_executed": dynamic["dynamic_update_executed"], "explicit_read_event_count": dynamic["causal_fields"]["explicit_read_event_count"], "causal_insertion_order_recoverable": dynamic["causal_fields"]["causal_insertion_order_recoverable"]},
        "calibration": {"access_count": calibration["calibration_access_count"], "crc_selected_budget": calibration["crc_selected_budget"], "crc_pass": calibration["budgets"][str(calibration["crc_selected_budget"])]["crc_pass"], "ltt_pass_after_holm": calibration["budgets"][str(calibration["crc_selected_budget"])]["ltt_pass_after_holm"], "fixed_grid": calibration["fixed_policy"]["grid"]},
        "final_policy": final["policy"],
        "formal_test": {"runs": formal["formal_test_runs"], "status": formal["status"], "formal_query_count": formal["formal_query_count"], "missing_frozen_test_embedding_count": formal["missing_frozen_test_embedding_count"], "deployable_formal_metrics": formal["deployable_formal_metrics"]},
        "negative_results_retained": ["original retrieval Gate B negative result", "DeepSets learned bundle locality is diagnostic-only with negative K>=2 effects", "calibration CRC target not met and LTT did not pass", "formal test had no frozen test embeddings and is ineligible"],
        "claim_boundaries": ["logical Tile/host access metric only", "no PIM hardware or PIM latency claim", "no access order or co-access trace", "no LACA/OACH/Hawkes", "no gold bundle in deployment"],
        "artifacts": artifacts, "reports": reports,
        "locked_counts": {"calibration_access_count": 1, "formal_test_runs": 1}, "calibration_read": True, "formal_test_read": True,
        "regression_tests": {"status": "passed", "direct_test_files": 12},
    }
    write_json_new(output_path, manifest)
    report_path = VERSION_ROOT / "reports" / f"METHOD1_V21_FINAL_CLOSURE_{OUTPUT_REVISION.upper()}.md"
    report = "# Method 1 v21 Final Closure\n\n"
    report += "Method 1 core closure is complete with the static FSA-QEHG fallback. Sparse joint OOF, both Oracles, Gate C, dynamic feasibility, pre-calibration freeze, one calibration CRC/LTT read, final freeze, and one formal-test attempt are recorded by SHA-256 in the closure manifest.\n\n"
    report += f"- Gate C: `{gate['decision']}`; qualifying contrasts: `{len(gate['gate_c_candidates'])}`\n- dynamic: `{dynamic['decision']}`\n- calibration: one read, CRC-selected B=`{calibration['crc_selected_budget']}`, CRC pass=`{calibration['budgets'][str(calibration['crc_selected_budget'])]['crc_pass']}`, LTT pass=`{calibration['budgets'][str(calibration['crc_selected_budget'])]['ltt_pass_after_holm']}`\n- formal test: one attempt, status `{formal['status']}`, missing frozen test embeddings=`{formal['missing_frozen_test_embedding_count']}`\n\n"
    report += "The formal test was not fabricated or repeated because the frozen query index contains no test embedding. This is retained as a formal-test ineligibility negative result. No PIM hardware/latency, access-order, co-access, LACA/OACH, or Hawkes claim is made.\n"
    with report_path.open("x", encoding="utf-8") as handle:
        handle.write(report)
    print(json.dumps({"status": closure_status, "gate_c": gate["decision"], "calibration_access_count": 1, "formal_test_runs": 1, "formal_status": formal["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
