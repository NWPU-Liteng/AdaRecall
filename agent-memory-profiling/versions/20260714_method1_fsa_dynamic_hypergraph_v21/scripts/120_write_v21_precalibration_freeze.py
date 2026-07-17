#!/usr/bin/env python3
"""Freeze every Method 1 source before the single calibration read."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


VERSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VERSION_ROOT / "src"))

from v21.io import load_config, read_json, resolve, sha256_file, write_json_new  # noqa: E402


RUN_ID = "v21_pre_calibration_freeze_r1_20260714"
SPARSE_RUN = "v21_sparse_joint_oof_r2_20260714"
SPARSE_ATTEMPT = "attempt_019"
SEEDS = (20260712, 20260713, 20260714, 20260715, 20260716)


def add_file(inventory: dict[str, dict], path: Path, category: str) -> None:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    key = str(path)
    previous = inventory.get(key)
    if previous is not None and previous["category"] != category:
        previous["category"] = f"{previous['category']}+{category}"
        return
    if previous is None:
        inventory[key] = {"category": category, "bytes": path.stat().st_size, "sha256": sha256_file(path), "path": str(path.relative_to(VERSION_ROOT)) if path.is_relative_to(VERSION_ROOT) else str(path)}


def add_config_files(inventory: dict[str, dict]) -> None:
    names = [
        "v21_data.json", "v21_freeze.json", "v21_effect_closure.json", "v21_method1_closure_r2.json", "v21_method1_closure_r3.json",
        "v21_phase1_oof.json", "v21_uot_semantic.json", "v21_qesa_r2.json", "v21_qhds.json", "v21_sparse_joint_oof_r2.json",
        "v21_sparse_joint.json", "v21_eligibility.json", "v21_evidence_association.json", "v21_bundle_locality.json",
    ]
    for name in names:
        add_file(inventory, VERSION_ROOT / "configs" / name, "source_config")


def add_source_inputs(inventory: dict[str, dict], data: dict) -> None:
    for label, raw_path in (
        ("frozen_memory_items", data["memory"]["items"]),
        ("frozen_memory_embeddings", data["memory"]["embedding_vectors"]),
        ("frozen_memory_embedding_ids", data["memory"]["embedding_ids"]),
        ("frozen_query_embeddings", data["query_embeddings"]["vectors"]),
        ("frozen_query_embedding_index", data["query_embeddings"]["index"]),
        ("frozen_train_ann_prefix", data["frozen_baseline"]["train_retrieval"]),
    ):
        add_file(inventory, resolve(raw_path), f"source_input:{label}")


def add_unit_artifacts(inventory: dict[str, dict], config: dict) -> None:
    for seed in SEEDS:
        for fold in range(5):
            phase_dir = VERSION_ROOT / "artifacts" / "v21_phase1" / config["phase1_run_id"] / f"seed_{seed}" / f"fold_{fold}"
            phase_manifest = phase_dir / "v21_unit_manifest.json"
            phase = read_json(phase_manifest)
            add_file(inventory, phase_manifest, "phase1_manifest")
            for key in ("metrics", "incidence", "model", "projection", "uot_semantic_incidence"):
                raw = phase.get("artifacts", {}).get(key)
                if raw:
                    add_file(inventory, VERSION_ROOT / raw, f"phase1_artifact:{key}")
            qesa_dir = VERSION_ROOT / "artifacts" / "v21_qesa" / config["qesa_run_id"] / f"seed_{seed}" / f"fold_{fold}"
            qesa_manifest = qesa_dir / "v21_qesa_r2_unit_manifest.json"
            qesa = read_json(qesa_manifest)
            add_file(inventory, qesa_manifest, "qesa_manifest")
            for key, raw in qesa.get("artifacts", {}).items():
                add_file(inventory, VERSION_ROOT / raw, f"qesa_artifact:{key}")
            qhds_dir = VERSION_ROOT / "artifacts" / "v21_qhds" / config["dense_qhds_run_id"] / f"seed_{seed}" / f"fold_{fold}"
            qhds_manifest = qhds_dir / "v21_qhds_unit_manifest.json"
            qhds = read_json(qhds_manifest)
            add_file(inventory, qhds_manifest, "qhds_manifest")
            for key, raw in qhds.get("artifacts", {}).items():
                add_file(inventory, VERSION_ROOT / raw, f"qhds_artifact:{key}")
            index_dir = VERSION_ROOT / "artifacts" / "v21_sparse_semantic_index" / config["index_run_id"] / f"seed_{seed}" / f"fold_{fold}" / "attempts" / config["index_attempt"]
            index_manifest_path = index_dir / "v21_sparse_semantic_index_manifest.json"
            index = read_json(index_manifest_path)
            add_file(inventory, index_manifest_path, "sparse_index_manifest")
            for key, raw in index.get("artifacts", {}).items():
                add_file(inventory, VERSION_ROOT / raw, f"sparse_index_artifact:{key}")
            sparse_dir = VERSION_ROOT / "artifacts" / "v21_sparse_joint" / SPARSE_RUN / f"seed_{seed}" / f"fold_{fold}" / "attempts" / SPARSE_ATTEMPT
            sparse_manifest_path = sparse_dir / "v21_sparse_joint_unit_manifest.json"
            sparse = read_json(sparse_manifest_path)
            if sparse.get("status") != "completed" or sparse.get("calibration_read") is not False or sparse.get("formal_test_read") is not False:
                raise ValueError(f"sparse unit is not frozen/unlocked: {sparse_manifest_path}")
            add_file(inventory, sparse_manifest_path, "sparse_joint_manifest")
            for key, raw in sparse.get("artifacts", {}).items():
                add_file(inventory, VERSION_ROOT / raw, f"sparse_joint_artifact:{key}")


def add_analysis_artifacts(inventory: dict[str, dict]) -> None:
    patterns = [
        "v21_bundle_locality_r1_phase_decision.json", "v21_bundle_locality_oof_summary_r1.json", "v21_sparse_semantic_index_summary_r1.json",
        "v21_bundle_layout_contract.json", "v21_sparse_joint_*_r3_attempt019.*", "v21_gold_coverage_oracle_*r1_20260714.*",
        "v21_learned_utility_oracle_*r2_20260714.*", "v21_gate_c_*", "v21_dynamic_feasibility_audit_r1_20260714.json",
        "v21_manifest_qesa_gate_b2_passed_r1.json", "v21_manifest_qesa_gate_b_negative_r1.json",
    ]
    for pattern in patterns:
        for path in sorted((VERSION_ROOT / "aggregate").glob(pattern)):
            add_file(inventory, path, "analysis_artifact")
    add_file(inventory, VERSION_ROOT / "artifacts" / "v21_effect_closure" / "v21_neutral_memory_to_tile.csv", "neutral_layout")
    add_file(inventory, VERSION_ROOT / "artifacts" / "v21_evidence_sets" / "train_query_evidence_incidence.jsonl", "train_evidence_incidence")
    add_file(inventory, VERSION_ROOT / "artifacts" / "v21_evidence_bundles" / "v21_train_acceptable_evidence_bundles.jsonl", "train_bundle_supervision")


def add_code_artifacts(inventory: dict[str, dict]) -> None:
    for relative in (
        "scripts/115_train_v21_sparse_joint_oof.py", "scripts/116_launch_v21_sparse_joint_oof.py", "scripts/117_aggregate_v21_sparse_joint_core.py",
        "scripts/109_run_v21_gold_coverage_oracle.py", "scripts/110_run_v21_learned_utility_oracle.py", "scripts/118_evaluate_v21_gate_c.py", "scripts/119_audit_v21_dynamic_feasibility.py",
        "src/v21/sparse_joint.py", "src/v21/sparse_semantic_index.py", "src/v21/qesa_r2.py", "src/v21/qhds.py", "src/v21/online_latency.py",
    ):
        add_file(inventory, VERSION_ROOT / relative, "source_code")


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the immutable v21 pre-calibration freeze.")
    parser.add_argument("--run-id", default=RUN_ID)
    args = parser.parse_args()
    if args.run_id != RUN_ID:
        raise ValueError("pre-calibration freeze run ID is immutable")
    data = load_config("v21_data.json")
    config = load_config("v21_sparse_joint_oof_r2.json")
    gate = read_json(VERSION_ROOT / "aggregate" / "v21_gate_c_decision_r1_20260714.json")
    dynamic = read_json(VERSION_ROOT / "aggregate" / "v21_dynamic_feasibility_audit_r1_20260714.json")
    if gate.get("decision") != "PASS" or dynamic.get("decision") != "dynamic_update_not_eligible_static_fsa_qehg_fallback":
        raise ValueError("pre-calibration freeze requires Gate C PASS and dynamic/static decision")
    inventory: dict[str, dict] = {}
    add_config_files(inventory)
    add_source_inputs(inventory, data)
    add_unit_artifacts(inventory, config)
    add_analysis_artifacts(inventory)
    add_code_artifacts(inventory)
    by_category: dict[str, int] = {}
    for entry in inventory.values():
        by_category[entry["category"]] = by_category.get(entry["category"], 0) + 1
    freeze = {
        "schema_version": "v21-pre-calibration-freeze-1", "status": "frozen_ready_for_single_calibration_read", "run_id": RUN_ID,
        "source_revision": "v21_method1_closure_r3_20260714", "sparse_joint_run_id": SPARSE_RUN, "sparse_joint_attempt": SPARSE_ATTEMPT,
        "gate_c_decision": gate["decision"], "dynamic_decision": dynamic["decision"],
        "inventory_count": len(inventory), "inventory_by_category": dict(sorted(by_category.items())), "inventory": sorted(inventory.values(), key=lambda item: item["path"]),
        "locked_split_state": {"calibration_access_count": 0, "calibration_read": False, "formal_test_runs": 0, "formal_test_read": False, "calibration_model_changes_allowed": False, "formal_test_runs_allowed": False},
        "post_freeze_changes_allowed": ["CRC/LTT risk mapping only after one calibration read", "final freeze metadata", "one formal test after final freeze"],
        "post_freeze_changes_prohibited": ["model weights", "prototype assignments", "candidate frontier", "selector architecture", "hyperparameters", "dynamic policy", "data split", "payload layout"],
        "hashes": {"gate_c": sha256_file(VERSION_ROOT / "aggregate" / "v21_gate_c_decision_r1_20260714.json"), "dynamic_audit": sha256_file(VERSION_ROOT / "aggregate" / "v21_dynamic_feasibility_audit_r1_20260714.json"), "config": sha256_file(VERSION_ROOT / "configs" / "v21_freeze.json")},
        "calibration_read": False, "formal_test_read": False,
    }
    output_path = VERSION_ROOT / "aggregate" / "v21_pre_calibration_freeze_r1_20260714.json"
    write_json_new(output_path, freeze)
    report_path = VERSION_ROOT / "reports" / "METHOD1_V21_PRE_CALIBRATION_FREEZE_R1.md"
    report = "# Method 1 v21 Pre-Calibration Freeze\n\n"
    report += "All Method 1 source configurations, models, sparse indices, neutral layout, train-only supervision artifacts, OOF aggregates, Oracle outputs, Gate C decision, dynamic audit, and source code are frozen by SHA-256 in `aggregate/v21_pre_calibration_freeze_r1_20260714.json`.\n\n"
    report += f"- inventory entries: {len(inventory)}\n- Gate C: `{gate['decision']}`\n- dynamic decision: `{dynamic['decision']}`\n- calibration access count: `0`\n- formal-test runs: `0`\n\n"
    report += "After this freeze, the only permitted development read is one calibration pass for CRC/LTT risk mapping. No model, prototype, selector, hyperparameter, dynamic policy, split, or layout change is permitted. Formal test remains locked until final freeze.\n"
    with report_path.open("x", encoding="utf-8") as handle:
        handle.write(report)
    print(json.dumps({"status": freeze["status"], "inventory_count": len(inventory), "calibration_access_count": 0, "formal_test_runs": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
