#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path


VERSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VERSION_ROOT / "src"))

from v21.io import sha256_file, write_json_new  # noqa: E402


def artifact(path: str) -> dict:
    resolved = VERSION_ROOT / path
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {"path": path, "sha256": sha256_file(resolved), "bytes": resolved.stat().st_size}


def main() -> None:
    r1 = json.loads((VERSION_ROOT / "aggregate" / "v21_qesa_oof_summary.json").read_text(encoding="utf-8"))
    r2 = json.loads((VERSION_ROOT / "aggregate" / "v21_qesa_oof_summary_r2.json").read_text(encoding="utf-8"))
    headroom = json.loads((VERSION_ROOT / "aggregate" / "v21_qesa_gold_prototype_headroom_r1.json").read_text(encoding="utf-8"))
    if r1["status"] != "not_passed" or r2["status"] != "not_passed":
        raise RuntimeError("negative Gate B freeze requires both recorded QESA revisions to be not_passed")
    if r1.get("calibration_read") or r1.get("formal_test_read") or r2.get("calibration_read") or r2.get("formal_test_read"):
        raise RuntimeError("locked split access flag is nonzero")
    r1_units = list((VERSION_ROOT / "artifacts" / "v21_qesa" / r1["run_id"]).glob("seed_*/fold_*/v21_qesa_unit_manifest.json"))
    r2_units = list((VERSION_ROOT / "artifacts" / "v21_qesa" / r2["run_id"]).glob("seed_*/fold_*/v21_qesa_r2_unit_manifest.json"))
    if len(r1_units) != 25 or len(r2_units) != 25:
        raise RuntimeError(f"expected 25 immutable units per revision, found r1={len(r1_units)} r2={len(r2_units)}")
    for path in r1_units + r2_units:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("calibration_read") or manifest.get("formal_test_read"):
            raise RuntimeError(f"locked split access recorded by {path}")

    tracked = [
        "configs/v21_qesa.json",
        "configs/v21_qesa_gate_b.json",
        "configs/v21_qesa_r2.json",
        "configs/v21_qesa_gate_b_r2.json",
        "reports/METHOD1_V21_QESA_GENERALIZATION_RESULTS.md",
        "reports/METHOD1_V21_QESA_FAILURE_DIAGNOSIS.md",
        "reports/METHOD1_V21_QESA_REPRESENTATION_HEADROOM.md",
        "reports/METHOD1_V21_QESA_R2_PREREGISTRATION.md",
        "reports/METHOD1_V21_QESA_R2_GENERALIZATION_RESULTS.md",
        "aggregate/v21_qesa_oof_summary.json",
        "aggregate/v21_qesa_oof_metrics.csv",
        "aggregate/v21_qesa_failure_diagnosis_r1.json",
        "aggregate/v21_qesa_gold_prototype_headroom_r1.json",
        "aggregate/v21_qesa_gold_prototype_headroom_r1.csv",
        "aggregate/v21_qesa_oof_summary_r2.json",
        "aggregate/v21_qesa_oof_metrics_r2.csv",
    ]
    manifest = {
        "schema_version": "v21-local-manifest-qesa-gate-b-negative-1",
        "revision_id": "v21_qesa_gate_b_negative_r1_20260714",
        "parent_manifest": artifact("aggregate/v21_manifest_phase1_r1.json"),
        "source_control": "local immutable run IDs, manifests, and SHA-256; Git intentionally unused",
        "status": "gate_b_not_passed_qhds_promotion_blocked",
        "gates": {
            "A_fsa": "passed",
            "B_qesa": "not_passed_after_single_preregistered_corrective_revision",
            "C_qhds": "not_run_blocked_by_gate_b",
            "D_crc_ltt": "not_run_locked",
            "E_dynamic": "not_run_locked",
        },
        "qesa_r1": {
            "run_id": r1["run_id"],
            "units": len(r1_units),
            "status": r1["status"],
            "summary": artifact("aggregate/v21_qesa_oof_summary.json"),
            "metrics": artifact("aggregate/v21_qesa_oof_metrics.csv"),
        },
        "mechanism_diagnosis": {
            "r1_aspect_map_collapse_detected": True,
            "oracle_a_headroom_above_flat_B16": headroom["summary"]["16"]["effect_oracle_minus_flat"],
            "oracle_a_multi_group_headroom_above_flat_B16": headroom["summary"]["16"]["multi_group_effect_oracle_minus_flat"],
        },
        "qesa_r2": {
            "run_id": r2["run_id"],
            "units": len(r2_units),
            "status": r2["status"],
            "revision_limit_reached": True,
            "collapse_diagnostics": r2["collapse_diagnostics"],
            "summary": artifact("aggregate/v21_qesa_oof_summary_r2.json"),
            "metrics": artifact("aggregate/v21_qesa_oof_metrics_r2.csv"),
        },
        "scientific_decision": {
            "qesa_latent_association_signal_vs_prior_and_shuffle": "present",
            "qesa_generalization_vs_flat_cosine_and_same_capacity_mlp": "failed",
            "multi_group_compositional_gate": "failed_against_strong_scorers",
            "qhds_promotion": "prohibited",
            "further_qesa_tuning": "prohibited_by_r2_preregistration",
            "negative_results_retained": True,
        },
        "tracked_artifacts": [artifact(path) for path in tracked],
        "calibration": {"locked": True, "path_read": False, "access_count": 0},
        "formal_test": {"locked": True, "path_read": False, "runs": 0},
    }
    output = VERSION_ROOT / "aggregate" / "v21_manifest_qesa_gate_b_negative_r1.json"
    write_json_new(output, manifest)
    report = "# Method 1 v21 QESA Gate B Freeze\n\n"
    report += "Gate A remains passed. Gate B is frozen as not passed after the single preregistered mechanism-driven QESA correction. QHDS promotion and all downstream calibration/formal-test steps remain blocked.\n\n"
    report += "QESA r1 completed all 25 fixed-seed outer OOF units but underperformed the query-independent prior, flat cosine, and same-capacity MLP. A train-only diagnosis found collapsed aspect-to-prototype maps. Oracle A confirmed that the frozen incidence itself had positive gold-target headroom, which justified one corrective revision.\n\n"
    report += "QESA r2 removed the map collapse and learned a real association signal: it improved over the prior and shuffled supervision in pooled all-query and K>=2 analyses. It nevertheless remained below flat cosine by 0.112128 group coverage and below the same-capacity MLP by 0.039373 at B=16; all five seed effects were negative for both strong comparators. K>=2 effects were also negative.\n\n"
    report += "This is a learned-generalization failure, not a temporal-access claim, a co-access result, or evidence that the two acceptable members of a group must both be selected. K>=2 remains the multi-group/compositional subset.\n\n"
    report += "The r1 and r2 artifacts, failed Gate results, Oracle A headroom, and diagnosis are retained under distinct run IDs with SHA-256 manifests. No QESA r3 is allowed. Calibration access count and formal-test runs remain zero.\n"
    report_path = VERSION_ROOT / "reports" / "METHOD1_V21_QESA_GATE_B_FREEZE.md"
    with report_path.open("x", encoding="utf-8") as handle:
        handle.write(report)
    print(json.dumps({"status": manifest["status"], "manifest": str(output.relative_to(VERSION_ROOT)), "tracked_artifacts": len(tracked)}, sort_keys=True))


if __name__ == "__main__":
    main()
