#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path


VERSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VERSION_ROOT / "src"))

from v21.io import sha256_file, write_json_new  # noqa: E402


def record(path: str) -> dict:
    item = VERSION_ROOT / path
    return {"path": path, "sha256": sha256_file(item), "bytes": item.stat().st_size}


def main() -> None:
    result = json.loads((VERSION_ROOT / "aggregate" / "v21_qesa_gate_b2_summary.json").read_text(encoding="utf-8"))
    if result["status"] != "passed" or result.get("calibration_read") or result.get("formal_test_read"):
        raise RuntimeError("Gate B2 is not eligible for promotion")
    manifest = {
        "schema_version": "v21-local-manifest-qesa-gate-b2-passed-1",
        "revision_id": "v21_qesa_gate_b2_passed_r1_20260714",
        "parent_manifest": record("aggregate/v21_manifest_qesa_gate_b_negative_r1.json"),
        "status": "gate_b2_representation_validity_passed_qhds_development_permitted",
        "original_gate_b": "not_passed_and_retained",
        "protocol_amendment": record("reports/METHOD1_V21_QESA_GATE_PROTOCOL_AMENDMENT.md"),
        "gate_b2_config": record("configs/v21_qesa_gate_b2.json"),
        "gate_b2_result": record("aggregate/v21_qesa_gate_b2_summary.json"),
        "gate_b2_metrics": record("aggregate/v21_qesa_gate_b2_representation_metrics.csv"),
        "gate_b2_report": record("reports/METHOD1_V21_QESA_GATE_B2_RESULTS.md"),
        "source_qesa_run_id": result["source_run_id"],
        "role_boundary": result["role_boundary"],
        "gates": {
            "A_fsa": "passed",
            "B_original_retrieval": "not_passed_retained",
            "B2_qesa_representation": "passed",
            "C_qhds_attribution": "not_run",
            "D_crc_ltt": "locked",
            "E_dynamic": "not_run"
        },
        "calibration": {"locked": True, "path_read": False, "access_count": 0},
        "formal_test": {"locked": True, "path_read": False, "runs": 0}
    }
    output = VERSION_ROOT / "aggregate" / "v21_manifest_qesa_gate_b2_passed_r1.json"
    write_json_new(output, manifest)
    print(json.dumps({"status": manifest["status"], "manifest": str(output.relative_to(VERSION_ROOT))}, sort_keys=True))


if __name__ == "__main__":
    main()
