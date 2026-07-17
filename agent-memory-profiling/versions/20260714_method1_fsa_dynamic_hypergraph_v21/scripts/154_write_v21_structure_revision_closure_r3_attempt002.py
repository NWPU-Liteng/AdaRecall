#!/usr/bin/env python3
"""Update the R3 closure boundary after persona-shard overlay integration."""
from __future__ import annotations

import json
import sys
from pathlib import Path

VERSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VERSION_ROOT / "src"))
from v21.io import read_json, sha256_file, write_json_new

PREVIOUS = VERSION_ROOT / "aggregate/v21_method1_structure_revision_closure_r3_attempt001.json"
SELECTOR = VERSION_ROOT / "aggregate/v21_structure_aware_selector_summary_r1_attempt004.json"
OVERLAY = VERSION_ROOT / "src/v21/incremental_sparse_overlay.py"
OVERLAY_TEST = VERSION_ROOT / "tests/test_v21_incremental_sparse_overlay.py"
OUTPUT = VERSION_ROOT / "aggregate/v21_method1_structure_revision_closure_r3_attempt002.json"
REPORT = VERSION_ROOT / "reports/METHOD1_V21_STRUCTURE_REVISION_CLOSURE_R3_ATTEMPT002_20260715.md"


def main() -> None:
    previous = read_json(PREVIOUS)
    selector = read_json(SELECTOR)
    if previous.get("status") != "completed_25_of_25_oof_structure_revision_negative_selector_result":
        raise ValueError("previous R3 closure is not complete")
    closure = dict(previous)
    closure["revision_id"] = "v21_method1_structure_revision_closure_r3_attempt002_20260715"
    closure["completion_scope"] = dict(previous["completion_scope"])
    closure["completion_scope"]["persona_sharded_incremental_publication"] = "implemented_and_unit_tested; full_oof_stress_not_run"
    closure["overlay_integration"] = {
        "status": "implemented_persona_by_prototype_append_delete_snapshot_compaction",
        "unit_tests_passed": 6,
        "causal_dynamic_access_update": "not_eligible_from_source_data",
        "static_content_update_oof_stress": "not_run_for_persona_shard_revision",
    }
    closure["hashes"] = dict(previous["hashes"])
    closure["hashes"]["previous_r3"] = sha256_file(PREVIOUS)
    closure["hashes"]["selector_summary"] = sha256_file(SELECTOR)
    closure["hashes"]["persona_sharded_overlay_source"] = sha256_file(OVERLAY)
    closure["hashes"]["persona_sharded_overlay_tests"] = sha256_file(OVERLAY_TEST)
    write_json_new(OUTPUT, closure)
    qhds = selector["pooled"]["existing_qhds|B=32"]
    structure = selector["pooled"]["structure_aware_selector|B=32"]
    report = f"""# Method 1 v21 Structure Revision Closure R3 Attempt 002

Retained semantic-bundle provenance remains complete for 25/25 OOF units and the structure-aware selector remains a retained negative result. Existing QHDS remains deployed: B32 group recall `{qhds['group_recall_mean']:.6f}` versus structure-aware `{structure['group_recall_mean']:.6f}`.

The persona-by-prototype incremental overlay is now integrated at the code level. It supports append/upsert, tombstones, immutable snapshots, monotonic atomic publication, persona-local rank merging, and offline compaction. Six focused tests pass, including cross-persona eligibility rejection and compaction lookup equivalence.

This does not claim a persona-sharded dynamic OOF stress result, a causal access/update stream, or PIM hardware behavior. QESA/candidate discovery remains the next quality revision; Agent answer execution remains unevaluated.

Calibration and formal test were not read. No minimum-effect threshold is used.
"""
    with REPORT.open("x", encoding="utf-8") as handle:
        handle.write(report)
    print(json.dumps({"status": closure["status"], "manifest": str(OUTPUT), "report": str(REPORT)}, sort_keys=True))


if __name__ == "__main__":
    main()
