#!/usr/bin/env python3
"""Summarize all authorized v21 OOF, e2e, maintenance, and Agent validations."""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from v21.io import read_json, sha256_file, write_json_new  # noqa: E402

INPUTS = {
    "final_design_r3": ROOT / "aggregate/v21_method1_final_design_closure_r3_attempt001.json",
    "dynamic_partition": ROOT / "aggregate/v21_persona_sharded_partition_update_stress_r1_attempt001.json",
    "agent_baseline": ROOT / "aggregate/v21_cap64_agent_answers_oof_summary_r1_attempt001.json",
    "agent_balanced": ROOT / "aggregate/v21_cap64_agent_answers_balanced_oof_summary_r1_attempt001.json",
    "agent_lexical": ROOT / "aggregate/v21_cap64_agent_answers_lexical_handoff_oof_summary_r1_attempt001.json",
    "agent_evidence_b8": ROOT / "aggregate/v21_cap64_agent_answers_evidence_context_oof_summary_r1_attempt001.json",
    "agent_evidence_b32": ROOT / "aggregate/v21_cap64_agent_answers_evidence_context_b32_oof_summary_r1_attempt001.json",
}
OUT = ROOT / "aggregate/v21_authorized_full_validation_closure_r1_attempt001.json"
REPORT = ROOT / "reports/METHOD1_V21_AUTHORIZED_FULL_VALIDATION_CLOSURE_R1_20260715.md"

def main() -> None:
    values = {name: read_json(path) for name, path in INPUTS.items()}
    for name, value in values.items():
        if value.get("calibration_read") is not False or value.get("formal_test_read") is not False:
            raise ValueError(f"locked split flag in {name}")
        if value.get("minimum_effect_threshold") is not None:
            raise ValueError(f"threshold in {name}")
    d = values["dynamic_partition"]
    if not d["quality_contract_satisfied"] or any(row["dispatch_mismatch_count"] or row["overlay_compacted_mismatch_count"] or row["selection_mismatch_count"] for row in d["results"]):
        raise ValueError("dynamic partition contract failed")
    b = values["agent_baseline"]["by_budget"]
    manifest = {
        "schema_version": "v21-authorized-full-validation-closure-1",
        "status": "all_authorized_oof_e2e_tile_partition_update_and_agent_handoff_validations_completed",
        "validation_boundary": "25-unit train-OOF, 25-unit matched e2e, static partition update stress, and 1355 train-query Agent executions; calibration/formal splits forbidden",
        "retrieval_default": {"candidate_pool_cap": 64, "activated_prototypes": 16, "selection_budget": 8, "group_recall_at_32": 0.6876383763837642, "exact_success": 0.6140221402214022},
        "retrieval_quality_tier": {"candidate_pool_cap": 64, "activated_prototypes": 24, "selection_budget": 8, "group_recall_at_32": 0.690836408364084, "exact_success": 0.6177121771217712},
        "retrieval_diagnostic": {"activated_prototypes": 32, "group_recall_at_32": 0.694587945879459, "exact_success": 0.6209594095940959, "promoted": False},
        "matched_e2e_units": {"active16": 25, "active24": 25, "rows_each": 6775, "selections_identical": True},
        "dynamic_partition": {"base_memory_count": d["base_memory_count"], "new_memory_pool_count": d["new_memory_pool_count"], "growth_points": [row["partition_growth_percent"] for row in d["results"]], "all_contracts_zero_mismatch": True, "causal_claim": False},
        "agent_handoff_comparison": {"baseline_B8": b["8"], "baseline_B16": b["16"], "baseline_B32": b["32"], "balanced_B8": values["agent_balanced"]["by_budget"]["8"], "lexical_B8": values["agent_lexical"]["by_budget"]["8"], "evidence_B8": values["agent_evidence_b8"]["by_budget"]["8"], "evidence_B32": values["agent_evidence_b32"]["by_budget"]["32"], "promoted": "none; all variants answer_accuracy=0.259041"},
        "remaining_unclosed_claims": ["final Agent answer quality is not acceptable as a generalization guarantee", "formal heldout result is unavailable", "causal dynamic access/update behavior is not identifiable from source data"],
        "minimum_effect_threshold": None,
        "calibration_read": False,
        "formal_test_read": False,
        "inputs": {name: str(path.relative_to(ROOT)) for name, path in INPUTS.items()},
        "hashes": {name: sha256_file(path) for name, path in INPUTS.items()},
    }
    write_json_new(OUT, manifest)
    report = f'''# Method 1 v21 Authorized Full Validation Closure

## What was fully validated

All currently authorized validation surfaces are complete: 25-unit train-OOF retrieval for active16/24/32, 25-unit matched global/sharded e2e controls for active16/24, corrected Tile OOF, deterministic static new-memory partition publication at 1/5/10/20% growth, atomic snapshot serving, overlay/compaction equivalence, and `1355 x 3` baseline Agent executions plus B8/B32 handoff revisions.

The default remains `cap=64`, active16, B8. Active24 is an optional quality tier. Active32 is diagnostic because its latency cost is large.

## Agent result

Baseline B8/B16/B32, balanced packing B8, lexical handoff B8, structured evidence B8, and structured evidence B32 all produced answer accuracy `0.259041` under the registered train-OOF scorer. The handoff changes did not improve accuracy. This is a negative result, not a reason to claim that the final Agent effect is closed.

## Boundary

The retrieval/index/Tile/static-publication engineering loop is closed. A full scientific claim of end-to-end Agent effectiveness is not possible yet: the authorized Agent score is low, calibration/formal data are locked, and the source data do not identify causal dynamic behavior. No threshold was used to hide small effects, and no locked split was read.

Next work must change the answer-generation/evaluation protocol or model, then repeat the same full train-OOF and newly authorized heldout validation. Simply increasing candidate count has been tested and is not sufficient.
'''
    REPORT.write_text(report, encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "manifest": str(OUT), "report": str(REPORT)}, sort_keys=True))

if __name__ == "__main__":
    main()
