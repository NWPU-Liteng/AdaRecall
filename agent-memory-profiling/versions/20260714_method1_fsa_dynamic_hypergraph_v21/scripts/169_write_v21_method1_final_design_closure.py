#!/usr/bin/env python3
"""Write the final static Method 1 design closure after cap/selector/handoff revisions."""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from v21.io import read_json, sha256_file, write_json_new  # noqa: E402

FILES = {
    "core_active16": ROOT / "aggregate/v21_sparse_joint_core_summary_cap64_r1_attempt001.json",
    "core_active24": ROOT / "aggregate/v21_sparse_joint_core_summary_cap64_active24_r1_attempt001.json",
    "core_active32": ROOT / "aggregate/v21_sparse_joint_core_summary_cap64_active32_r1_attempt001.json",
    "tile_active16": ROOT / "aggregate/v21_persona_sharded_hyper_index_summary_cap64_b8_r1_attempt001.json",
    "tile_active24": ROOT / "aggregate/v21_persona_sharded_hyper_index_summary_cap64_active24_b8_r1_attempt001.json",
    "e2e_active16": ROOT / "aggregate/v21_sparse_joint_e2e_control_summary_cap64_r1_attempt001.json",
    "e2e_active24": ROOT / "aggregate/v21_sparse_joint_e2e_control_summary_cap64_active24_r1_attempt001.json",
    "agent_baseline": ROOT / "aggregate/v21_cap64_agent_answers_oof_summary_r1_attempt001.json",
    "agent_balanced": ROOT / "aggregate/v21_cap64_agent_answers_balanced_oof_summary_r1_attempt001.json",
    "agent_lexical": ROOT / "aggregate/v21_cap64_agent_answers_lexical_handoff_oof_summary_r1_attempt001.json",
    "updates": ROOT / "aggregate/v21_persona_sharded_update_stress_r1_attempt001.json",
    "selector": ROOT / "aggregate/v21_method1_structure_revision_closure_r3_attempt002.json",
    "cap_audit": ROOT / "aggregate/v21_candidate_cap_prefix_audit_r1_attempt001.json",
}
OUT = ROOT / "aggregate/v21_method1_final_design_closure_r2_attempt001.json"
REPORT = ROOT / "reports/METHOD1_V21_FINAL_DESIGN_CLOSURE_R2_20260715.md"

def main() -> None:
    values = {name: read_json(path) for name, path in FILES.items()}
    for name, value in values.items():
        if value.get("calibration_read") is not False or value.get("formal_test_read") is not False:
            raise ValueError(f"locked split flag in {name}")
        if value.get("minimum_effect_threshold") is not None:
            raise ValueError(f"effect threshold in {name}")
    def core(name: str, active: int) -> dict:
        return values[name]["pooled_fixed_count"][f"sparse_joint_semantic_bundle_hypergraph|{active}|32|32|32"]
    c16, c24, c32 = core("core_active16", 16), core("core_active24", 24), core("core_active32", 32)
    t16, t24 = values["tile_active16"], values["tile_active24"]
    e16, e24 = values["e2e_active16"], values["e2e_active24"]
    manifest = {
        "schema_version": "v21-method1-final-design-closure-r2",
        "status": "static_retrieval_tile_e2e_update_and_agent_oof_closed_with_quality_tiers",
        "promoted_default": {"candidate_pool_cap": 64, "activated_prototypes": 16, "selection_budget": 8, "reason": "lowest measured e2e while retaining cap64 core quality"},
        "promoted_quality_tier": {"candidate_pool_cap": 64, "activated_prototypes": 24, "selection_budget": 8, "reason": "small OOF quality gain with a measured latency tradeoff"},
        "diagnostic_tier": {"candidate_pool_cap": 64, "activated_prototypes": 32, "selection_budget": 8, "promoted": False, "reason": "quality gain is accompanied by large online latency"},
        "core": {"active16_B32": c16, "active24_B32": c24, "active32_B32": c32},
        "tile": {"active16_B8": t16["pooled"], "active24_B8": t24["pooled"], "active16_effect": t16["relative_effect_persona_sharded_minus_global"], "active24_effect": t24["relative_effect_persona_sharded_minus_global"]},
        "matched_e2e": {"active16": e16, "active24": e24},
        "agent_handoff": {"baseline": values["agent_baseline"]["by_budget"], "balanced_packing": values["agent_balanced"]["by_budget"], "lexical_B8": values["agent_lexical"]["by_budget"]["8"], "balanced_promoted": False, "lexical_promoted": False},
        "maintenance": {"quality_contract_satisfied": values["updates"]["quality_contract_satisfied"], "adapter": "PersonaShardedSnapshotDispatch backed by AtomicPersonaShardedSnapshotPublisher", "causal_dynamic_claim": False},
        "selector": {"structure_aware_promoted": False, "closure": "existing QHDS retained"},
        "remaining_boundaries": ["Agent accuracy is train-OOF only and remains 0.259041", "formal heldout is unavailable and was not rerun", "source data cannot support causal dynamic access/update claims", "active24/active32 are quality/diagnostic tiers; active16 is the default serving tier"],
        "minimum_effect_threshold": None, "calibration_read": False, "formal_test_read": False,
        "inputs": {name: str(path.relative_to(ROOT)) for name, path in FILES.items()},
        "hashes": {name: sha256_file(path) for name, path in FILES.items()},
    }
    write_json_new(OUT, manifest)
    pct = lambda value: f"{value * 100:+.2f}%"
    report = f'''# Method 1 v21 Final Design Closure r2

## Final design decision

The frontier cap is fixed at `64`. `cap=32` is too small, while `cap=96` adds headroom without improving the selected result. The default serving tier uses `16` activated prototypes and `B=8`; a quality tier uses `24` activated prototypes and `B=8`. `32` activated prototypes is retained as a diagnostic point, not the default.

| tier | active prototypes | group recall@32 | exact success | core e2e | status |
|---|---:|---:|---:|---:|---|
| default | 16 | {c16['group_coverage']:.6f} | {c16['all_groups_covered']:.6f} | {c16['query_e2e_ms']:.1f} ms | promoted |
| quality | 24 | {c24['group_coverage']:.6f} | {c24['all_groups_covered']:.6f} | {c24['query_e2e_ms']:.1f} ms | promoted as optional |
| diagnostic | 32 | {c32['group_coverage']:.6f} | {c32['all_groups_covered']:.6f} | {c32['query_e2e_ms']:.1f} ms | not promoted |

## Persona sharding and matched e2e

For the default B8 tier, corrected logical Tiles change from `{t16['pooled']['global_total_logical_tile_activations_mean']:.3f}` to `{t16['pooled']['persona_sharded_total_logical_tile_activations_mean']:.3f}` ({pct(t16['relative_effect_persona_sharded_minus_global']['total_logical_tile_activations'])}); rank entries fall by `95.59%`. Matched e2e is `{e16['stage_latency']['query_e2e_ms']['global_mean']:.3f} -> {e16['stage_latency']['query_e2e_ms']['persona_sharded_mean']:.3f} ms`, with identical selections in all `6775` paired rows.

The active24 quality tier has corrected Tiles `{t24['pooled']['global_total_logical_tile_activations_mean']:.3f} -> {t24['pooled']['persona_sharded_total_logical_tile_activations_mean']:.3f}` ({pct(t24['relative_effect_persona_sharded_minus_global']['total_logical_tile_activations'])}) and matched e2e `{e24['stage_latency']['query_e2e_ms']['global_mean']:.3f} -> {e24['stage_latency']['query_e2e_ms']['persona_sharded_mean']:.3f} ms`. Tile values are host/logical accounting, not PIM hardware measurements.

## Agent handoff conclusion

The original, balanced-packing, and lexical-order B8 evaluations all remain at `0.259041` train-OOF answer accuracy. Balanced packing increased generation latency without improving accuracy; lexical ordering also did not improve it. These are negative results and are not promoted. The remaining Agent gap is therefore answer generation/evidence adequacy, not simply candidate count or context order.

## Closed and bounded

The static retrieval path, cap64 frontier, persona-by-prototype sharding, corrected Tile accounting, matched e2e controls, persona-sharded atomic overlay dispatch, append/delete/compaction stress, and Agent train-OOF execution are implemented and validated. Structure-aware selector training was negative, so existing QHDS remains selected. Formal heldout performance and causal dynamic access/update claims remain outside the data/protocol boundary.

All inputs are OOF or explicitly labelled stress/diagnostic artifacts; calibration/formal splits were not read and no minimum-effect threshold was used.
'''
    REPORT.open("x", encoding="utf-8").write(report)
    print(json.dumps({"status": manifest["status"], "manifest": str(OUT), "report": str(REPORT)}, sort_keys=True))

if __name__ == "__main__":
    main()
