#!/usr/bin/env python3
"""Close the retained-bundle and structure-aware selector revision."""
from __future__ import annotations

import json
import sys
from pathlib import Path

VERSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VERSION_ROOT / "src"))
from v21.io import read_json, sha256_file, write_json_new  # noqa: E402

R2 = VERSION_ROOT / "aggregate/v21_method1_complete_closure_r2_attempt001.json"
SELECTOR = VERSION_ROOT / "aggregate/v21_structure_aware_selector_summary_r1_attempt004.json"
OUTPUT = VERSION_ROOT / "aggregate/v21_method1_structure_revision_closure_r3_attempt001.json"
REPORT = VERSION_ROOT / "reports/METHOD1_V21_STRUCTURE_REVISION_CLOSURE_R3_20260715.md"
PROVENANCE_ROOT = VERSION_ROOT / "artifacts/v21_sparse_frontier_provenance/v21_sparse_frontier_provenance_oof_r1_20260715"


def main() -> None:
    r2 = read_json(R2)
    selector = read_json(SELECTOR)
    if selector.get("status") != "completed_25_of_25_oof" or selector.get("calibration_read") is not False or selector.get("formal_test_read") is not False or selector.get("minimum_effect_threshold") is not None:
        raise ValueError("selector summary violates closure contract")
    manifests, provenance_hashes = [], {}
    for manifest_path in sorted(PROVENANCE_ROOT.glob("seed_*/fold_*/attempts/attempt_001/v21_sparse_frontier_provenance_manifest.json")):
        manifest = read_json(manifest_path)
        provenance_path = VERSION_ROOT / manifest["artifacts"]["provenance"]
        if manifest.get("status") != "completed" or manifest.get("exact_frontier_equivalence") is not True:
            raise ValueError(f"invalid provenance manifest {manifest_path}")
        if manifest.get("calibration_read") is not False or manifest.get("formal_test_read") is not False or manifest.get("minimum_effect_threshold") is not None:
            raise ValueError(f"provenance contract violation {manifest_path}")
        if sha256_file(provenance_path) != manifest["hashes"]["provenance"]:
            raise ValueError(f"provenance hash mismatch {provenance_path}")
        key = f"seed_{manifest['seed']}_fold_{manifest['fold']}"
        provenance_hashes[key] = sha256_file(provenance_path)
        manifests.append(manifest)
    if len(manifests) != 25:
        raise ValueError("expected 25 provenance manifests")
    qhds = selector["pooled"]["existing_qhds|B=32"]
    structure = selector["pooled"]["structure_aware_selector|B=32"]
    closure = {
        "schema_version": "v21-method1-structure-revision-closure-r3-1",
        "status": "completed_25_of_25_oof_structure_revision_negative_selector_result",
        "revision_id": "v21_method1_structure_revision_closure_r3_20260715",
        "retained_bundle_provenance": {
            "status": "completed_25_of_25_exact_frontier_equivalent",
            "queries": sum(int(manifest["queries"]) for manifest in manifests),
            "retained_semantic_bundles": sum(int(manifest["retained_semantic_bundles"]) for manifest in manifests),
            "returned_semantic_edges": sum(int(manifest["returned_semantic_edges"]) for manifest in manifests),
            "frontier_equivalence_all_units": all(manifest["exact_frontier_equivalence"] is True for manifest in manifests),
        },
        "selector_decision": {
            "promoted": False,
            "deployed_selector_remains": "existing_qhds",
            "reason": "structure-aware selector reduced OOF quality and increased selector latency",
            "existing_qhds_B32": qhds,
            "structure_aware_B32": structure,
            "delta_group_recall": structure["group_recall_mean"] - qhds["group_recall_mean"],
            "delta_exact_task_success": structure["all_groups_covered_mean"] - qhds["all_groups_covered_mean"],
            "delta_member_recall": structure["evidence_member_recall_mean"] - qhds["evidence_member_recall_mean"],
        },
        "next_primary_optimization": "improve QESA/candidate discovery; cap-96 candidate gap remains upstream of selector",
        "unchanged_r2_deployment_path": True,
        "completion_scope": {
            "persona_sharded_incremental_publication": "not_yet_integrated",
            "agent_answer_execution": "not_evaluated",
            "causal_dynamic_access_update": "not_eligible_from_source_data",
            "formal_heldout_test": "not_reportable_and_not_rerun",
        },
        "minimum_effect_threshold": None,
        "calibration_read": False,
        "formal_test_read": False,
        "hashes": {"r2_closure": sha256_file(R2), "selector_summary": sha256_file(SELECTOR), **{f"provenance_{key}": value for key, value in provenance_hashes.items()}},
    }
    write_json_new(OUTPUT, closure)
    report = f"""# Method 1 v21 Structure Revision Closure R3

## Retained provenance

The revision deterministically rebuilt the frozen cap-96 frontier and persisted true retained semantic-bundle membership and provenance for all 25 OOF units. All frontiers match the prior fixed frontier exactly.

| item | result |
|---|---:|
| OOF queries | {closure['retained_bundle_provenance']['queries']:,} |
| retained semantic bundles | {closure['retained_bundle_provenance']['retained_semantic_bundles']:,} |
| returned semantic edges | {closure['retained_bundle_provenance']['returned_semantic_edges']:,} |
| exact frontier equivalence | true |

## Structure-aware selector

The selector uses separate unary and stateful structural branches. Its greedy state includes bundle completion/progress, prototype coverage novelty, and embedding diversity. It was trained only on outer-train queries and evaluated on 25 target folds.

| B=32 selector | group recall | member recall | exact success | selector ms |
|---|---:|---:|---:|---:|
| existing QHDS | {qhds['group_recall_mean']:.6f} | {qhds['evidence_member_recall_mean']:.6f} | {qhds['all_groups_covered_mean']:.6f} | {qhds['selector_ms_mean']:.3f} |
| structure-aware revision | {structure['group_recall_mean']:.6f} | {structure['evidence_member_recall_mean']:.6f} | {structure['all_groups_covered_mean']:.6f} | {structure['selector_ms_mean']:.3f} |

This is a retained negative result. The structure-aware selector is not promoted, and existing QHDS remains the selector on the R2 deployment path. The next quality revision targets QESA/candidate discovery because the remaining cap-96 candidate gap is upstream of selection.

No minimum-effect threshold is used. Calibration and formal test were not read. Persona-sharded incremental publication is not integrated, Agent answer execution is not evaluated, and host/logical Tile results remain non-PIM measurements.
"""
    with REPORT.open("x", encoding="utf-8") as handle:
        handle.write(report)
    print(json.dumps({"status": closure["status"], "manifest": str(OUTPUT), "report": str(REPORT)}, sort_keys=True))


if __name__ == "__main__":
    main()
