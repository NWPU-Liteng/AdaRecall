#!/usr/bin/env python3
"""Write the revised Method 1 closure after Tile co-design experiments."""
from __future__ import annotations

import json
import sys
from pathlib import Path


VERSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VERSION_ROOT / "src"))

from v21.io import read_json, sha256_file, write_json_new  # noqa: E402


INPUTS = {
    "core_kpi": VERSION_ROOT / "aggregate/v21_method1_core_kpi_closure_r1_attempt001.json",
    "tile_index": VERSION_ROOT / "aggregate/v21_persona_sharded_hyper_index_summary_r2_attempt001.json",
    "selector_refit": VERSION_ROOT / "aggregate/v21_sparse_frontier_selector_refit_summary_r1_attempt001.json",
    "frontier_headroom": VERSION_ROOT / "aggregate/v21_sparse_frontier_quality_headroom_r1_attempt001.json",
    "e2e_control": VERSION_ROOT / "aggregate/v21_sparse_joint_e2e_control_summary_r1_attempt001.json",
    "static_update": VERSION_ROOT / "aggregate/v21_static_content_update_stress_r1_attempt002.json",
    "legacy_layout_headroom": VERSION_ROOT / "aggregate/v21_legacy_hyper_layout_headroom_r1_attempt001.json",
}
OUTPUT = VERSION_ROOT / "aggregate/v21_method1_complete_closure_r2_attempt001.json"
REPORT = VERSION_ROOT / "reports/METHOD1_V21_COMPLETE_CLOSED_LOOP_R2_20260715.md"


def main() -> None:
    values = {name: read_json(path) for name, path in INPUTS.items()}
    for name, value in values.items():
        if value.get("calibration_read") is not False or value.get("formal_test_read") is not False:
            raise ValueError(f"{name} touched a locked split")
        if value.get("minimum_effect_threshold", None) is not None:
            raise ValueError(f"{name} introduced an effect threshold")
    core = values["core_kpi"]["pooled"]["sparse_joint_semantic_bundle_hypergraph|B=32"]
    ann = values["core_kpi"]["pooled"]["ann_flat|B=32"]
    tile = values["tile_index"]
    refit = values["selector_refit"]
    headroom = values["frontier_headroom"]["pooled"]
    e2e = values["e2e_control"]["stage_latency"]
    update = values["static_update"]["results"][-1]
    refit_b32 = refit["pooled"]["sparse_frontier_refit_mlp|B=32"]
    manifest = {
        "schema_version": "v21-method1-complete-closure-r2",
        "status": "completed_train_oof_retrieval_tile_and_static_maintenance_loop",
        "revision_id": "v21_method1_complete_closure_r2_20260715",
        "primary_method": {
            "offline": "frozen embedding and UOT incidence; persona x prototype rank shards; QESA and existing QHDS outer-train models",
            "online": "protected ANN top-5 plus QESA-activated persona-sharded prototype edges; semantic bundle beam; cap-96 shared frontier; QHDS B=32 selection",
            "payload": "single-copy neutral capacity-16 payload layout",
            "update": "static incidence overlay, tombstone, atomic snapshot, and periodic compaction; persona-shard-aware publication remains to integrate",
        },
        "core_quality_B32": core,
        "same_frontier_ann_B32": ann,
        "tile_and_index": {
            "rank_entries_global": tile["pooled"]["global_rank_entries_visited_mean"],
            "rank_entries_persona_sharded": tile["pooled"]["persona_rank_entries_visited_mean"],
            "metadata_bytes_global": tile["pooled"]["global_metadata_bytes_accessed_mean"],
            "metadata_bytes_persona_sharded": tile["pooled"]["persona_metadata_bytes_accessed_mean"],
            "corrected_total_tiles_global": tile["pooled"]["global_total_logical_tile_activations_mean"],
            "corrected_total_tiles_persona_sharded": tile["pooled"]["persona_sharded_total_logical_tile_activations_mean"],
            "relative_total_tile_effect": tile["relative_effect_persona_sharded_minus_global"]["total_logical_tile_activations"],
            "standalone_lookup_global_ms": tile["pooled"]["global_lookup_ms_mean"],
            "standalone_lookup_persona_sharded_ms": tile["pooled"]["persona_sharded_lookup_ms_mean"],
            "storage_overhead_bytes": 0,
            "returned_members_identical": True,
        },
        "matched_integrated_e2e": e2e,
        "quality_diagnosis": {
            "frontier96_group_recall_upper_bound": headroom["frontier96_group_recall_upper_bound_mean"],
            "selector_headroom_inside_frontier": headroom["selector_group_recall_headroom_inside_frontier_mean"],
            "candidate_gap_to_perfect": headroom["candidate_group_recall_gap_to_perfect_mean"],
            "sparse_frontier_refit_B32_group_recall": refit_b32["group_recall_mean"],
            "sparse_frontier_refit_promoted": False,
            "reason": "fixed 4-feature refit is below both same-frontier ANN and existing QHDS",
        },
        "layout_diagnosis": {
            "legacy_all_train_selected_payload_tiles_B32": values["legacy_layout_headroom"]["pooled"]["legacy_all_train_hyper_diagnostic|B=32"]["selected_payload_tile_activations_mean"],
            "legacy_layout_deployable": False,
            "strict_oof_single_copy_static_relayout": "positive but small pilot; not promoted",
            "promoted_tile_mechanism": "persona x prototype rank sharding with exact lookup equivalence",
        },
        "static_update_20_percent": update,
        "completion_scope": {
            "offline_index_and_models": "implemented_and_train_oof_evaluated",
            "online_sparse_retrieval": "implemented_and_train_oof_evaluated",
            "hypergraph_aware_rank_layout": "implemented_and_25_unit_oof_evaluated",
            "matched_integrated_e2e": "implemented_and_25_unit_oof_evaluated",
            "static_content_maintenance": "implemented_and_stress_evaluated_on_global_overlay",
            "persona_sharded_incremental_publication": "not_yet_integrated",
            "agent_answer_execution": "not_evaluated",
            "formal_heldout_test": "not_reportable_missing_frozen_embeddings_and_not_rerun",
            "causal_dynamic_access_update": "not_eligible_from_source_data",
        },
        "next_primary_optimization": [
            "persist retained semantic-bundle incidence/provenance in the frontier artifact",
            "train a structure-aware selector with bundle completion and prototype-coverage marginal state",
            "improve QESA candidate discovery because frontier96 still misses 19.06 percent group recall",
            "integrate append/delete overlays per persona-prototype shard",
            "execute B=8/16/32 retrieval handoff through the Agent and score final answers",
        ],
        "minimum_effect_threshold": None,
        "calibration_read": False,
        "formal_test_read": False,
        "inputs": {name: str(path.relative_to(VERSION_ROOT)) for name, path in INPUTS.items()},
        "hashes": {name: sha256_file(path) for name, path in INPUTS.items()},
    }
    write_json_new(OUTPUT, manifest)
    e2e_total = e2e["query_e2e_ms"]
    report = f"""# Method 1 v21 Complete Closed Loop r2

This revision adds a corrected logical-Tile model, a persona-by-prototype sharded hyper-index, matched integrated e2e controls, sparse-frontier selector refit, and candidate-quality headroom analysis. All results are train OOF or explicitly labelled diagnostics. Calibration and formal test were not read.

## Promoted Method 1 path

```text
offline memory -> frozen embedding -> UOT fuzzy incidence
              -> persona x prototype rank shards

query embedding
    |-> protected ANN top-5
    `-> QESA active prototypes -> query-persona rank shards -> bounded edge members
                              -> arity-3 semantic bundle beam
                              -> shared cap-96 frontier
                              -> existing QHDS B=32 (quality-first)
                                 or cosine B=32 (latency-first)
                              -> single-copy payload gather -> Agent handoff
```

## Core quality

| path | group recall@32 | member recall@32 | exact task success | payload |
|---|---:|---:|---:|---:|
| persona-sharded frontier + QHDS | {core['group_recall_at_k_mean']:.6f} | {core['evidence_member_recall_at_k_mean']:.6f} | {core['exact_task_success_rate']:.6f} | {core['payload_bytes_mean']:.1f} B |
| same frontier + cosine | {ann['group_recall_at_k_mean']:.6f} | {ann['evidence_member_recall_at_k_mean']:.6f} | {ann['exact_task_success_rate']:.6f} | {ann['payload_bytes_mean']:.1f} B |

Sharding changes no returned member or selection. The post-closure cap-96 frontier remains responsible for the larger quality gain over immutable r3; QHDS adds only a small quality-first effect over same-frontier cosine.

## Corrected Tile advantage

| metric | global rank index | persona x prototype shards | change |
|---|---:|---:|---:|
| rank entries visited | {manifest['tile_and_index']['rank_entries_global']:.1f} | {manifest['tile_and_index']['rank_entries_persona_sharded']:.1f} | {(manifest['tile_and_index']['rank_entries_persona_sharded'] / manifest['tile_and_index']['rank_entries_global'] - 1) * 100:+.2f}% |
| metadata bytes accessed | {manifest['tile_and_index']['metadata_bytes_global']:.1f} | {manifest['tile_and_index']['metadata_bytes_persona_sharded']:.1f} | {(manifest['tile_and_index']['metadata_bytes_persona_sharded'] / manifest['tile_and_index']['metadata_bytes_global'] - 1) * 100:+.2f}% |
| corrected total logical Tiles | {manifest['tile_and_index']['corrected_total_tiles_global']:.3f} | {manifest['tile_and_index']['corrected_total_tiles_persona_sharded']:.3f} | {manifest['tile_and_index']['relative_total_tile_effect'] * 100:+.2f}% |
| standalone sparse lookup | {manifest['tile_and_index']['standalone_lookup_global_ms']:.3f} ms | {manifest['tile_and_index']['standalone_lookup_persona_sharded_ms']:.3f} ms | {(manifest['tile_and_index']['standalone_lookup_persona_sharded_ms'] / manifest['tile_and_index']['standalone_lookup_global_ms'] - 1) * 100:+.2f}% |

The old `52.61` mixed metric is retired and is not directly comparable with these corrected totals. The new metric separately accounts for actual 4 KiB rank-prefix blocks and capacity-16 memory feature/payload Tiles.

## Matched integrated e2e

The single-grid controls reuse identical model weights and produce identical selections. Global mean e2e is `{e2e_total['global_mean']:.3f} ms`; persona-sharded mean e2e is `{e2e_total['persona_sharded_mean']:.3f} ms`, paired delta `{e2e_total['delta_mean']:+.3f} ms`. Bundle construction and CPU selection now dominate the online path.

## Negative selector refit and remaining quality headroom

The fixed sparse-frontier MLP refit is not promoted: B=32 group recall is `{refit_b32['group_recall_mean']:.6f}`, below ANN `{ann['group_recall_at_k_mean']:.6f}` and QHDS `{core['group_recall_at_k_mean']:.6f}`. Nevertheless, the cap-96 frontier gold upper bound is `{headroom['frontier96_group_recall_upper_bound_mean']:.6f}`, leaving `{headroom['selector_group_recall_headroom_inside_frontier_mean']:.6f}` selector headroom and `{headroom['candidate_group_recall_gap_to_perfect_mean']:.6f}` candidate-discovery gap.

## What remains before a full method claim

- Persist real retained bundle membership/provenance so the selector can optimize bundle completion rather than four flat features.
- Integrate incremental append/delete publication into persona-prototype shards; the existing static overlay stress is global-index based.
- Run the selected B-memory contexts through the Agent and measure final answer accuracy, token cost, and total latency.
- Obtain a newly authorized independent heldout protocol; v21 formal test remains non-reportable and is not rerun.
- Use a dataset with genuine creation/read events for causal dynamic claims. Current data supports static content maintenance only.

No minimum-effect threshold is used. Host/logical Tile results are not PIM hardware results.
"""
    with REPORT.open("x", encoding="utf-8") as handle:
        handle.write(report)
    print(json.dumps({"status": manifest["status"], "manifest": str(OUTPUT), "report": str(REPORT)}, sort_keys=True))


if __name__ == "__main__":
    main()
