#!/usr/bin/env python3
"""Write the complete static Method 1 closure from immutable component results."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


VERSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VERSION_ROOT / "src"))

from v21.io import read_json, sha256_file, write_json_new  # noqa: E402


KPI_SUMMARY = VERSION_ROOT / "aggregate/v21_method1_core_kpi_closure_r1_attempt001.json"
UPDATE_SUMMARY = VERSION_ROOT / "aggregate/v21_static_content_update_stress_r1_attempt002.json"
R3_FIXED = VERSION_ROOT / "aggregate/v21_sparse_joint_fixed_count_r3_attempt019.csv"
TUNED_FIXED = VERSION_ROOT / "aggregate/v21_sparse_joint_fixed_count_tuned_r2_attempt001.csv"
OUTPUT = VERSION_ROOT / "aggregate/v21_method1_complete_closure_r1_attempt001.json"
REPORT = VERSION_ROOT / "reports/METHOD1_V21_COMPLETE_CLOSED_LOOP_R1_20260715.md"
METHOD = "sparse_joint_semantic_bundle_hypergraph"


def fixed_point(path: Path) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (
                row["method"] == METHOD
                and int(row["activated_prototypes"]) == 16
                and int(row["members_per_semantic_edge"]) == 32
                and int(row["semantic_bundle_beam"]) == 32
                and int(row["budget"]) == 32
            ):
                return {key: float(value) for key, value in row.items() if key not in {"method"} and value != ""}
    raise ValueError(f"deployment point missing from {path}")


def require_locked(summary: dict, name: str) -> None:
    if summary.get("calibration_read") is not False or summary.get("formal_test_read") is not False:
        raise ValueError(f"{name} touched a locked split")


def main() -> None:
    kpi = read_json(KPI_SUMMARY)
    update = read_json(UPDATE_SUMMARY)
    require_locked(kpi, "KPI closure")
    require_locked(update, "static update stress")
    if kpi.get("minimum_effect_threshold") is not None or update.get("minimum_effect_threshold") is not None:
        raise ValueError("complete closure cannot introduce an effect threshold")
    if update.get("quality_contract_satisfied") is not True:
        raise ValueError("static overlay did not reproduce compacted lookup")
    r3 = fixed_point(R3_FIXED)
    tuned = fixed_point(TUNED_FIXED)
    primary = kpi["pooled"][f"{METHOD}|B=32"]
    ann = kpi["pooled"]["ann_flat|B=32"]
    selector_delta = next(
        row for row in kpi["comparisons"]
        if int(row["budget"]) == 32 and row["comparator"] == "ann_flat"
    )
    manifest = {
        "schema_version": "v21-method1-complete-closure-1",
        "status": "completed_static_oof_and_content_maintenance_closed_loop",
        "revision_id": "v21_method1_complete_closure_r1_20260715",
        "method_definition": {
            "offline": [
                "frozen memory embedding",
                "UOT fuzzy semantic incidence over 64 prototypes",
                "per-prototype sparse memory rank lists",
                "neutral capacity-16 logical Tile layout",
                "outer-train QESA and QHDS fitting",
            ],
            "online": [
                "precomputed query embedding",
                "five protected ANN seeds",
                "QESA top activated prototypes",
                "eligible-only bounded sparse rank-prefix lookup",
                "query-conditioned arity-3 semantic bundle beam",
                "shared frontier: protected seeds union bundle members union returned-edge fallback, capped at 96",
                "QHDS greedy nested selection through B=32",
                "selected B-memory retrieval handoff to the Agent",
            ],
            "content_update": [
                "frozen embedding/projection and online prototype incidence assignment",
                "append incidence to sparse overlay and append-only logical Tile overlay",
                "tombstone deletion",
                "atomic versioned snapshot publication",
                "periodic offline compaction into new rank lists",
            ],
        },
        "important_boundaries": {
            "offline_materialized_structure": "prototype incidence and rank lists",
            "query_conditioned_structure": "semantic bundle tuples and candidate frontier",
            "bundle_locality_deepsets_primary": False,
            "layout": "frozen neutral host/logical layout",
            "rerank_executed": False,
            "agent_answer_accuracy_evaluated": False,
            "true_causal_dynamic_stream_evaluated": False,
            "pim_hardware_evaluated": False,
        },
        "core_kpis_at_16_32_32_B32": primary,
        "same_frontier_ann_at_16_32_32_B32": ann,
        "postclosure_revision_delta_tuned_r2_minus_r3": {
            "group_recall": tuned["group_coverage_mean"] - r3["group_coverage_mean"],
            "exact_task_success": tuned["all_groups_covered_mean"] - r3["all_groups_covered_mean"],
            "payload_bytes": tuned["payload_bytes_mean"] - r3["payload_bytes_mean"],
            "logical_tile_activations": tuned["total_unique_tile_activations_mean"] - r3["total_unique_tile_activations_mean"],
            "query_e2e_ms": tuned["query_e2e_ms_mean"] - r3["query_e2e_ms_mean"],
            "interpretation": "mainly expanded-frontier revision effect; not isolated selector effect",
        },
        "selector_delta_hypergraph_minus_same_frontier_ann_B32": selector_delta,
        "static_content_update_stress": update["results"],
        "operating_points": {
            "quality_first": "expanded sparse frontier plus QHDS, B=32",
            "latency_first": "the same expanded sparse frontier plus cosine ordering, B=32",
            "selection_rule": "choose by the deployment objective; no minimum-effect threshold or method-validity gate is applied",
        },
        "completion_scope": {
            "offline_training_and_index": "implemented_and_oof_evaluated",
            "online_sparse_retrieval": "implemented_and_oof_evaluated",
            "core_quality_latency_payload_logical_tile": "reported",
            "static_content_update_overlay": "implemented_and_stress_evaluated",
            "agent_answer_execution": "interface_boundary_only_not_evaluated",
            "causal_dynamic_access_update": "not_eligible_from_source_data",
            "formal_test": "prior_once_attempt_ineligible_missing_embeddings_not_rerun",
        },
        "minimum_effect_threshold": None,
        "threshold_based_method_failure": False,
        "calibration_read": False,
        "formal_test_read": False,
        "inputs": {
            "kpi_summary": str(KPI_SUMMARY.relative_to(VERSION_ROOT)),
            "update_summary": str(UPDATE_SUMMARY.relative_to(VERSION_ROOT)),
            "immutable_r3_fixed_count": str(R3_FIXED.relative_to(VERSION_ROOT)),
            "tuned_r2_fixed_count": str(TUNED_FIXED.relative_to(VERSION_ROOT)),
        },
        "hashes": {
            "kpi_summary": sha256_file(KPI_SUMMARY),
            "update_summary": sha256_file(UPDATE_SUMMARY),
            "immutable_r3_fixed_count": sha256_file(R3_FIXED),
            "tuned_r2_fixed_count": sha256_file(TUNED_FIXED),
        },
    }
    write_json_new(OUTPUT, manifest)

    update_last = update["results"][-1]
    report = f"""# Method 1 v21 Complete Closed Loop

This document closes the static Method 1 system boundary without replacing the immutable final freeze. It combines tuned-r2 train OOF retrieval, threshold-free core KPIs, and the permitted static content-update stress. Calibration and formal test were not read.

## Correct system model

```text
OFFLINE
memory content -> embedding -> UOT fuzzy incidence (42,034 x 64)
                              -> 64 sparse prototype rank lists
memory payload ----------------> neutral capacity-16 logical Tile map
outer-train queries -----------> QESA + QHDS fitting

ONLINE, batch=1
precomputed query embedding
    |-> protected ANN top-5 --------------------------|
    `-> QESA -> active prototypes -> bounded lookup  |
                    -> arity-3 semantic bundle beam  |
                                                     v
          shared frontier (cap=96, no dense scan)
                    -> QHDS greedy order -> B=8/16/32
                    -> payload gather -> Agent handoff

STATIC CONTENT MAINTENANCE FALLBACK
new content -> frozen embedding/projection -> prototype incidence
            -> sparse overlay + append-only logical Tile
            -> atomic snapshot publication -> online lookup
delete ----> tombstone
periodic --> offline compaction -> new frozen rank-list base
```

The persistent offline structure is fuzzy prototype incidence plus rank lists. Arity-3 semantic bundles are query-conditioned candidate tuples built online; they are not a fabricated co-access hypergraph. The negative DeepSets bundle-locality result is not in the primary path, and the layout remains neutral.

## Core result at 16 activated prototypes / 32 members / beam 32 / B=32

| path | group recall | member recall | exact task success | mean e2e | p95 e2e | payload | logical Tiles |
|---|---:|---:|---:|---:|---:|---:|---:|
| expanded frontier + QHDS | {primary['group_recall_at_k_mean']:.6f} | {primary['evidence_member_recall_at_k_mean']:.6f} | {primary['exact_task_success_rate']:.6f} | {primary['query_e2e_ms_mean']:.3f} ms | {primary['query_e2e_ms_p95']:.3f} ms | {primary['payload_bytes_mean']:.1f} B | {primary['logical_tile_activations_mean']:.3f} |
| same frontier + cosine | {ann['group_recall_at_k_mean']:.6f} | {ann['evidence_member_recall_at_k_mean']:.6f} | {ann['exact_task_success_rate']:.6f} | {ann['query_e2e_ms_mean']:.3f} ms | {ann['query_e2e_ms_p95']:.3f} ms | {ann['payload_bytes_mean']:.1f} B | {ann['logical_tile_activations_mean']:.3f} |

Compared with immutable r3, tuned-r2 changes group recall by `{manifest['postclosure_revision_delta_tuned_r2_minus_r3']['group_recall']:+.6f}` and exact task success by `{manifest['postclosure_revision_delta_tuned_r2_minus_r3']['exact_task_success']:+.6f}`. This is mainly the cap-32 to cap-96 frontier revision effect.

Within the exact same cap-96 frontier, QHDS versus cosine changes group recall by `{selector_delta['delta_group_recall_at_k_mean']:+.6f}`, member recall by `{selector_delta['delta_evidence_member_recall_at_k_mean']:+.6f}`, exact task success by `{selector_delta['delta_exact_task_success_rate']:+.6f}`, mean e2e by `{selector_delta['delta_query_e2e_ms_mean']:+.3f} ms`, payload by `{selector_delta['delta_payload_bytes_mean']:+.1f} B`, and logical Tiles by `{selector_delta['delta_logical_tile_activations_mean']:+.3f}`. These are descriptive effects; no small-effect invalidation threshold is applied.

`all-groups-covered` is an exact retrieval-set success proxy, not final Agent answer accuracy. `rerank_executed=false`: the B-memory selection itself is the handoff. Logical Tiles are host-side logical counts and not PIM hardware measurements.

## Static content-update closure

The allowed maintenance stress completed growth points 1/5/10/20% with exact overlay-versus-compacted bounded-lookup agreement. At 20% growth it appended `{update_last['appended_memory_count']}` memories, used `{update_last['overlay_logical_tiles']}` append-overlay logical Tiles, built a query snapshot in `{update_last['snapshot_build_ms']:.3f} ms`, atomically published it in `{update_last['atomic_publish_ms']:.6f} ms`, served probes at mean `{update_last['overlay_lookup_ms_mean']:.3f} ms`, and compacted the index in `{update_last['periodic_full_compaction_ms']:.3f} ms`. These are content-maintenance measurements over deterministic corpus slices, not causal memory-stream or access-order results.

## Operating decision without an effect threshold

- Quality-first: expanded sparse frontier plus QHDS at B=32.
- Latency-first: the same expanded sparse frontier plus cosine ordering at B=32.

This split attributes the large post-closure improvement to semantic candidate expansion while exposing the much smaller incremental selector effect. Neither operating point is labelled valid or invalid by an arbitrary minimum gain.

## Remaining boundary

The offline index, online retrieval, Agent handoff contract, and static content maintenance are closed. Final Agent answer execution is not evaluated. A true causal dynamic update experiment remains ineligible because the source has no recoverable agent read/access stream. The previous formal-test attempt remains non-reportable due to missing frozen test embeddings and is not rerun here.
"""
    with REPORT.open("x", encoding="utf-8") as handle:
        handle.write(report)
    print(json.dumps({"status": manifest["status"], "manifest": str(OUTPUT), "report": str(REPORT)}, sort_keys=True))


if __name__ == "__main__":
    main()
