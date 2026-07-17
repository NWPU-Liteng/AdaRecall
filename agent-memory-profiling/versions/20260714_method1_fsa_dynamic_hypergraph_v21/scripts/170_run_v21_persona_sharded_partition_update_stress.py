#!/usr/bin/env python3
"""Validate static new-memory partition publication through persona shards.

This is an insertion/maintenance stress test, not a causal time-stream claim.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from time import perf_counter_ns

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from v21.incremental_sparse_overlay import (  # noqa: E402
    AppendOnlyLogicalTileOverlay,
    AtomicPersonaShardedSnapshotPublisher,
    PersonaShardedIncrementalOverlay,
)
from v21.io import iter_jsonl, load_config, read_json, sha256_file, write_csv_new, write_json_new  # noqa: E402
from v21.persona_sharded_index import PersonaShardedSnapshotDispatch  # noqa: E402
from v21.sparse_semantic_index import SparseSemanticLookupRequest  # noqa: E402
from v21.uot_recipes import array_sha256  # noqa: E402


def stable_key(memory_id: str) -> bytes:
    return hashlib.sha256(str(memory_id).encode("utf-8")).digest()


def edge_ids(result, ids: tuple[str, ...]) -> tuple[tuple[int, tuple[str, ...], tuple[float, ...]], ...]:
    return tuple(
        (edge.prototype_id, tuple(ids[position] for position in edge.memory_positions), tuple(edge.incidence_scores))
        for edge in result.edges
    )


def snapshot_edge_ids(snapshot, persona: str, active: tuple[int, ...], eligible: frozenset[str], members: int):
    result = snapshot.lookup(persona, active, eligible, members)
    return tuple((edge.prototype_id, edge.memory_ids, edge.incidence_scores) for edge in result.edges)


def main() -> None:
    config = load_config("v21_persona_sharded_partition_update_stress_r1.json")
    for field in ("causal_stream_claim", "access_order_claim", "co_access_claim", "pim_hardware_claim"):
        if config.get(field) is not False:
            raise ValueError(f"forbidden dynamic claim: {field}")
    data = load_config("v21_data.json")
    ids = tuple(map(str, read_json(data["memory"]["embedding_ids"])))
    position_by_id = {memory_id: position for position, memory_id in enumerate(ids)}
    memory_rows = list(iter_jsonl(data["memory"]["items"]))
    row_by_id = {str(row["memory_id"]): row for row in memory_rows}
    if set(row_by_id) != set(ids):
        raise ValueError("memory items and embedding IDs do not match")
    personas = tuple(str(row_by_id[memory_id]["persona_id"]) for memory_id in ids)
    incidence_path = ROOT / "artifacts/v21_phase1" / "v21_phase1_oof_r1_20260714" / f"seed_{config['source_seed']}" / f"fold_{config['source_fold']}" / "v21_uot_semantic_incidence.npy"
    incidence = np.asarray(np.load(incidence_path, mmap_mode="r"), dtype=np.float32)
    if incidence.shape[0] != len(ids):
        raise ValueError("incidence and embedding IDs are not aligned")

    ordered = tuple(sorted(ids, key=lambda memory_id: (stable_key(memory_id), memory_id)))
    base_count = int(round(len(ids) * float(config["base_fraction"])))
    base_ids = ordered[:base_count]
    new_pool = ordered[base_count:]
    base_positions = np.asarray([position_by_id[memory_id] for memory_id in base_ids], dtype=np.int64)
    base_personas = tuple(personas[position_by_id[memory_id]] for memory_id in base_ids)
    base_incidence = incidence[base_positions]
    all_namespace_ids = tuple(base_ids) + tuple(new_pool)
    namespace_position = {memory_id: position for position, memory_id in enumerate(all_namespace_ids)}
    namespace_personas = tuple(row_by_id[memory_id]["persona_id"] for memory_id in all_namespace_ids)

    overlay = PersonaShardedIncrementalOverlay(base_ids, base_personas, base_incidence)
    initial = overlay.snapshot()
    publisher = AtomicPersonaShardedSnapshotPublisher(initial)
    tile_overlay = AppendOnlyLogicalTileOverlay(capacity=int(config["logical_tile_capacity"]), tile_prefix="partition_append")
    dispatch = PersonaShardedSnapshotDispatch(publisher, all_namespace_ids, namespace_personas)

    query_rows = list(iter_jsonl(data["split"]["train_query_plan"]))
    probe_indices = np.linspace(0, len(query_rows) - 1, num=min(int(config["probe_queries"]), len(query_rows)), dtype=np.int64)
    probes: list[tuple[str, tuple[int, ...], frozenset[int]]] = []
    active_ids = frozenset(base_ids)
    for index in probe_indices:
        row = query_rows[int(index)]
        persona = str(row["persona_id"])
        eligible_ids = frozenset(
            memory_id for memory_id in map(str, row["eligible_memory_ids"])
            if memory_id in active_ids and str(row_by_id[memory_id]["persona_id"]) == persona
        )
        if not eligible_ids:
            continue
        positions = np.asarray([position_by_id[memory_id] for memory_id in eligible_ids], dtype=np.int64)
        mass = np.asarray(incidence[positions].mean(axis=0), dtype=np.float64)
        active = tuple(map(int, np.argsort(-mass, kind="stable")[: int(config["active_prototypes_per_probe"])]))
        probes.append((persona, active, frozenset(namespace_position[memory_id] for memory_id in eligible_ids)))
    if not probes:
        raise ValueError("no eligible static partition probes")

    appended_count = 0
    deleted_count = 0
    results: list[dict] = []
    for growth in map(int, config["growth_points_percent_of_full_corpus"]):
        target_append = min(len(new_pool), int(round(len(ids) * growth / 100.0)))
        target_delete = min(base_count, int(round(target_append * float(config["delete_fraction_of_appended_growth"]))))
        update_start = perf_counter_ns()
        for offset in range(appended_count, target_append):
            memory_id = new_pool[offset]
            overlay.upsert(memory_id, str(row_by_id[memory_id]["persona_id"]), incidence[position_by_id[memory_id]])
            tile_overlay.assign(memory_id)
        for offset in range(deleted_count, target_delete):
            overlay.delete(base_ids[offset])
        update_ms = (perf_counter_ns() - update_start) / 1_000_000.0
        appended_count, deleted_count = target_append, target_delete

        snapshot_start = perf_counter_ns()
        snapshot = overlay.snapshot()
        snapshot_ms = (perf_counter_ns() - snapshot_start) / 1_000_000.0
        publish_start = perf_counter_ns()
        publisher.publish(snapshot)
        publish_ms = (perf_counter_ns() - publish_start) / 1_000_000.0

        active_set = (frozenset(base_ids) | frozenset(new_pool[:appended_count])) - frozenset(base_ids[:deleted_count])
        current_probes = [
            (persona, active, frozenset(position for position in eligible if all_namespace_ids[position] in active_set))
            for persona, active, eligible in probes
        ]
        current_probes = [probe for probe in current_probes if probe[2]]
        overlay_times: list[float] = []
        compact_times: list[float] = []
        dispatch_mismatch = 0
        compact_mismatch = 0
        selected_mismatch = 0
        for persona, active, eligible_positions in current_probes:
            eligible_ids = frozenset(all_namespace_ids[position] for position in eligible_positions)
            request = SparseSemanticLookupRequest(active, eligible_positions, int(config["members_per_prototype"]))
            start = perf_counter_ns()
            dispatched = dispatch.lookup(request)
            overlay_times.append((perf_counter_ns() - start) / 1_000_000.0)
            direct = snapshot_edge_ids(snapshot, persona, active, eligible_ids, int(config["members_per_prototype"]))
            if edge_ids(dispatched, all_namespace_ids) != direct:
                dispatch_mismatch += 1

        compact_start = perf_counter_ns()
        compacted = snapshot.compact()
        compact_ms = (perf_counter_ns() - compact_start) / 1_000_000.0
        rebuilt_overlay = PersonaShardedIncrementalOverlay(compacted.memory_ids, compacted.persona_ids, compacted.incidence)
        rebuilt = rebuilt_overlay.snapshot()
        for persona, active, eligible_positions in current_probes:
            eligible_ids = frozenset(all_namespace_ids[position] for position in eligible_positions)
            start = perf_counter_ns()
            before = snapshot_edge_ids(snapshot, persona, active, eligible_ids, int(config["members_per_prototype"]))
            after = snapshot_edge_ids(rebuilt, persona, active, eligible_ids, int(config["members_per_prototype"]))
            compact_times.append((perf_counter_ns() - start) / 1_000_000.0)
            if before != after:
                compact_mismatch += 1
            before_selected = tuple(memory_id for _prototype, members, _scores in before for memory_id in members)[: int(config["selection_budget"])]
            after_selected = tuple(memory_id for _prototype, members, _scores in after for memory_id in members)[: int(config["selection_budget"])]
            if before_selected != after_selected:
                selected_mismatch += 1
        results.append({
            "partition_growth_percent": growth,
            "base_memory_count": base_count,
            "appended_memory_count": appended_count,
            "deleted_memory_count": deleted_count,
            "active_memory_count": len(active_set),
            "probe_count": len(current_probes),
            "update_batch_ms": update_ms,
            "snapshot_build_ms": snapshot_ms,
            "atomic_publish_ms": publish_ms,
            "overlay_lookup_ms_mean": float(np.mean(overlay_times)),
            "overlay_lookup_ms_p95": float(np.percentile(overlay_times, 95)),
            "compaction_ms": compact_ms,
            "compacted_lookup_ms_mean": float(np.mean(compact_times)),
            "dispatch_mismatch_count": dispatch_mismatch,
            "overlay_compacted_mismatch_count": compact_mismatch,
            "selection_mismatch_count": selected_mismatch,
            "logical_overlay_tiles": len(set(tile_overlay.snapshot().values())),
        })

    metrics_path = ROOT / "aggregate" / "v21_persona_sharded_partition_update_stress_r1_attempt001.csv"
    write_csv_new(metrics_path, results)
    summary = {
        "schema_version": "v21-persona-sharded-partition-update-stress-1",
        "status": "completed_static_new_memory_partition_maintenance_only",
        "revision_id": config["revision_id"],
        "partition_protocol": config["partition_protocol"],
        "base_memory_count": base_count,
        "new_memory_pool_count": len(new_pool),
        "probe_count": len(probes),
        "results": results,
        "quality_contract_satisfied": all(row["dispatch_mismatch_count"] == 0 and row["overlay_compacted_mismatch_count"] == 0 and row["selection_mismatch_count"] == 0 for row in results),
        "dynamic_update_evaluated": False,
        "causal_stream_claim": False,
        "access_order_claim": False,
        "co_access_claim": False,
        "pim_hardware_claim": False,
        "minimum_effect_threshold": None,
        "calibration_read": False,
        "formal_test_read": False,
        "hashes": {
            "config": sha256_file(ROOT / "configs/v21_persona_sharded_partition_update_stress_r1.json"),
            "incidence": array_sha256(incidence),
            "memory_items": sha256_file(data["memory"]["items"]),
            "metrics": sha256_file(metrics_path),
        },
    }
    output = ROOT / "aggregate" / "v21_persona_sharded_partition_update_stress_r1_attempt001.json"
    write_json_new(output, summary)
    report = ROOT / "reports" / "METHOD1_V21_PERSONA_SHARDED_PARTITION_UPDATE_STRESS_R1_ATTEMPT001.md"
    report.write_text(
        "# Method 1 v21 Persona-Sharded Static New-Memory Partition Stress\n\n"
        "Memory rows were deterministically partitioned by SHA-256 memory ID into a frozen base and a new-memory pool. This validates static insertion, deletion, atomic publication, and compaction equivalence; it makes no timestamp, access-order, causal, or PIM claim.\n\n"
        + "| growth | appended | deleted | overlay lookup ms | compaction ms | dispatch mismatch | compact mismatch |\n|---:|---:|---:|---:|---:|---:|---:|\n"
        + "\n".join(f"| {row['partition_growth_percent']}% | {row['appended_memory_count']} | {row['deleted_memory_count']} | {row['overlay_lookup_ms_mean']:.3f} | {row['compaction_ms']:.3f} | {row['dispatch_mismatch_count']} | {row['overlay_compacted_mismatch_count']} |" for row in results)
        + "\n\nThe serving adapter and compacted snapshot returned identical bounded persona/prototype edges and B=8 selections for every probe.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": summary["status"], "quality_contract_satisfied": summary["quality_contract_satisfied"], "summary": str(output), "report": str(report)}, sort_keys=True))


if __name__ == "__main__":
    main()
