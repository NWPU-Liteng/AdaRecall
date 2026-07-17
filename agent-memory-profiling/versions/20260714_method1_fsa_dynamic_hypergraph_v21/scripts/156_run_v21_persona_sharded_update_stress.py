#!/usr/bin/env python3
"""Stress persona-prototype append/delete publication on deterministic train slices."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter_ns

import numpy as np

VERSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VERSION_ROOT / "src"))
from v21.incremental_sparse_overlay import AppendOnlyLogicalTileOverlay, AtomicPersonaShardedSnapshotPublisher, PersonaShardedIncrementalOverlay  # noqa: E402
from v21.io import iter_jsonl, load_config, read_json, sha256_file, write_csv_new, write_json_new  # noqa: E402
from v21.uot_recipes import array_sha256  # noqa: E402


def timed_lookup(snapshot, probes: list[tuple[str, tuple[int, ...], frozenset[str]]], members: int) -> tuple[list[float], list[tuple]]:
    times, outputs = [], []
    for persona, active, eligible in probes:
        start = perf_counter_ns()
        result = snapshot.lookup(persona, active, eligible, members)
        times.append((perf_counter_ns() - start) / 1_000_000.0)
        outputs.append(tuple((edge.prototype_id, edge.memory_ids, edge.incidence_scores) for edge in result.edges))
    return times, outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="v21_persona_sharded_update_stress_r1.json")
    parser.add_argument("--output-revision", default="r1_attempt001")
    args = parser.parse_args()
    config = load_config(args.config)
    for field in ("causal_stream_claim", "access_order_claim", "co_access_claim", "pim_hardware_claim"):
        if config.get(field) is not False:
            raise ValueError("persona-shard maintenance stress cannot make causal/access/PIM claims")
    if config.get("minimum_effect_threshold") is not None or config.get("calibration_read") is not False or config.get("formal_test_read") is not False:
        raise ValueError("threshold and locked split access are forbidden")
    data = load_config("v21_data.json")
    all_ids = tuple(map(str, read_json(data["memory"]["embedding_ids"])))
    embedding_position = {memory_id: position for position, memory_id in enumerate(all_ids)}
    memory_rows = list(iter_jsonl(data["memory"]["items"]))
    row_by_id = {str(row["memory_id"]): row for row in memory_rows}
    ordered_ids = tuple(str(row["memory_id"]) for row in sorted(memory_rows, key=lambda row: (int(row["chronological_rank"]), str(row["memory_id"]))))
    personas = tuple(str(row_by_id[memory_id]["persona_id"]) for memory_id in ordered_ids)
    seed, fold = int(config["source_seed"]), int(config["source_fold"])
    incidence_path = VERSION_ROOT / "artifacts/v21_phase1" / config["phase1_run_id"] / f"seed_{seed}" / f"fold_{fold}" / "v21_uot_semantic_incidence.npy"
    incidence_embedding_order = np.load(incidence_path, mmap_mode="r")
    ordered_positions = np.asarray([embedding_position[memory_id] for memory_id in ordered_ids], dtype=np.int64)
    incidence = np.asarray(incidence_embedding_order[ordered_positions], dtype=np.float32)
    full_count = len(ordered_ids)
    base_count = int(round(full_count * float(config["base_fraction"])))
    base_ids, base_personas, base_incidence = ordered_ids[:base_count], personas[:base_count], incidence[:base_count]
    overlay = PersonaShardedIncrementalOverlay(base_ids, base_personas, base_incidence)
    initial = overlay.snapshot()
    publisher = AtomicPersonaShardedSnapshotPublisher(initial)
    tiles = AppendOnlyLogicalTileOverlay(capacity=int(config["logical_tile_capacity"]), tile_prefix="persona_static_append")
    persona_by_id = dict(zip(ordered_ids, personas, strict=True))

    query_rows = list(iter_jsonl(data["split"]["train_query_plan"]))
    probe_positions = np.linspace(0, len(query_rows) - 1, num=min(int(config["probe_queries"]), len(query_rows)), dtype=np.int64)
    probes = []
    for position in probe_positions:
        row = query_rows[int(position)]
        persona = str(row["persona_id"])
        eligible = frozenset(memory_id for memory_id in map(str, row["eligible_memory_ids"]) if persona_by_id[memory_id] == persona)
        eligible_positions = np.asarray([embedding_position[memory_id] for memory_id in eligible], dtype=np.int64)
        prototype_mass = np.asarray(incidence_embedding_order[eligible_positions].mean(axis=0), dtype=np.float64)
        active = tuple(map(int, np.argsort(-prototype_mass, kind="stable")[: int(config["active_prototypes_per_probe"])]))
        probes.append((persona, active, eligible))
    _ = timed_lookup(initial, probes[: min(int(config["warmup_probes"]), len(probes))], int(config["members_per_prototype"]))

    results, appended_count, deleted_count = [], 0, 0
    for growth_percent in map(int, config["growth_points_percent_of_full_corpus"]):
        target_append = min(full_count - base_count, int(round(full_count * growth_percent / 100.0)))
        target_delete = min(base_count, int(round(target_append * float(config["delete_fraction_of_appended_growth"]))))
        update_start = perf_counter_ns()
        for offset in range(appended_count, target_append):
            absolute = base_count + offset
            overlay.upsert(ordered_ids[absolute], personas[absolute], incidence[absolute])
            tiles.assign(ordered_ids[absolute])
        for offset in range(deleted_count, target_delete):
            overlay.delete(base_ids[base_count - 1 - offset])
        update_ms = (perf_counter_ns() - update_start) / 1_000_000.0
        appended_count, deleted_count = target_append, target_delete
        snapshot_start = perf_counter_ns()
        snapshot = overlay.snapshot()
        snapshot_ms = (perf_counter_ns() - snapshot_start) / 1_000_000.0
        publish_start = perf_counter_ns()
        publisher.publish(snapshot)
        publish_ms = (perf_counter_ns() - publish_start) / 1_000_000.0
        lookup_times, outputs = timed_lookup(snapshot, probes, int(config["members_per_prototype"]))
        compact_start = perf_counter_ns()
        compacted = snapshot.compact()
        compact_ms = (perf_counter_ns() - compact_start) / 1_000_000.0
        rebuilt = PersonaShardedIncrementalOverlay(compacted.memory_ids, compacted.persona_ids, compacted.incidence).snapshot()
        rebuilt_times, rebuilt_outputs = timed_lookup(rebuilt, probes, int(config["members_per_prototype"]))
        mismatches = sum(left != right for left, right in zip(outputs, rebuilt_outputs, strict=True))
        if mismatches:
            raise ValueError(f"persona overlay/compaction mismatch at {growth_percent}%: {mismatches}")
        results.append({"growth_percent_of_full_corpus": growth_percent, "base_memory_count": base_count, "appended_memory_count": appended_count, "deleted_memory_count": deleted_count, "active_memory_count": len(compacted.memory_ids), "update_batch_ms": update_ms, "snapshot_build_ms": snapshot_ms, "atomic_publish_ms": publish_ms, "overlay_lookup_ms_mean": float(np.mean(lookup_times)), "overlay_lookup_ms_p95": float(np.percentile(lookup_times, 95)), "periodic_full_compaction_ms": compact_ms, "compacted_lookup_ms_mean": float(np.mean(rebuilt_times)), "lookup_probe_count": len(probes), "lookup_mismatch_count": mismatches, "overlay_logical_tiles": len(set(tiles.snapshot().values()))})
    metrics_path = VERSION_ROOT / "aggregate" / f"v21_persona_sharded_update_stress_{args.output_revision}.csv"
    write_csv_new(metrics_path, results)
    summary = {"schema_version": "v21-persona-sharded-update-stress-1", "status": "completed_static_persona_sharded_content_maintenance_only", "revision_id": config["revision_id"], "source_seed": seed, "source_fold": fold, "memory_count": full_count, "prototype_count": int(incidence.shape[1]), "results": results, "quality_contract_satisfied": all(row["lookup_mismatch_count"] == 0 for row in results), "static_content_update_mechanics_evaluated": True, "dynamic_update_evaluated": False, "causal_stream_claim": False, "access_order_claim": False, "co_access_claim": False, "pim_hardware_claim": False, "minimum_effect_threshold": None, "calibration_read": False, "formal_test_read": False, "hashes": {"config": sha256_file(VERSION_ROOT / "configs" / args.config), "incidence": array_sha256(incidence_embedding_order), "memory_items": sha256_file(data["memory"]["items"]), "metrics": sha256_file(metrics_path)}}
    output = VERSION_ROOT / "aggregate" / f"v21_persona_sharded_update_stress_{args.output_revision}.json"
    write_json_new(output, summary)
    print(json.dumps({"status": summary["status"], "quality_contract_satisfied": summary["quality_contract_satisfied"], "summary": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
