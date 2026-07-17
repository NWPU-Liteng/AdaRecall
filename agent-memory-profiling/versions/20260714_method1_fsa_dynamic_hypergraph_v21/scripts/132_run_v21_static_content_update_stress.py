#!/usr/bin/env python3
"""Benchmark the permitted static content-update overlay on frozen train data.

Corpus ordering is used only to make deterministic maintenance slices. It is
not interpreted as an agent access trace or a causal insertion stream.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter_ns

import numpy as np


VERSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VERSION_ROOT / "src"))

from v21.incremental_sparse_overlay import (  # noqa: E402
    AppendOnlyLogicalTileOverlay,
    AtomicSparseSnapshotPublisher,
    IncrementalSparseOverlay,
)
from v21.io import iter_jsonl, load_config, read_json, sha256_file, write_csv_new, write_json_new  # noqa: E402
from v21.uot_recipes import array_sha256  # noqa: E402


def build_ranks(ids: tuple[str, ...], incidence: np.ndarray) -> np.ndarray:
    ranks = np.empty((incidence.shape[1], incidence.shape[0]), dtype=np.int64)
    for prototype in range(incidence.shape[1]):
        ranks[prototype] = np.asarray(sorted(
            range(len(ids)),
            key=lambda position: (-float(incidence[position, prototype]), ids[position]),
        ), dtype=np.int64)
    return ranks


def timed_lookup(snapshot, probes: list[tuple[tuple[int, ...], frozenset[str]]], members: int) -> tuple[list[float], list[tuple]]:
    times: list[float] = []
    outputs: list[tuple] = []
    for active, eligible in probes:
        start = perf_counter_ns()
        result = snapshot.lookup(active, eligible, members)
        times.append((perf_counter_ns() - start) / 1_000_000.0)
        outputs.append(tuple((edge.prototype_id, edge.memory_ids, edge.incidence_scores) for edge in result.edges))
    return times, outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="v21_static_content_update_stress_r1.json")
    parser.add_argument("--output-revision", default="r1_attempt001")
    args = parser.parse_args()
    config = load_config(args.config)
    prohibited = ("causal_stream_claim", "access_order_claim", "co_access_claim", "pim_hardware_claim")
    if any(config.get(field) is not False for field in prohibited):
        raise ValueError("static update stress cannot make causal/access/PIM claims")
    if config.get("calibration_read") is not False or config.get("formal_test_read") is not False:
        raise ValueError("locked split access is forbidden")
    if config.get("minimum_effect_threshold") is not None:
        raise ValueError("update stress must not use an effect threshold")

    data = load_config("v21_data.json")
    all_embedding_ids = tuple(map(str, read_json(data["memory"]["embedding_ids"])))
    embedding_position = {memory_id: position for position, memory_id in enumerate(all_embedding_ids)}
    memory_rows = list(iter_jsonl(data["memory"]["items"]))
    if len(memory_rows) != len(all_embedding_ids):
        raise ValueError("memory item and embedding counts do not align")
    ordered_ids = tuple(str(row["memory_id"]) for row in sorted(memory_rows, key=lambda row: (int(row["chronological_rank"]), str(row["memory_id"]))))
    if set(ordered_ids) != set(all_embedding_ids):
        raise ValueError("content ordering does not cover the frozen embedding IDs")

    seed = int(config["source_seed"])
    fold = int(config["source_fold"])
    incidence_path = VERSION_ROOT / "artifacts" / "v21_phase1" / config["phase1_run_id"] / f"seed_{seed}" / f"fold_{fold}" / "v21_uot_semantic_incidence.npy"
    incidence_embedding_order = np.load(incidence_path, mmap_mode="r")
    ordered_positions = np.asarray([embedding_position[memory_id] for memory_id in ordered_ids], dtype=np.int64)
    incidence = np.asarray(incidence_embedding_order[ordered_positions], dtype=np.float32)
    full_count = len(ordered_ids)
    base_count = int(round(full_count * float(config["base_fraction"])))
    if base_count <= 0 or base_count >= full_count:
        raise ValueError("base fraction must leave a nonempty append slice")
    base_ids = ordered_ids[:base_count]
    base_incidence = incidence[:base_count]
    base_ranks = build_ranks(base_ids, base_incidence)
    overlay = IncrementalSparseOverlay(base_ids, base_incidence, base_ranks)
    initial = overlay.snapshot()
    publisher = AtomicSparseSnapshotPublisher(initial)
    tile_overlay = AppendOnlyLogicalTileOverlay(capacity=int(config["logical_tile_capacity"]), tile_prefix="static_append")

    query_rows = list(iter_jsonl(data["split"]["train_query_plan"]))
    probe_count = min(int(config["probe_queries"]), len(query_rows))
    probe_positions = np.linspace(0, len(query_rows) - 1, num=probe_count, dtype=np.int64)
    active_count = int(config["active_prototypes_per_probe"])
    probes: list[tuple[tuple[int, ...], frozenset[str]]] = []
    for position in probe_positions:
        row = query_rows[int(position)]
        eligible = frozenset(map(str, row["eligible_memory_ids"]))
        eligible_positions = np.asarray([embedding_position[memory_id] for memory_id in eligible], dtype=np.int64)
        prototype_mass = np.asarray(incidence_embedding_order[eligible_positions].mean(axis=0), dtype=np.float64)
        active = tuple(map(int, np.argsort(-prototype_mass, kind="stable")[:active_count]))
        probes.append((active, eligible))
    warmup = probes[: min(int(config["warmup_probes"]), len(probes))]
    _ = timed_lookup(initial, warmup, int(config["members_per_prototype"]))

    results: list[dict] = []
    appended_count = 0
    for growth_percent in map(int, config["growth_points_percent_of_full_corpus"]):
        target = min(full_count - base_count, int(round(full_count * growth_percent / 100.0)))
        if target < appended_count:
            raise ValueError("growth points must increase")
        update_start = perf_counter_ns()
        for offset in range(appended_count, target):
            absolute = base_count + offset
            memory_id = ordered_ids[absolute]
            overlay.upsert(memory_id, incidence[absolute])
            tile_overlay.assign(memory_id)
        update_ms = (perf_counter_ns() - update_start) / 1_000_000.0
        appended_count = target
        snapshot_start = perf_counter_ns()
        snapshot = overlay.snapshot()
        snapshot_build_ms = (perf_counter_ns() - snapshot_start) / 1_000_000.0
        publish_start = perf_counter_ns()
        publisher.publish(snapshot)
        atomic_publish_ms = (perf_counter_ns() - publish_start) / 1_000_000.0

        lookup_times, lookup_outputs = timed_lookup(publisher.current(), probes, int(config["members_per_prototype"]))
        compact_start = perf_counter_ns()
        compacted = snapshot.compact()
        compaction_ms = (perf_counter_ns() - compact_start) / 1_000_000.0
        rebuilt = IncrementalSparseOverlay(compacted.memory_ids, compacted.incidence, compacted.prototype_memory_rank_index).snapshot()
        rebuilt_times, rebuilt_outputs = timed_lookup(rebuilt, probes, int(config["members_per_prototype"]))
        mismatches = sum(left != right for left, right in zip(lookup_outputs, rebuilt_outputs, strict=True))
        if mismatches:
            raise ValueError(f"overlay/compacted lookup mismatch at growth {growth_percent}%: {mismatches}")
        results.append({
            "growth_percent_of_full_corpus": growth_percent,
            "base_memory_count": base_count,
            "appended_memory_count": appended_count,
            "active_memory_count": len(compacted.memory_ids),
            "incremental_upsert_batch_ms": update_ms,
            "incremental_upsert_us_per_new_memory": update_ms * 1000.0 / max(target - (results[-1]["appended_memory_count"] if results else 0), 1),
            "snapshot_build_ms": snapshot_build_ms,
            "atomic_publish_ms": atomic_publish_ms,
            "overlay_lookup_ms_mean": float(np.mean(lookup_times)),
            "overlay_lookup_ms_p95": float(np.percentile(lookup_times, 95)),
            "periodic_full_compaction_ms": compaction_ms,
            "compacted_lookup_ms_mean": float(np.mean(rebuilt_times)),
            "compacted_lookup_ms_p95": float(np.percentile(rebuilt_times, 95)),
            "lookup_probe_count": len(probes),
            "lookup_mismatch_count": mismatches,
            "overlay_logical_tiles": len(set(tile_overlay.snapshot().values())),
        })

    metrics_path = VERSION_ROOT / "aggregate" / f"v21_static_content_update_stress_{args.output_revision}.csv"
    write_csv_new(metrics_path, results)
    summary = {
        "schema_version": "v21-static-content-update-stress-1",
        "status": "completed_static_content_maintenance_only",
        "revision_id": config["revision_id"],
        "source_seed": seed,
        "source_fold": fold,
        "memory_count": full_count,
        "prototype_count": int(incidence.shape[1]),
        "ordering_role": config["ordering_role"],
        "results": results,
        "quality_contract_satisfied": all(row["lookup_mismatch_count"] == 0 for row in results),
        "dynamic_update_evaluated": False,
        "static_content_update_mechanics_evaluated": True,
        "causal_stream_claim": False,
        "access_order_claim": False,
        "co_access_claim": False,
        "pim_hardware_claim": False,
        "minimum_effect_threshold": None,
        "calibration_read": False,
        "formal_test_read": False,
        "artifacts": {"metrics": str(metrics_path.relative_to(VERSION_ROOT))},
        "hashes": {
            "config": sha256_file(VERSION_ROOT / "configs" / args.config),
            "incidence": array_sha256(incidence_embedding_order),
            "memory_items": sha256_file(data["memory"]["items"]),
            "metrics": sha256_file(metrics_path),
        },
    }
    summary_path = VERSION_ROOT / "aggregate" / f"v21_static_content_update_stress_{args.output_revision}.json"
    write_json_new(summary_path, summary)
    print(json.dumps({"status": summary["status"], "quality_contract_satisfied": summary["quality_contract_satisfied"], "summary": str(summary_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
