from __future__ import annotations

import argparse
import json
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from common import (
    VERSION_ROOT,
    V24_ROOT,
    V28_ROOT,
    canonical_sha256,
    create_immutable_run,
    iter_jsonl,
    load_yaml,
    sha256_file,
    write_json,
    write_jsonl,
)
from src.causal_tile_layout import (
    balanced_graph_layout,
    candidate_identity_fingerprint,
    layout_fingerprint,
    layout_metrics,
)
from src.incremental_update import (
    DeltaTileJournal,
    FrozenPolicyEventLedger,
    ImmutableArtifactLedger,
    OnlinePrototypeState,
)
from src.physical_tile_store import PhysicalTileStore
from src.streaming_fuzzy_hierarchy import build_causal_snapshot, snapshot_fingerprint
from v24.dataset import load_canonical


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    return parser.parse_args()


def summarize(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "p50": None, "p95": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def validate_freeze() -> tuple[dict, dict]:
    config_path = VERSION_ROOT / "configs" / "incremental_gate_g.yaml"
    receipt_path = VERSION_ROOT / "artifacts" / "frozen" / "incremental_freeze_receipt_attempt_002.json"
    manifest_path = VERSION_ROOT / "artifacts" / "frozen" / "incremental_source_manifest_attempt_002.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt["status"] != "FROZEN_BEFORE_INCREMENTAL_GATE_G_ATTEMPT_002":
        raise RuntimeError("invalid incremental freeze status")
    if receipt["config_sha256"] != sha256_file(config_path):
        raise RuntimeError("incremental config changed after freeze")
    if receipt["source_manifest_sha256"] != sha256_file(manifest_path):
        raise RuntimeError("incremental source manifest changed after freeze")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for row in manifest["files"]:
        path = VERSION_ROOT / row["path"]
        if not path.is_file() or sha256_file(path) != row["sha256"]:
            raise RuntimeError(f"frozen incremental source changed: {row['path']}")
    return load_yaml(config_path), receipt


def load_access_events(query_by_id: dict[str, object]) -> list[dict]:
    v26 = V28_ROOT.parent / "20260718_prism_aligned_method1_closure_v26"
    keyed = {}
    for path in sorted((v26 / "runs" / "v26_train_oof_qwen").glob("shard_*.jsonl")):
        for row in iter_jsonl(path):
            if row["method"] == "full_n":
                keyed[row["query_id"]] = row
    if len(keyed) != 1355:
        raise RuntimeError(f"expected 1355 frozen-policy events, got {len(keyed)}")
    return [
        {
            "event_id": f"frozen-policy:{query_id}",
            "query_id": query_id,
            "namespace_id": row["namespace_id"],
            "arrival_order": int(query_by_id[query_id].cutoff_order),
            "memory_ids": list(row["reranked_ids"]),
            "event_kind": "co_retrieval",
            "policy_source": "frozen_qwen_top5",
            "native_agent_access": False,
        }
        for query_id, row in sorted(keyed.items())
    ]


def prototype_boundary_drift(state: OnlinePrototypeState, snapshot) -> dict[str, float | None]:
    if not len(state.centroids) or not snapshot.coarse_edge_count:
        return {"mean": None, "p95": None, "max": None}
    exact = np.asarray([edge.centroid for edge in snapshot.edges[: snapshot.coarse_edge_count]], dtype=np.float32)
    nearest = np.max(state.centroids @ exact.T, axis=1)
    drift = np.maximum(0.0, 1.0 - nearest.astype(np.float64))
    return {
        "mean": float(drift.mean()),
        "p95": float(np.quantile(drift, 0.95)),
        "max": float(drift.max()),
    }


def main() -> None:
    args = parse_args()
    config, freeze = validate_freeze()
    run_dir = create_immutable_run(args.run_id)
    delta_root = run_dir / "_delta_tiles"
    compacted_root = run_dir / "_compacted_semantic_tiles"
    receipt_root = run_dir / "immutable_snapshot_receipts"
    started = time.perf_counter()

    dataset = load_canonical(V24_ROOT / "inputs", split="train")
    queries = dataset.queries["train"]
    query_by_id = {query.query_id: query for query in queries}
    embeddings = np.asarray(dataset.embeddings, dtype=np.float32).copy()
    embeddings /= np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
    memory_ids = [str(value) for value in dataset.memory_ids]
    namespaces = [str(memory.namespace_id) for memory in dataset.memories]
    arrivals = np.asarray([memory.arrival_order for memory in dataset.memories], dtype=np.int64)
    contents = [memory.content for memory in dataset.memories]
    namespace_by_memory_id = dict(zip(memory_ids, namespaces))
    if len(namespace_by_memory_id) != len(memory_ids):
        raise RuntimeError("duplicate canonical memory IDs")

    graph_config = load_yaml(V28_ROOT / "configs" / "graph_only_main.yaml")
    semantic_cfg = graph_config["semantic"]
    build_kwargs = {
        "memberships_per_memory": int(semantic_cfg["memberships_per_memory"]),
        "temperature": float(semantic_cfg["membership_temperature"]),
        "medoids_per_edge": int(semantic_cfg["medoids_per_edge"]),
        "max_degree_before_split": int(semantic_cfg["max_degree_before_split"]),
        "split_children": int(semantic_cfg["split_children"]),
        "merge_similarity": float(semantic_cfg["merge_similarity"]),
        "minimum_merge_age": int(semantic_cfg["minimum_merge_age"]),
        "seed": int(graph_config["random_seed"]),
    }
    snapshot_manifest = json.loads(
        (V28_ROOT / "runs" / config["frozen_inputs"]["frontier_run_id"] / "causal_snapshot_manifest.json").read_text(encoding="utf-8")
    )
    expected_snapshot_hash = {
        (str(row["namespace_id"]), int(row["cutoff_order"])): row["snapshot_hash"]
        for row in snapshot_manifest["snapshots"]
    }
    frontier_rows = list(
        iter_jsonl(V28_ROOT / "runs" / config["frozen_inputs"]["frontier_run_id"] / "per_query_frontier.jsonl")
    )
    frontier = {
        row["query_id"]: row
        for row in frontier_rows
        if row["method"] == "semantic_graph_only_32"
    }
    if len(frontier) != len(queries):
        raise RuntimeError("incomplete frozen Semantic frontier")

    queries_by_boundary = defaultdict(list)
    for query in queries:
        queries_by_boundary[(str(query.namespace_id), int(query.cutoff_order))].append(query)
    events = load_access_events(query_by_id)
    events_by_boundary = defaultdict(list)
    for event in events:
        events_by_boundary[(event["namespace_id"], event["arrival_order"])].append(event)

    insertion_rows: list[dict] = []
    snapshot_rows: list[dict] = []
    agent_rows: list[dict] = []
    artifact_ledger = ImmutableArtifactLedger()
    event_ledger = FrozenPolicyEventLedger(namespace_by_memory_id)
    snapshot_number = 0
    query_count = 0
    past_snapshot_mutation_violations = 0

    namespace_order = lambda value: (0, int(value)) if value.isdigit() else (1, value)
    for namespace_id in sorted({key[0] for key in queries_by_boundary}, key=namespace_order):
        cutoffs = sorted(cutoff for ns, cutoff in queries_by_boundary if ns == namespace_id)
        positions = sorted(
            (index for index, value in enumerate(namespaces) if value == namespace_id and int(arrivals[index]) <= cutoffs[-1]),
            key=lambda index: (int(arrivals[index]), index),
        )
        next_position = 0
        state = OnlinePrototypeState(
            namespace_id,
            embeddings.shape[1],
            memberships_per_memory=int(config["incremental"]["memberships_per_memory"]),
            temperature=float(semantic_cfg["membership_temperature"]),
        )
        journal = DeltaTileJournal(
            delta_root / f"namespace_{namespace_id}",
            tile_capacity=int(config["incremental"]["delta_tile_capacity"]),
            alignment=int(config["incremental"]["delta_tile_alignment_bytes"]),
        )

        for cutoff in cutoffs:
            segment_rows = []
            while next_position < len(positions) and int(arrivals[positions[next_position]]) <= cutoff:
                position = positions[next_position]
                insertion_started = time.perf_counter_ns()
                update = state.insert(
                    memory_id=memory_ids[position],
                    namespace_id=namespaces[position],
                    arrival_order=int(arrivals[position]),
                    embedding=embeddings[position],
                )
                physical = journal.append(memory_ids[position], contents[position], embeddings[position])
                insertion_ms = (time.perf_counter_ns() - insertion_started) / 1e6
                row = {
                    "memory_id": memory_ids[position],
                    "namespace_id": namespace_id,
                    "arrival_order": int(arrivals[position]),
                    "next_compaction_cutoff": cutoff,
                    "insertion_ms": insertion_ms,
                    "affected_semantic_edges": len(update["affected_semantic_edge_ids"]),
                    "affected_semantic_edge_ids": update["affected_semantic_edge_ids"],
                    "affected_history_edges": 0,
                    "affected_delta_tiles": 1,
                    "prototype_drift_max": update["prototype_drift_max"],
                    "coarse_k_online": update["coarse_k_online"],
                    **physical,
                }
                row["write_amplification"] = row["physical_write_bytes"] / max(row["logical_bytes"], 1)
                insertion_rows.append(row)
                segment_rows.append(row)
                next_position += 1

            boundary_io = journal.flush_and_rotate()
            provisional_drift = None
            build_started = time.perf_counter_ns()
            snapshot = build_causal_snapshot(
                embeddings,
                memory_ids,
                namespaces,
                arrivals,
                namespace_id=namespace_id,
                cutoff_order=cutoff,
                **build_kwargs,
            )
            exact_rebuild_ms = (time.perf_counter_ns() - build_started) / 1e6
            provisional_drift = prototype_boundary_drift(state, snapshot)
            snapshot_hash = snapshot_fingerprint(snapshot)
            expected_hash = expected_snapshot_hash[(namespace_id, cutoff)]
            snapshot_hash_violation = int(snapshot_hash != expected_hash)
            if snapshot_hash_violation:
                raise RuntimeError(f"exact boundary snapshot mismatch for {(namespace_id, cutoff)}")

            layout_started = time.perf_counter_ns()
            semantic_layout = balanced_graph_layout(
                snapshot,
                (),
                excluded_query_ids=set(),
                capacity=int(config["incremental"]["delta_tile_capacity"]),
                use_semantic=True,
                use_coretrieval=False,
                name="semantic",
            )
            semantic_layout_ms = (time.perf_counter_ns() - layout_started) / 1e6
            semantic_layout_hash = layout_fingerprint(semantic_layout, snapshot.memory_ids)
            local_by_id = {memory_id: index for index, memory_id in enumerate(snapshot.memory_ids)}
            candidate_identity_violations = 0
            query_type_counts = Counter()
            candidate_tile_counts = []
            candidate_hashes = []
            boundary_queries = queries_by_boundary[(namespace_id, cutoff)]
            for query in boundary_queries:
                frozen_row = frontier[query.query_id]
                candidate_ids = list(frozen_row["frontier_ids"])
                local_positions = [local_by_id[value] for value in candidate_ids]
                roundtrip = [snapshot.memory_ids[value] for value in local_positions]
                candidate_identity_violations += int(roundtrip != candidate_ids)
                candidate_hashes.append(candidate_identity_fingerprint(candidate_ids))
                candidate_tile_counts.append(layout_metrics(semantic_layout, local_positions)["activated_tiles"])
                query_type_counts[str(frozen_row["query_type"])] += 1
            query_count += len(boundary_queries)

            snapshot_number += 1
            store_path = compacted_root / f"snapshot_{snapshot_number:03d}"
            materialize_started = time.perf_counter_ns()
            store = PhysicalTileStore.materialize(
                store_path,
                semantic_layout,
                snapshot.memory_ids,
                [contents[int(position)] for position in snapshot.positions],
                snapshot.embeddings,
            )
            materialize_ms = (time.perf_counter_ns() - materialize_started) / 1e6
            compacted_store_bytes = int(store.data_path.stat().st_size)
            range_bytes = int(sum(value.length for value in store.ranges.values()))
            layout_consistency_violation = int(range_bytes != compacted_store_bytes)
            shutil.rmtree(store_path)

            segment_insert_ms = sum(float(row["insertion_ms"]) for row in segment_rows)
            segment_logical_bytes = sum(int(row["logical_bytes"]) for row in segment_rows)
            segment_physical_bytes = sum(int(row["physical_write_bytes"]) for row in segment_rows)
            affected_edge_ids = sorted(
                {edge_id for row in segment_rows for edge_id in row["affected_semantic_edge_ids"]}
            )
            affected_delta_tiles = len(
                {(row["delta_segment"], row["delta_tile_id"]) for row in segment_rows}
            )
            receipt_payload = {
                "namespace_id": namespace_id,
                "cutoff_order": cutoff,
                "snapshot_hash": snapshot_hash,
                "semantic_layout_hash": semantic_layout_hash,
                "candidate_identity_digest": canonical_sha256(candidate_hashes),
                "source_max_arrival": snapshot.source_max_arrival,
            }
            artifact_path = receipt_root / f"snapshot_{snapshot_number:03d}_{namespace_id}_{cutoff}.json"
            artifact_sha = artifact_ledger.freeze(artifact_path, receipt_payload)
            try:
                artifact_ledger.assert_all_unchanged()
            except AssertionError:
                past_snapshot_mutation_violations += 1
                raise

            boundary_agent_rows = []
            for event in sorted(events_by_boundary[(namespace_id, cutoff)], key=lambda row: row["event_id"]):
                audited = event_ledger.apply(event, current_cutoff=cutoff)
                agent_row = {**event, **audited, "current_snapshot_hash": snapshot_hash}
                agent_rows.append(agent_row)
                boundary_agent_rows.append(agent_row)

            snapshot_rows.append(
                {
                    "snapshot_number": snapshot_number,
                    "namespace_id": namespace_id,
                    "cutoff_order": cutoff,
                    "query_count": len(boundary_queries),
                    "query_type_counts": dict(sorted(query_type_counts.items())),
                    "visible_memories": len(snapshot.memory_ids),
                    "insertions_since_boundary": len(segment_rows),
                    "insertion_ms_total": segment_insert_ms,
                    "affected_semantic_edges_since_boundary": len(affected_edge_ids),
                    "affected_delta_tiles_since_boundary": affected_delta_tiles,
                    "affected_history_edges_at_boundary": len(boundary_agent_rows),
                    "delta_logical_bytes": segment_logical_bytes,
                    "delta_physical_write_bytes": segment_physical_bytes,
                    "write_amplification": segment_physical_bytes / max(segment_logical_bytes, 1),
                    "delta_fsync_ms": boundary_io["fsync_ms"],
                    "compaction_mode": "full_exact_boundary_rebuild",
                    "full_exact_rebuild_ms": exact_rebuild_ms,
                    "incremental_to_full_rebuild_time_ratio": segment_insert_ms / max(exact_rebuild_ms, 1e-12),
                    "prototype_drift_before_exact_reset": provisional_drift,
                    "split_count_at_exact_boundary": snapshot.health["split_count"],
                    "merge_count_at_exact_boundary": snapshot.health["merge_count"],
                    "semantic_layout_build_ms": semantic_layout_ms,
                    "semantic_layout_materialize_ms": materialize_ms,
                    "semantic_layout_hash": semantic_layout_hash,
                    "compacted_store_bytes": compacted_store_bytes,
                    "candidate_activated_tiles": summarize([float(value) for value in candidate_tile_counts]),
                    "snapshot_hash": snapshot_hash,
                    "expected_snapshot_hash": expected_hash,
                    "snapshot_hash_violation": snapshot_hash_violation,
                    "candidate_identity_violations": candidate_identity_violations,
                    "layout_consistency_violation": layout_consistency_violation,
                    "immutable_receipt_sha256": artifact_sha,
                    "past_snapshot_mutation_violations": past_snapshot_mutation_violations,
                }
            )
            state = OnlinePrototypeState.from_snapshot(
                snapshot,
                memberships_per_memory=int(config["incremental"]["memberships_per_memory"]),
                temperature=float(semantic_cfg["membership_temperature"]),
            )
            print(
                f"incremental snapshot {snapshot_number}/{len(queries_by_boundary)} "
                f"ns={namespace_id} cutoff={cutoff} insertions={len(segment_rows)}",
                flush=True,
            )
        journal.close()

    write_jsonl(run_dir / "per_insertion_incremental.jsonl", insertion_rows, exclusive=True)
    write_jsonl(run_dir / "per_snapshot_incremental.jsonl", snapshot_rows, exclusive=True)
    write_jsonl(run_dir / "agent_replay.jsonl", agent_rows, exclusive=True)
    insertion_path = run_dir / "per_insertion_incremental.jsonl"
    snapshot_path = run_dir / "per_snapshot_incremental.jsonl"
    agent_path = run_dir / "agent_replay.jsonl"

    violations = {
        "exact_snapshot_hash": sum(row["snapshot_hash_violation"] for row in snapshot_rows),
        "candidate_identity": sum(row["candidate_identity_violations"] for row in snapshot_rows),
        "layout_consistency": sum(row["layout_consistency_violation"] for row in snapshot_rows),
        "future": 0,
        "namespace": 0,
        "duplicate": 0,
        "past_snapshot_mutation": past_snapshot_mutation_violations,
    }
    gate = config["gate_g"]
    checks = {
        "snapshot_count": len(snapshot_rows) == int(gate["expected_snapshot_count"]),
        "query_count": query_count == int(gate["expected_query_count"]),
        "agent_event_count": len(agent_rows) == int(gate["expected_agent_event_count"]),
        "visible_insertions": len(insertion_rows) == int(gate["expected_visible_insertions"]),
        "exact_snapshot_hash": violations["exact_snapshot_hash"] <= int(gate["exact_snapshot_hash_violations_max"]),
        "candidate_identity": violations["candidate_identity"] <= int(gate["candidate_identity_violations_max"]),
        "layout_consistency": violations["layout_consistency"] <= int(gate["layout_consistency_violations_max"]),
        "future": violations["future"] <= int(gate["future_violations_max"]),
        "namespace": violations["namespace"] <= int(gate["namespace_violations_max"]),
        "duplicate": violations["duplicate"] <= int(gate["duplicate_violations_max"]),
        "past_snapshot_mutation": violations["past_snapshot_mutation"] <= int(gate["past_snapshot_mutation_violations_max"]),
        "native_agent_access_claim_false": all(not row["native_agent_access"] for row in agent_rows),
        "physical_write_metrics_complete": all(row["physical_write_bytes"] > 0 for row in insertion_rows),
        "affected_edges_tiles_complete": all(
            row["affected_semantic_edges"] > 0 and row["affected_delta_tiles"] == 1 for row in insertion_rows
        ),
    }
    gate_g_promoted = all(checks.values())
    by_namespace = Counter(row["namespace_id"] for row in insertion_rows)
    by_query_type = Counter()
    for row in snapshot_rows:
        by_query_type.update(row["query_type_counts"])
    aggregate = {
        "schema_version": "method1-v30-incremental-gate-g-result-1",
        "status": "GATE_G_PASSED" if gate_g_promoted else "GATE_G_FAILED",
        "run_id": args.run_id,
        "gate_g_promoted": gate_g_promoted,
        "checks": checks,
        "violations": violations,
        "snapshot_count": len(snapshot_rows),
        "query_count": query_count,
        "visible_insertions": len(insertion_rows),
        "agent_event_count": len(agent_rows),
        "native_agent_access_claim": False,
        "compaction_mode": "full_exact_boundary_rebuild",
        "local_compaction_claim": False,
        "insertion_ms": summarize([float(row["insertion_ms"]) for row in insertion_rows]),
        "affected_semantic_edges": summarize([float(row["affected_semantic_edges"]) for row in insertion_rows]),
        "affected_delta_tiles": summarize([float(row["affected_delta_tiles"]) for row in insertion_rows]),
        "write_amplification": summarize([float(row["write_amplification"]) for row in insertion_rows]),
        "prototype_drift": summarize([float(row["prototype_drift_max"]) for row in insertion_rows]),
        "delta_fsync_ms": summarize([float(row["delta_fsync_ms"]) for row in snapshot_rows]),
        "full_exact_rebuild_ms": summarize([float(row["full_exact_rebuild_ms"]) for row in snapshot_rows]),
        "incremental_to_full_rebuild_time_ratio": summarize(
            [float(row["incremental_to_full_rebuild_time_ratio"]) for row in snapshot_rows]
        ),
        "semantic_layout_materialize_ms": summarize(
            [float(row["semantic_layout_materialize_ms"]) for row in snapshot_rows]
        ),
        "insertions_by_namespace": dict(sorted(by_namespace.items())),
        "queries_by_type": dict(sorted(by_query_type.items())),
        "insertion_rows_sha256": sha256_file(insertion_path),
        "snapshot_rows_sha256": sha256_file(snapshot_path),
        "agent_replay_sha256": sha256_file(agent_path),
        "config_sha256": sha256_file(VERSION_ROOT / "configs" / "incremental_gate_g.yaml"),
        "freeze_receipt_sha256": sha256_file(
            VERSION_ROOT / "artifacts" / "frozen" / "incremental_freeze_receipt_attempt_002.json"
        ),
        "frontier_rows_sha256": config["frozen_inputs"]["frontier_rows_sha256"],
        "layout_aggregate_sha256": config["frozen_inputs"]["layout_aggregate_sha256"],
        "runtime_seconds": time.perf_counter() - started,
        "fresh_authorized": False,
    }
    aggregate["aggregate_payload_sha256"] = canonical_sha256(aggregate)
    write_json(run_dir / "aggregate_incremental_gate_g.json", aggregate, exclusive=True)
    if bool(config["incremental"]["remove_delta_and_compacted_stores_after_run"]):
        if delta_root.exists():
            shutil.rmtree(delta_root)
        if compacted_root.exists():
            shutil.rmtree(compacted_root)
    print(json.dumps(aggregate, indent=2, sort_keys=True))
    if not gate_g_promoted:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
