from __future__ import annotations

import json
import time
from collections import defaultdict

import numpy as np

from common import VERSION_ROOT, V24_ROOT, V28_ROOT, iter_jsonl, load_yaml, sha256_file, write_json
from src.causal_tile_layout import balanced_graph_layout, layout_fingerprint, layout_metrics
from src.streaming_fuzzy_hierarchy import build_causal_snapshot, snapshot_fingerprint
from v24.dataset import load_canonical


def main() -> None:
    gate_g_run = VERSION_ROOT / "runs" / "v30_incremental_gate_g_20260719"
    layout_run = VERSION_ROOT / "runs" / "v30_layout_io_e2e_20260719"
    expected = {
        "gate_g_snapshot_rows_sha256": "f8137ce07aed99d8e68b47f6417c72cdf8c2c06a2e863034173b5faf01911e39",
        "layout_rows_sha256": "8813ce0f389bbb4015b1b4c69da8eaacc0b3eaa7d5be8e8a4c9beaeae284e1d0",
        "frontier_rows_sha256": "834b86286a88b9617632452dd72c9d824ac88ad1f3152eb77b52447b91520b98",
    }
    frontier_path = (
        V28_ROOT
        / "runs"
        / "v28_frontier_oof_20260719_attempt003_913cd3f"
        / "per_query_frontier.jsonl"
    )
    observed = {
        "gate_g_snapshot_rows_sha256": sha256_file(gate_g_run / "per_snapshot_incremental.jsonl"),
        "layout_rows_sha256": sha256_file(layout_run / "per_query_layout_io_e2e.jsonl"),
        "frontier_rows_sha256": sha256_file(frontier_path),
    }
    if observed != expected:
        raise RuntimeError(f"cross-run layout input mismatch: {observed}")

    started = time.perf_counter()
    dataset = load_canonical(V24_ROOT / "inputs", split="train")
    queries = dataset.queries["train"]
    embeddings = dataset.embeddings.copy()
    embeddings /= np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
    memory_ids = [str(value) for value in dataset.memory_ids]
    namespaces = [str(memory.namespace_id) for memory in dataset.memories]
    arrivals = np.asarray([memory.arrival_order for memory in dataset.memories])
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
    prior_rows = {row["query_id"]: row for row in iter_jsonl(layout_run / "per_query_layout_io_e2e.jsonl")}
    frontier = {
        row["query_id"]: row
        for row in iter_jsonl(frontier_path)
        if row["method"] == "semantic_graph_only_32"
    }
    immutable_receipts = {}
    for path in (gate_g_run / "immutable_snapshot_receipts").glob("*.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        immutable_receipts[(str(row["namespace_id"]), int(row["cutoff_order"]))] = row
    grouped = defaultdict(list)
    for query in queries:
        grouped[(str(query.namespace_id), int(query.cutoff_order))].append(query)

    snapshot_hash_violations = 0
    immutable_layout_hash_violations = 0
    query_layout_violations = 0
    candidate_identity_violations = 0
    checked_queries = 0
    for number, ((namespace_id, cutoff), boundary_queries) in enumerate(sorted(grouped.items()), start=1):
        snapshot = build_causal_snapshot(
            embeddings,
            memory_ids,
            namespaces,
            arrivals,
            namespace_id=namespace_id,
            cutoff_order=cutoff,
            **build_kwargs,
        )
        immutable = immutable_receipts[(namespace_id, cutoff)]
        snapshot_hash_violations += int(snapshot_fingerprint(snapshot) != immutable["snapshot_hash"])
        layout = balanced_graph_layout(
            snapshot,
            (),
            excluded_query_ids=set(),
            capacity=32,
            use_semantic=True,
            use_coretrieval=False,
            name="semantic",
        )
        immutable_layout_hash_violations += int(
            layout_fingerprint(layout, snapshot.memory_ids) != immutable["semantic_layout_hash"]
        )
        local_by_id = {memory_id: index for index, memory_id in enumerate(snapshot.memory_ids)}
        for query in boundary_queries:
            prior = prior_rows[query.query_id]
            candidate_ids = list(frontier[query.query_id]["frontier_ids"])
            candidate_identity_violations += int(candidate_ids != prior["candidate_ids"])
            local_positions = [local_by_id[value] for value in candidate_ids]
            current = layout_metrics(layout, local_positions)
            query_layout_violations += int(current != prior["layouts"]["semantic"]["logical"])
            checked_queries += 1
        print(f"verify layout snapshot {number}/{len(grouped)}", flush=True)

    result = {
        "schema_version": "method1-v30-gate-g-cross-run-layout-audit-1",
        "status": "CROSS_RUN_LAYOUT_EXACT" if not any(
            (
                snapshot_hash_violations,
                immutable_layout_hash_violations,
                query_layout_violations,
                candidate_identity_violations,
            )
        ) else "CROSS_RUN_LAYOUT_MISMATCH",
        "snapshot_count": len(grouped),
        "query_count": checked_queries,
        "snapshot_hash_violations": snapshot_hash_violations,
        "immutable_layout_hash_violations": immutable_layout_hash_violations,
        "query_tile_id_violations_vs_gate_e_run": query_layout_violations,
        "candidate_identity_violations_vs_gate_e_run": candidate_identity_violations,
        "inputs": observed,
        "runtime_seconds": time.perf_counter() - started,
        "fresh_authorized": False,
    }
    write_json(VERSION_ROOT / "artifacts" / "incremental_layout_crossrun_audit.json", result, exclusive=True)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "CROSS_RUN_LAYOUT_EXACT":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
