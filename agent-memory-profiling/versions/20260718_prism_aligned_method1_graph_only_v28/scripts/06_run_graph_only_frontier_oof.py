from __future__ import annotations

import argparse
import copy
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from common import V24_ROOT, V26_ROOT, VERSION_ROOT, iter_jsonl, load_yaml, write_json, write_jsonl
from src.baselines import BM25SnapshotIndex, HNSWSnapshotIndex, exact_dense, standard_ivf_prototype
from src.causal_replay import run_graph_only_query
from src.metrics import paired_bootstrap, win_loss_tie
from src.namespace_guard import assert_candidate_scope
from src.provenance import canonical_sha256, create_immutable_run, sha256_file
from src.streaming_fuzzy_hierarchy import build_causal_snapshot, snapshot_fingerprint
from v24.dataset import load_canonical
from v24.metrics import quality_metrics

METHODS = (
    "full_hnsw_128",
    "dense_hnsw_32",
    "standard_ivf_32",
    "lexical_bm25_32",
    "semantic_graph_only_32",
    "dual_graph_only_32",
)
GRAPH_METHODS = ("semantic_graph_only_32", "dual_graph_only_32")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--smoke-queries", type=int, default=0)
    parser.add_argument("--freeze-receipt", default="design_freeze_receipt.json")
    return parser.parse_args()


def load_access_events(query_by_id: dict[str, object]) -> list[dict]:
    keyed = {}
    for path in sorted((V26_ROOT / "runs" / "v26_train_oof_qwen").glob("shard_*.jsonl")):
        for row in iter_jsonl(path):
            if row["method"] == "full_n":
                keyed[row["query_id"]] = row
    if len(keyed) != 1355:
        raise RuntimeError(f"expected 1355 frozen-policy co-retrieval events, got {len(keyed)}")
    return [
        {
            "event_id": f"frozen-policy:{query_id}",
            "query_id": query_id,
            "namespace_id": row["namespace_id"],
            "arrival_order": int(query_by_id[query_id].cutoff_order),
            "memory_ids": list(row["reranked_ids"]),
            "event_kind": "co_retrieval",
            "native_agent_access": False,
        }
        for query_id, row in sorted(keyed.items())
    ]


def metric_fields(memory_ids: list[str], evaluation: object) -> dict[str, float]:
    values = quality_metrics(memory_ids, evaluation)
    return {
        "candidate_recall_frontier": float(values["candidate_recall"]),
        "evidence_group_coverage_frontier": float(values["evidence_group_coverage"]),
        "evidence_density": float(len(set(memory_ids) & set(evaluation.gold_memory_ids)) / max(len(memory_ids), 1)),
    }


def main() -> None:
    args = parse_args()
    config_path = VERSION_ROOT / "configs" / "graph_only_main.yaml"
    config = load_yaml(config_path)
    formal = args.smoke_queries <= 0
    if formal:
        receipt_path = VERSION_ROOT / "artifacts" / "frozen" / args.freeze_receipt
        if not receipt_path.is_file():
            raise RuntimeError("formal OOF requires the design freeze receipt")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt["status"] != "FROZEN_BEFORE_FORMAL_OOF" or receipt["config_sha256"] != sha256_file(config_path):
            raise RuntimeError("formal config differs from freeze receipt")
        source_manifest_path = VERSION_ROOT / "artifacts" / "frozen" / receipt.get("source_manifest", "source_manifest.json")
        frozen_sources = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        for row in frozen_sources["files"]:
            path = VERSION_ROOT / row["path"]
            if not path.is_file() or sha256_file(path) != row["sha256"]:
                raise RuntimeError(f"formal source differs from freeze: {row['path']}")
        suffix = "" if int(receipt.get("freeze_attempt", 1)) == 1 else f"_attempt_{int(receipt['freeze_attempt']):03d}"
        start_receipt = VERSION_ROOT / "artifacts" / "frozen" / f"formal_oof_start_receipt{suffix}.json"
        start_receipt.parent.mkdir(parents=True, exist_ok=True)
        with start_receipt.open("x", encoding="utf-8") as handle:
            json.dump({"run_id": args.run_id, "started_at_unix": time.time(), "formal_run_number": 1}, handle, indent=2, sort_keys=True)
            handle.write("\n")
    run_dir = create_immutable_run(VERSION_ROOT, args.run_id)
    started = time.perf_counter()
    dataset = load_canonical(V24_ROOT / "inputs", split="train")
    queries = list(dataset.queries["train"])
    if args.smoke_queries > 0:
        queries = queries[: args.smoke_queries]
    all_query_by_id = {query.query_id: query for query in dataset.queries["train"]}
    fold_manifest_path = V26_ROOT / "artifacts" / "manifests" / "oof_fold_manifest.json"
    fold_manifest = json.loads(fold_manifest_path.read_text(encoding="utf-8"))
    fold_by_id = {query_id: int(fold) for fold, ids in fold_manifest["fold_query_ids"].items() for query_id in ids}
    heldout_by_fold = {fold: {query_id for query_id, value in fold_by_id.items() if value == fold} for fold in range(5)}
    if formal and set(fold_by_id) != set(all_query_by_id):
        raise RuntimeError("fold manifest does not cover canonical train queries exactly")

    embeddings = np.asarray(dataset.embeddings, dtype=np.float32).copy()
    embeddings /= np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
    memory_ids = dataset.memory_ids
    namespaces = [memory.namespace_id for memory in dataset.memories]
    arrivals = np.asarray([memory.arrival_order for memory in dataset.memories], dtype=np.int64)
    contents = [memory.content for memory in dataset.memories]
    memory_namespace = {memory.memory_id: memory.namespace_id for memory in dataset.memories}
    memory_arrival = {memory.memory_id: memory.arrival_order for memory in dataset.memories}
    query_embeddings = {query.query_id: np.asarray(dataset.query_vector("train", query.query_id), dtype=np.float32) for query in dataset.queries["train"]}
    access_events = load_access_events(all_query_by_id)

    grouped: dict[tuple[str, int], list[object]] = defaultdict(list)
    for query in queries:
        grouped[(query.namespace_id, int(query.cutoff_order))].append(query)
    semantic_config = copy.deepcopy(config)
    semantic_config["history"]["history_weight"] = 0.0
    semantic_config["history"]["agreement_weight"] = 0.0

    rows = []
    snapshot_rows = []
    exact_hnsw_recall = []
    for snapshot_number, ((namespace_id, cutoff), snapshot_queries) in enumerate(sorted(grouped.items()), start=1):
        build_started = time.perf_counter_ns()
        semantic_cfg = config["semantic"]
        snapshot = build_causal_snapshot(
            embeddings,
            memory_ids,
            namespaces,
            arrivals,
            namespace_id=namespace_id,
            cutoff_order=cutoff,
            memberships_per_memory=int(semantic_cfg["memberships_per_memory"]),
            temperature=float(semantic_cfg["membership_temperature"]),
            medoids_per_edge=int(semantic_cfg["medoids_per_edge"]),
            max_degree_before_split=int(semantic_cfg["max_degree_before_split"]),
            split_children=int(semantic_cfg["split_children"]),
            merge_similarity=float(semantic_cfg["merge_similarity"]),
            minimum_merge_age=int(semantic_cfg["minimum_merge_age"]),
            seed=int(config["random_seed"]),
        )
        build_ms = (time.perf_counter_ns() - build_started) / 1e6
        hnsw_started = time.perf_counter_ns()
        hnsw = HNSWSnapshotIndex(snapshot)
        hnsw_build_ms = (time.perf_counter_ns() - hnsw_started) / 1e6
        local_contents = [contents[int(position)] for position in snapshot.positions]
        bm25_started = time.perf_counter_ns()
        bm25 = BM25SnapshotIndex(local_contents, snapshot.memory_ids)
        bm25_build_ms = (time.perf_counter_ns() - bm25_started) / 1e6
        snapshot_rows.append(
            {
                "namespace_id": namespace_id,
                "cutoff_order": cutoff,
                "query_count": len(snapshot_queries),
                "snapshot_hash": snapshot_fingerprint(snapshot),
                "graph_build_ms": build_ms,
                "hnsw_build_ms": hnsw_build_ms,
                "bm25_build_ms": bm25_build_ms,
                "health": snapshot.health,
            }
        )
        print(f"snapshot {snapshot_number}/{len(grouped)} ns={namespace_id} cutoff={cutoff} q={len(snapshot_queries)} visible={len(snapshot.positions)}", flush=True)
        for query in snapshot_queries:
            query_vector = query_embeddings[query.query_id]
            fold = fold_by_id[query.query_id]
            evaluation = dataset.evaluations["train"][query.query_id]

            sem_started = time.perf_counter_ns()
            semantic, semantic_audit = run_graph_only_query(
                snapshot,
                query_vector,
                embeddings,
                config=semantic_config,
                history_events=access_events,
                excluded_query_ids=heldout_by_fold[fold],
                query_embeddings_by_id=query_embeddings,
            )
            semantic_ms = (time.perf_counter_ns() - sem_started) / 1e6
            dual_started = time.perf_counter_ns()
            dual, dual_audit = run_graph_only_query(
                snapshot,
                query_vector,
                embeddings,
                config=config,
                history_events=access_events,
                excluded_query_ids=heldout_by_fold[fold],
                query_embeddings_by_id=query_embeddings,
            )
            dual_ms = (time.perf_counter_ns() - dual_started) / 1e6

            ann_started = time.perf_counter_ns()
            full = hnsw.search(query_vector, min(int(config["budget"]["full_n"]), len(snapshot.positions)))
            ann_ms = (time.perf_counter_ns() - ann_started) / 1e6
            dense_ids = full.memory_ids[: min(int(config["budget"]["frontier_c"]), len(full.memory_ids))]
            dense_positions = full.local_positions[: len(dense_ids)]
            exact = exact_dense(snapshot, query_vector, min(32, len(snapshot.positions)))
            exact_hnsw_recall.append(len(set(dense_ids) & set(exact.memory_ids)) / max(len(exact.memory_ids), 1))
            ivf_started = time.perf_counter_ns()
            ivf = standard_ivf_prototype(
                snapshot,
                query_vector,
                active_cells=len(semantic_audit.activation.active_columns),
                pool_cap=int(config["budget"]["expanded_pool_cap"]),
                budget=min(32, len(snapshot.positions)),
            )
            ivf_ms = (time.perf_counter_ns() - ivf_started) / 1e6
            lexical_started = time.perf_counter_ns()
            lexical = bm25.search(query.query, min(32, len(snapshot.positions)))
            lexical_ms = (time.perf_counter_ns() - lexical_started) / 1e6

            outputs = {
                "full_hnsw_128": (full.memory_ids, full.local_positions, ann_ms, None),
                "dense_hnsw_32": (dense_ids, dense_positions, ann_ms, None),
                "standard_ivf_32": (ivf.memory_ids, ivf.local_positions, ivf_ms, None),
                "lexical_bm25_32": (lexical.memory_ids, lexical.local_positions, lexical_ms, None),
                "semantic_graph_only_32": (semantic.memory_ids, semantic.local_positions, semantic_ms, semantic),
                "dual_graph_only_32": (dual.memory_ids, dual.local_positions, dual_ms, dual),
            }
            for method, (selected_ids, selected_positions, latency_ms, graph_result) in outputs.items():
                safety = assert_candidate_scope(
                    selected_ids,
                    memory_namespace,
                    memory_arrival,
                    namespace_id=namespace_id,
                    cutoff_order=cutoff,
                )
                expected = min(128 if method == "full_hnsw_128" else 32, len(snapshot.positions))
                safety["candidate_count_mismatch"] = int(len(selected_ids) != expected)
                if graph_result is not None:
                    safety["graph_pool_underflow"] = int(graph_result.underflow)
                row = {
                    "query_id": query.query_id,
                    "query_type": query.metadata.get("query_type", "unknown"),
                    "namespace_id": namespace_id,
                    "namespace_source": "dataset_provided",
                    "cutoff_order": cutoff,
                    "fold": fold,
                    "method": method,
                    "frontier_ids": selected_ids,
                    "metrics": metric_fields(selected_ids, evaluation),
                    "safety": safety,
                    "latency": {"retrieval_or_graph_ms": latency_ms, "query_encoding_ms": 0.0, "query_embedding_cache": True},
                    "graph_audit": graph_result.audit if graph_result is not None else None,
                    "graph_provenance": [row.__dict__ for row in graph_result.provenance] if graph_result is not None else None,
                    "run_id": args.run_id,
                    "scientific_scope": "train strict five-fold OOF development evidence" if formal else "pre-freeze smoke no promotion",
                }
                rows.append(row)

    write_jsonl(run_dir / "per_query_frontier.jsonl", rows)
    write_json(run_dir / "causal_snapshot_manifest.json", {"snapshots": snapshot_rows})
    by_method = {method: {row["query_id"]: row for row in rows if row["method"] == method} for method in METHODS}
    summary = {}
    baseline = by_method["dense_hnsw_32"]
    gate = config["gate"]
    for method in METHODS:
        method_rows = list(by_method[method].values())
        absolute = {
            metric: float(np.mean([row["metrics"][metric] for row in method_rows]))
            for metric in ("candidate_recall_frontier", "evidence_group_coverage_frontier", "evidence_density")
        }
        result = {"absolute": absolute, "query_count": len(method_rows), "safety_violations": int(sum(sum(row["safety"].values()) for row in method_rows))}
        if method != "dense_hnsw_32":
            result["paired_vs_dense"] = {}
            for metric in ("candidate_recall_frontier", "evidence_group_coverage_frontier"):
                deltas = [row["metrics"][metric] - baseline[row["query_id"]]["metrics"][metric] for row in method_rows]
                result["paired_vs_dense"][metric] = {
                    **paired_bootstrap(deltas, seed=int(config["random_seed"]), samples=int(gate["bootstrap_samples"])),
                    **win_loss_tie(deltas, float(gate["per_query_difference_threshold"])),
                }
            result["candidate_delta_by_fold"] = {
                str(fold): float(np.mean([
                    row["metrics"]["candidate_recall_frontier"] - baseline[row["query_id"]]["metrics"]["candidate_recall_frontier"]
                    for row in method_rows if row["fold"] == fold
                ]))
                for fold in range(5)
            }
        if method in GRAPH_METHODS:
            graph_owned = [row["graph_audit"].get("graph_owned_slots", 0) for row in method_rows]
            result["graph_owned_slots_mean"] = float(np.mean(graph_owned))
            result["dense_fallback_slots"] = int(sum(row["graph_audit"]["dense_fallback_slots"] for row in method_rows))
            recall = result["paired_vs_dense"]["candidate_recall_frontier"]
            coverage = result["paired_vs_dense"]["evidence_group_coverage_frontier"]
            result["gate_c_passed"] = bool(
                recall["ci_95"][0] >= float(gate["candidate_recall_ci_lower_min"])
                and coverage["ci_95"][0] >= float(gate["group_coverage_ci_lower_min"])
                and sum(value >= -0.01 for value in result["candidate_delta_by_fold"].values()) >= int(gate["noninferior_fold_count_min"])
                and recall["loss_rate"] - recall["win_rate"] <= float(gate["loss_rate_minus_win_rate_max"])
                and result["dense_fallback_slots"] == 0
                and result["safety_violations"] == 0
            )
        summary[method] = result
    aggregate = {
        "schema_version": "method1-v28-frontier-oof-1",
        "run_id": args.run_id,
        "status": "FRONTIER_OOF_COMPLETE" if formal else "SMOKE_COMPLETE_NO_PROMOTION",
        "formal": formal,
        "query_count": len(queries),
        "method_rows": len(rows),
        "snapshot_count": len(snapshot_rows),
        "runtime_seconds": time.perf_counter() - started,
        "hnsw_recall_vs_exact_at_32_mean": float(np.mean(exact_hnsw_recall)),
        "config_sha256": sha256_file(config_path),
        "freeze_receipt": args.freeze_receipt if formal else None,
        "freeze_receipt_sha256": sha256_file(VERSION_ROOT / "artifacts" / "frozen" / args.freeze_receipt) if formal else None,
        "fold_manifest_sha256": sha256_file(fold_manifest_path),
        "rows_sha256": sha256_file(run_dir / "per_query_frontier.jsonl"),
        "snapshot_manifest_sha256": sha256_file(run_dir / "causal_snapshot_manifest.json"),
        "methods": summary,
        "gate_c_promoted": [method for method in GRAPH_METHODS if summary[method].get("gate_c_passed")],
    }
    aggregate["aggregate_payload_sha256"] = canonical_sha256(aggregate)
    write_json(run_dir / "aggregate_metrics.json", aggregate)
    print(json.dumps({"status": aggregate["status"], "gate_c_promoted": aggregate["gate_c_promoted"], "runtime_seconds": aggregate["runtime_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
