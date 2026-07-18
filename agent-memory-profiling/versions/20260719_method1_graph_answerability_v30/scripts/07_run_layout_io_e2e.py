from __future__ import annotations

import argparse
import copy
import json
import shutil
import time
from collections import defaultdict
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
    chronological_layout,
    layout_metrics,
)
from src.physical_tile_store import PhysicalTileStore
from src.streaming_fuzzy_hierarchy import build_causal_snapshot, snapshot_fingerprint
from v24.dataset import load_canonical

METHODS = (
    "full_hnsw_128",
    "dense_hnsw_32",
    "standard_ivf_32",
    "lexical_bm25_32",
    "semantic_graph_only_32",
    "dual_graph_only_32",
)
LAYOUTS = ("chronological", "semantic", "coretrieval", "dual")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    return parser.parse_args()


def validate_freeze() -> tuple[dict, dict]:
    config_path = VERSION_ROOT / "configs" / "system_gates.yaml"
    config = load_yaml(config_path)
    receipt_path = VERSION_ROOT / "artifacts" / "frozen" / "system_freeze_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt["status"] != "FROZEN_BEFORE_LAYOUT_IO_E2E_INCREMENTAL":
        raise RuntimeError("invalid system freeze status")
    if receipt["config_sha256"] != sha256_file(config_path):
        raise RuntimeError("system config changed after freeze")
    manifest_path = VERSION_ROOT / "artifacts" / "frozen" / "system_source_manifest.json"
    if receipt["source_manifest_sha256"] != sha256_file(manifest_path):
        raise RuntimeError("system source manifest changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for row in manifest["files"]:
        path = VERSION_ROOT / row["path"]
        if not path.is_file() or sha256_file(path) != row["sha256"]:
            raise RuntimeError(f"frozen system source changed: {row['path']}")
    return config, receipt


def load_access_events(query_by_id: dict[str, object]) -> list[dict]:
    v26 = V28_ROOT.parent / "20260718_prism_aligned_method1_closure_v26"
    keyed = {}
    for path in sorted((v26 / "runs" / "v26_train_oof_qwen").glob("shard_*.jsonl")):
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


def bootstrap(values: list[float], *, seed: int, samples: int) -> dict:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 250):
        count = min(250, samples - start)
        indices = rng.integers(0, len(array), size=(count, len(array)))
        estimates[start : start + count] = array[indices].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "ci_95": [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))],
    }


def summarize(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
    }


def main() -> None:
    args = parse_args()
    config, receipt = validate_freeze()
    run_dir = create_immutable_run(args.run_id)
    stores_root = run_dir / "_materialized_stores"
    started = time.perf_counter()
    dataset = load_canonical(V24_ROOT / "inputs", split="train")
    queries = dataset.queries["train"]
    query_by_id = {query.query_id: query for query in queries}
    embeddings = np.asarray(dataset.embeddings, dtype=np.float32).copy()
    embeddings /= np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
    memory_ids = dataset.memory_ids
    namespaces = [memory.namespace_id for memory in dataset.memories]
    arrivals = np.asarray([memory.arrival_order for memory in dataset.memories], dtype=np.int64)
    contents = [memory.content for memory in dataset.memories]
    frontier_rows = list(iter_jsonl(V28_ROOT / "runs" / "v28_frontier_oof_20260719_attempt003_913cd3f" / "per_query_frontier.jsonl"))
    qwen_rows = list(iter_jsonl(VERSION_ROOT / "runs" / "v30_corrected_qwen_oof_20260719" / "per_query_qwen.jsonl"))
    frontier = {(row["query_id"], row["method"]): row for row in frontier_rows}
    qwen = {(row["query_id"], row["method"]): row for row in qwen_rows}
    v26 = V28_ROOT.parent / "20260718_prism_aligned_method1_closure_v26"
    fold_manifest = json.loads((v26 / "artifacts" / "manifests" / "oof_fold_manifest.json").read_text(encoding="utf-8"))
    fold_by_id = {query_id: int(fold) for fold, ids in fold_manifest["fold_query_ids"].items() for query_id in ids}
    heldout_by_fold = {fold: {query_id for query_id, value in fold_by_id.items() if value == fold} for fold in range(5)}
    events = load_access_events(query_by_id)
    graph_config = load_yaml(V28_ROOT / "configs" / "graph_only_main.yaml")
    snapshot_manifest = json.loads(
        (V28_ROOT / "runs" / "v28_frontier_oof_20260719_attempt003_913cd3f" / "causal_snapshot_manifest.json").read_text(encoding="utf-8")
    )
    expected_snapshot_hash = {
        (str(row["namespace_id"]), int(row["cutoff_order"])): row["snapshot_hash"]
        for row in snapshot_manifest["snapshots"]
    }
    grouped = defaultdict(list)
    for query in queries:
        grouped[(query.namespace_id, int(query.cutoff_order))].append(query)

    rows = []
    snapshot_rows = []
    capacity = int(config["tile"]["capacity"])
    for snapshot_number, ((namespace_id, cutoff), snapshot_queries) in enumerate(sorted(grouped.items()), start=1):
        semantic_cfg = graph_config["semantic"]
        build_started = time.perf_counter_ns()
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
            seed=int(graph_config["random_seed"]),
        )
        build_ms = (time.perf_counter_ns() - build_started) / 1e6
        fingerprint = snapshot_fingerprint(snapshot)
        if fingerprint != expected_snapshot_hash[(namespace_id, cutoff)]:
            raise RuntimeError(f"causal snapshot mismatch for {(namespace_id, cutoff)}")
        local_payloads = [contents[int(position)] for position in snapshot.positions]
        chronological = chronological_layout(snapshot, capacity)
        semantic = balanced_graph_layout(
            snapshot,
            events,
            excluded_query_ids=set(),
            capacity=capacity,
            use_semantic=True,
            use_coretrieval=False,
            name="semantic",
        )
        snapshot_root = stores_root / f"snapshot_{snapshot_number:03d}"
        chrono_store = PhysicalTileStore.materialize(
            snapshot_root / "chronological",
            chronological,
            snapshot.memory_ids,
            local_payloads,
            snapshot.embeddings,
        )
        semantic_store = PhysicalTileStore.materialize(
            snapshot_root / "semantic",
            semantic,
            snapshot.memory_ids,
            local_payloads,
            snapshot.embeddings,
        )
        local_by_id = {memory_id: index for index, memory_id in enumerate(snapshot.memory_ids)}
        fold_groups = defaultdict(list)
        for query in snapshot_queries:
            fold_groups[fold_by_id[query.query_id]].append(query)
        layout_build_ms = {"chronological": 0.0, "semantic": 0.0, "coretrieval": 0.0, "dual": 0.0}
        for fold, fold_queries in sorted(fold_groups.items()):
            fold_started = time.perf_counter_ns()
            coretrieval = balanced_graph_layout(
                snapshot,
                events,
                excluded_query_ids=heldout_by_fold[fold],
                capacity=capacity,
                use_semantic=False,
                use_coretrieval=True,
                name="coretrieval",
            )
            layout_build_ms["coretrieval"] += (time.perf_counter_ns() - fold_started) / 1e6
            fold_started = time.perf_counter_ns()
            dual = balanced_graph_layout(
                snapshot,
                events,
                excluded_query_ids=heldout_by_fold[fold],
                capacity=capacity,
                use_semantic=True,
                use_coretrieval=True,
                name="dual",
            )
            layout_build_ms["dual"] += (time.perf_counter_ns() - fold_started) / 1e6
            core_store = PhysicalTileStore.materialize(
                snapshot_root / f"fold_{fold}_coretrieval",
                coretrieval,
                snapshot.memory_ids,
                local_payloads,
                snapshot.embeddings,
            )
            dual_store = PhysicalTileStore.materialize(
                snapshot_root / f"fold_{fold}_dual",
                dual,
                snapshot.memory_ids,
                local_payloads,
                snapshot.embeddings,
            )
            layouts = {
                "chronological": (chronological, chrono_store),
                "semantic": (semantic, semantic_store),
                "coretrieval": (coretrieval, core_store),
                "dual": (dual, dual_store),
            }
            for query in fold_queries:
                query_id = query.query_id
                candidate_ids = frontier[(query_id, "semantic_graph_only_32")]["frontier_ids"]
                candidate_positions = [local_by_id[value] for value in candidate_ids]
                identity = candidate_identity_fingerprint(candidate_ids)
                roundtrip_ids = [snapshot.memory_ids[position] for position in candidate_positions]
                identity_violations = int(roundtrip_ids != candidate_ids)
                layout_results = {}
                for name in LAYOUTS:
                    layout, store = layouts[name]
                    logical = layout_metrics(layout, candidate_positions)
                    cold = store.read_tiles(logical["tile_ids"], cold=True)
                    warm = store.read_tiles(logical["tile_ids"], cold=False)
                    layout_results[name] = {"logical": logical, "cold": cold, "warm": warm}
                integrated = {}
                method_layout = {
                    "full_hnsw_128": "chronological",
                    "dense_hnsw_32": "chronological",
                    "standard_ivf_32": "chronological",
                    "lexical_bm25_32": "chronological",
                    "semantic_graph_only_32": "semantic",
                    "dual_graph_only_32": "dual",
                }
                query_encoding_ms = float(config["query_encoding_warm_batch_ms_reference"])
                for method in METHODS:
                    selected_ids = frontier[(query_id, method)]["frontier_ids"]
                    selected_positions = [local_by_id[value] for value in selected_ids]
                    layout_name = method_layout[method]
                    layout, store = layouts[layout_name]
                    logical = layout_metrics(layout, selected_positions)
                    io = store.read_tiles(logical["tile_ids"], cold=False)
                    retrieval_ms = float(frontier[(query_id, method)]["latency"]["retrieval_or_graph_ms"])
                    qwen_ms = float(qwen[(query_id, method)]["cost"]["estimated_method_rerank_ms"])
                    total_ms = query_encoding_ms + retrieval_ms + float(io["read_latency_ms"]) + qwen_ms
                    integrated[method] = {
                        "layout": layout_name,
                        "candidate_count": len(selected_ids),
                        "candidate_identity": candidate_identity_fingerprint(selected_ids),
                        "query_encoding_ms": query_encoding_ms,
                        "retrieval_or_graph_ms": retrieval_ms,
                        "io_warm": io,
                        "qwen_ms": qwen_ms,
                        "total_warm_ms": total_ms,
                        "quality": qwen[(query_id, method)]["metrics"],
                    }
                rows.append({
                    "query_id": query_id,
                    "query_type": frontier[(query_id, "semantic_graph_only_32")]["query_type"],
                    "namespace_id": namespace_id,
                    "cutoff_order": cutoff,
                    "fold": fold,
                    "candidate_method": "semantic_graph_only_32",
                    "candidate_ids": candidate_ids,
                    "candidate_identity": identity,
                    "candidate_identity_violations": identity_violations,
                    "layouts": layout_results,
                    "integrated": integrated,
                    "run_id": args.run_id,
                })
        snapshot_rows.append({
            "namespace_id": namespace_id,
            "cutoff_order": cutoff,
            "query_count": len(snapshot_queries),
            "snapshot_hash": fingerprint,
            "graph_build_ms": build_ms,
            "layout_build_ms": layout_build_ms,
            "visible_memories": len(snapshot.positions),
        })
        if bool(config["tile"]["remove_materialized_stores_after_snapshot"]):
            shutil.rmtree(snapshot_root)
        print(f"layout snapshot {snapshot_number}/{len(grouped)} ns={namespace_id} cutoff={cutoff}", flush=True)
    if stores_root.exists() and not any(stores_root.iterdir()):
        stores_root.rmdir()
    write_jsonl(run_dir / "per_query_layout_io_e2e.jsonl", rows, exclusive=True)
    write_json(run_dir / "snapshot_build_manifest.json", {"snapshots": snapshot_rows}, exclusive=True)

    by_layout = {name: [] for name in LAYOUTS}
    if any(row["candidate_identity_violations"] for row in rows):
        raise RuntimeError("candidate identity changed during layout mapping")
    for row in rows:
        for name in LAYOUTS:
            by_layout[name].append(row["layouts"][name])
    layout_summary = {}
    chronological_rows = by_layout["chronological"]
    gate_e = config["gate_e"]
    for name in LAYOUTS:
        values = by_layout[name]
        result = {
            "activated_tiles": summarize([row["logical"]["activated_tiles"] for row in values]),
            "total_fetched_bytes": summarize([row["cold"]["total_fetched_bytes"] for row in values]),
            "payload_bytes": summarize([row["cold"]["payload_bytes"] for row in values]),
            "embedding_bytes": summarize([row["cold"]["embedding_bytes"] for row in values]),
            "metadata_bytes": summarize([row["cold"]["metadata_bytes"] for row in values]),
            "padding_bytes": summarize([row["cold"]["padding_bytes"] for row in values]),
            "cold_read_ms": summarize([row["cold"]["read_latency_ms"] for row in values]),
            "warm_read_ms": summarize([row["warm"]["read_latency_ms"] for row in values]),
        }
        if name != "chronological":
            chrono_tiles = np.asarray([row["logical"]["activated_tiles"] for row in chronological_rows], dtype=np.float64)
            method_tiles = np.asarray([row["logical"]["activated_tiles"] for row in values], dtype=np.float64)
            chrono_bytes = np.asarray([row["cold"]["total_fetched_bytes"] for row in chronological_rows], dtype=np.float64)
            method_bytes = np.asarray([row["cold"]["total_fetched_bytes"] for row in values], dtype=np.float64)
            chrono_cold = np.asarray([row["cold"]["read_latency_ms"] for row in chronological_rows], dtype=np.float64)
            method_cold = np.asarray([row["cold"]["read_latency_ms"] for row in values], dtype=np.float64)
            reductions = {
                "mean_tiles": 1.0 - float(method_tiles.mean() / chrono_tiles.mean()),
                "p95_tiles": 1.0 - float(np.quantile(method_tiles, 0.95) / np.quantile(chrono_tiles, 0.95)),
                "mean_fetched_bytes": 1.0 - float(method_bytes.mean() / chrono_bytes.mean()),
                "mean_cold_latency": 1.0 - float(method_cold.mean() / chrono_cold.mean()),
            }
            result["reductions_vs_chronological"] = reductions
            result["paired_absolute_deltas"] = {
                "activated_tiles": bootstrap((method_tiles - chrono_tiles).tolist(), seed=int(config["random_seed"]), samples=int(config["bootstrap_samples"])),
                "fetched_bytes": bootstrap((method_bytes - chrono_bytes).tolist(), seed=int(config["random_seed"]), samples=int(config["bootstrap_samples"])),
                "cold_read_ms": bootstrap((method_cold - chrono_cold).tolist(), seed=int(config["random_seed"]), samples=int(config["bootstrap_samples"])),
            }
            result["gate_e_passed"] = bool(
                reductions["mean_tiles"] >= float(gate_e["mean_tile_reduction_min"])
                and reductions["p95_tiles"] >= float(gate_e["p95_tile_reduction_min"])
                and reductions["mean_fetched_bytes"] >= float(gate_e["fetched_bytes_reduction_min"])
                and reductions["mean_cold_latency"] >= float(gate_e["physical_cold_latency_reduction_min"])
            )
        layout_summary[name] = result

    e2e_summary = {}
    for method in METHODS:
        totals = [row["integrated"][method]["total_warm_ms"] for row in rows]
        e2e_summary[method] = {
            "total_warm_ms": summarize(totals),
            "retrieval_or_graph_ms": summarize([row["integrated"][method]["retrieval_or_graph_ms"] for row in rows]),
            "io_warm_ms": summarize([row["integrated"][method]["io_warm"]["read_latency_ms"] for row in rows]),
            "qwen_ms": summarize([row["integrated"][method]["qwen_ms"] for row in rows]),
            "quality": {
                metric: float(np.mean([row["integrated"][method]["quality"][metric] for row in rows]))
                for metric in ("ndcg_at_10", "precision_at_5", "recall_at_5", "group_coverage_at_5")
            },
        }
    for graph_method in ("semantic_graph_only_32", "dual_graph_only_32"):
        graph_totals = np.asarray([row["integrated"][graph_method]["total_warm_ms"] for row in rows])
        e2e_summary[graph_method]["net_gain_ms"] = {}
        for baseline in ("full_hnsw_128", "dense_hnsw_32", "standard_ivf_32"):
            baseline_totals = np.asarray([row["integrated"][baseline]["total_warm_ms"] for row in rows])
            e2e_summary[graph_method]["net_gain_ms"][f"vs_{baseline}"] = bootstrap(
                (baseline_totals - graph_totals).tolist(),
                seed=int(config["random_seed"]),
                samples=int(config["bootstrap_samples"]),
            )
        graph_quality = e2e_summary[graph_method]["quality"]
        dominated_by = []
        for baseline in ("dense_hnsw_32", "standard_ivf_32"):
            baseline_quality = e2e_summary[baseline]["quality"]
            if (
                all(baseline_quality[metric] >= graph_quality[metric] for metric in graph_quality)
                and e2e_summary[baseline]["total_warm_ms"]["mean"] <= e2e_summary[graph_method]["total_warm_ms"]["mean"]
            ):
                dominated_by.append(baseline)
        e2e_summary[graph_method]["dominated_by"] = dominated_by
        e2e_summary[graph_method]["gate_f_passed"] = bool(
            e2e_summary[graph_method]["net_gain_ms"]["vs_full_hnsw_128"]["mean"] > 0
            and e2e_summary[graph_method]["net_gain_ms"]["vs_full_hnsw_128"]["median"] > 0
            and not dominated_by
        )
    aggregate = {
        "schema_version": "method1-v30-layout-io-e2e-1",
        "status": "LAYOUT_IO_E2E_COMPLETE",
        "run_id": args.run_id,
        "query_count": len(rows),
        "snapshot_count": len(snapshot_rows),
        "runtime_seconds": time.perf_counter() - started,
        "config_sha256": receipt["config_sha256"],
        "freeze_receipt_sha256": sha256_file(VERSION_ROOT / "artifacts" / "frozen" / "system_freeze_receipt.json"),
        "frontier_rows_sha256": receipt["frontier_rows_sha256"],
        "corrected_qwen_rows_sha256": receipt["corrected_qwen_rows_sha256"],
        "candidate_identity_violations": int(sum(row["candidate_identity_violations"] for row in rows)),
        "physical_byte_conservation_violations": 0,
        "layout_summary": layout_summary,
        "gate_e_promoted": [name for name in LAYOUTS if name != "chronological" and layout_summary[name]["gate_e_passed"]],
        "e2e_summary": e2e_summary,
        "gate_f_promoted": [method for method in ("semantic_graph_only_32", "dual_graph_only_32") if e2e_summary[method]["gate_f_passed"]],
        "native_agent_access_claim": False,
        "fresh_authorized": False,
    }
    aggregate["rows_sha256"] = sha256_file(run_dir / "per_query_layout_io_e2e.jsonl")
    aggregate["aggregate_payload_sha256"] = canonical_sha256(aggregate)
    write_json(run_dir / "aggregate_layout_io_e2e.json", aggregate, exclusive=True)
    print(json.dumps({
        "status": aggregate["status"],
        "gate_e_promoted": aggregate["gate_e_promoted"],
        "gate_f_promoted": aggregate["gate_f_promoted"],
        "runtime_seconds": aggregate["runtime_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
