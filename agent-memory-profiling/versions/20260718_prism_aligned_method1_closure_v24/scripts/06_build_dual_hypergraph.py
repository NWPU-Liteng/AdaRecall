from __future__ import annotations

import argparse
import json

import numpy as np
from scipy import sparse

from common import VERSION_ROOT, add_common_args, artifact_root, config_hash, load_config
from v24.contracts import HistoryEvent
from v24.dataset import load_canonical
from v24.history_hypergraph import build_history_edges
from v24.io import iter_jsonl, write_json
from v24.semantic_hypergraph import fit_spherical_soft


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--index-run-id", default="v24_index_20260718")
    args = parser.parse_args()
    config = load_config(args.config)
    data = load_canonical(VERSION_ROOT / "inputs")
    output = artifact_root(args.index_run_id)
    sem_config = config["semantic_hypergraph"]
    semantic = fit_spherical_soft(
        data.embeddings, n_prototypes=int(sem_config["n_prototypes"]), top_memberships=int(sem_config["top_memberships"]),
        temperature=float(sem_config["temperature"]), seed=int(config["random_seed"]),
    )
    write_json(VERSION_ROOT / "artifacts" / "hypergraph" / "semantic_hypergraph_stats.json", {
        "config_hash": config_hash(config), "diagnostics": semantic.diagnostics,
        "construction": "soft_spherical_kmeans", "labels_used": False,
    })
    if not semantic.diagnostics["health_passed"]:
        write_json(VERSION_ROOT / "negative_results" / "semantic_health_failure.json", semantic.diagnostics)
        raise RuntimeError(f"semantic hypergraph health gate failed: {semantic.diagnostics}")
    np.save(output / "semantic_prototypes.npy", semantic.prototypes, allow_pickle=False)
    sparse.save_npz(output / "semantic_incidence.npz", semantic.incidence)
    write_json(output / "memory_ids.json", data.memory_ids)
    events = [HistoryEvent(
        event_id=str(row["event_id"]), namespace_id=str(row["namespace_id"]), query_id=str(row["query_id"]),
        query_embedding_id=str(row["query_embedding_id"]), selected_memory_ids=tuple(str(value) for value in row["selected_memory_ids"]),
        event_kind=str(row["event_kind"]), task_success=row.get("task_success"), arrival_order=int(row["arrival_order"]),
    ) for row in iter_jsonl(VERSION_ROOT / "inputs" / "history_events.jsonl")]
    query_vectors = data.query_vectors["train"]
    history = build_history_edges(events, set(data.memory_ids), query_vectors, max_event_size=int(config["history_hypergraph"]["max_event_size"]))
    (output / "history_edges.json").write_text(json.dumps(history.edges, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    np.save(output / "history_edge_vectors.npy", history.vectors, allow_pickle=False)
    write_json(output / "history_inverted.json", history.inverted)
    write_json(VERSION_ROOT / "artifacts" / "hypergraph" / "history_hypergraph_stats.json", {"config_hash": config_hash(config), "diagnostics": history.diagnostics, "gold_used": False, "test_used": False})
    manifest = {
        "index_run_id": args.index_run_id, "config_hash": config_hash(config),
        "semantic": semantic.diagnostics, "history": history.diagnostics,
        "artifacts": {"semantic_prototypes": str(output / "semantic_prototypes.npy"), "semantic_incidence": str(output / "semantic_incidence.npz"), "history_edges": str(output / "history_edges.json")},
    }
    write_json(output / "index_manifest.json", manifest)
    write_json(VERSION_ROOT / "artifacts" / "manifests" / "hypergraph_manifest.json", manifest)
    print(json.dumps({"status": "completed", "index_run_id": args.index_run_id, "semantic": semantic.diagnostics, "history": history.diagnostics}, sort_keys=True))


if __name__ == "__main__":
    main()
