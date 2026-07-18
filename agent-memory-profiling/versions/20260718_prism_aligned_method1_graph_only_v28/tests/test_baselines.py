from __future__ import annotations

from src.baselines import BM25SnapshotIndex, HNSWSnapshotIndex, exact_dense, standard_ivf_prototype
from src.streaming_fuzzy_hierarchy import build_causal_snapshot
from src.causal_replay import run_graph_only_query


def test_baselines_are_isolated_and_return_budget(synthetic_data) -> None:
    embeddings, ids, namespaces, arrivals = synthetic_data
    snapshot = build_causal_snapshot(embeddings, ids, namespaces, arrivals, namespace_id="a", cutoff_order=128)
    query = embeddings[3]
    exact = exact_dense(snapshot, query, 16)
    hnsw = HNSWSnapshotIndex(snapshot).search(query, 16)
    ivf = standard_ivf_prototype(snapshot, query, active_cells=8, pool_cap=64, budget=16)
    bm25 = BM25SnapshotIndex([f"medical record {value}" for value in snapshot.memory_ids], snapshot.memory_ids).search("medical record m003", 16)
    assert all(len(result.memory_ids) == 16 for result in (exact, hnsw, ivf, bm25))
    assert exact.query_memory_similarity_calls == len(snapshot.positions)
    assert hnsw.ann_calls == 1
    assert ivf.query_memory_similarity_calls > 0


def test_read_only_query_views_are_supported(config, synthetic_data) -> None:
    embeddings, ids, namespaces, arrivals = synthetic_data
    snapshot = build_causal_snapshot(embeddings, ids, namespaces, arrivals, namespace_id="a", cutoff_order=128)
    query = embeddings[3].copy()
    query.flags.writeable = False
    graph, _ = run_graph_only_query(snapshot, query, embeddings, config=config)
    exact = exact_dense(snapshot, query, 16)
    hnsw = HNSWSnapshotIndex(snapshot).search(query, 16)
    ivf = standard_ivf_prototype(snapshot, query, active_cells=8, pool_cap=64, budget=16)
    assert len(graph.memory_ids) == 32
    assert all(len(result.memory_ids) == 16 for result in (exact, hnsw, ivf))
