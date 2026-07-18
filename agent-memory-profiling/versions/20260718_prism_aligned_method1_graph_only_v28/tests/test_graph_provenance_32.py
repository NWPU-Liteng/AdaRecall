from __future__ import annotations

from src.causal_replay import run_graph_only_query
from src.streaming_fuzzy_hierarchy import build_causal_snapshot


def test_all_32_candidates_are_graph_owned_without_dense(config, synthetic_data) -> None:
    embeddings, ids, namespaces, arrivals = synthetic_data
    snapshot = build_causal_snapshot(embeddings, ids, namespaces, arrivals, namespace_id="a", cutoff_order=128)
    result, _ = run_graph_only_query(snapshot, embeddings[7], embeddings, config=config)
    assert len(result.memory_ids) == 32
    assert len(set(result.memory_ids)) == 32
    assert all(row.graph_owned and row.semantic_edge_ids for row in result.provenance)
    assert result.audit["graph_owned_slots"] == 32
    assert result.audit["dense_fallback_slots"] == 0
    assert result.audit["memory_ann_calls"] == 0
    assert result.audit["candidate_query_memory_similarity_calls"] == 0
