from __future__ import annotations

from src.causal_replay import run_graph_only_query
from src.streaming_fuzzy_hierarchy import build_causal_snapshot


def test_visible_under_budget_returns_all_graph_nodes(config, synthetic_data) -> None:
    embeddings, ids, namespaces, arrivals = synthetic_data
    snapshot = build_causal_snapshot(embeddings, ids, namespaces, arrivals, namespace_id="b", cutoff_order=20)
    result, _ = run_graph_only_query(snapshot, embeddings[128], embeddings, config=config)
    assert len(result.memory_ids) == 20
    assert result.underflow is False


def test_real_graph_pool_underflow_is_not_dense_filled(config, synthetic_data) -> None:
    embeddings, ids, namespaces, arrivals = synthetic_data
    snapshot = build_causal_snapshot(embeddings, ids, namespaces, arrivals, namespace_id="a", cutoff_order=64)
    limited = dict(config)
    limited["budget"] = dict(config["budget"], expanded_pool_cap=8)
    result, _ = run_graph_only_query(snapshot, embeddings[0], embeddings, config=limited)
    assert result.underflow is True
    assert result.memory_ids == []
    assert result.audit["dense_fallback_slots"] == 0
