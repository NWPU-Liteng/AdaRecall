from __future__ import annotations

from src.posterior import posterior_rank


def fixture_rows():
    ids = ["a", "b", "c"]
    qwen = {"frontier_ids": ids, "reranked_ids": ["c", "b", "a"], "reranker_scores": [0.9, 0.5, 0.1]}
    provenance = [
        {"memory_id": value, "selection_order": index + 1, "x_sem": 3 - index, "relevance_marginal": 3 - index, "edge_coverage_marginal": 0, "facility_marginal": 0}
        for index, value in enumerate(ids)
    ]
    frontier = {"frontier_ids": ids, "graph_provenance": provenance}
    return qwen, frontier


def test_alpha_one_is_qwen_order() -> None:
    qwen, frontier = fixture_rows()
    assert posterior_rank(qwen, frontier, alpha_qwen=1.0, prior_kind="x_sem", graph_method=True) == ["c", "b", "a"]


def test_graph_prior_can_change_order_without_dense_signal() -> None:
    qwen, frontier = fixture_rows()
    assert posterior_rank(qwen, frontier, alpha_qwen=0.25, prior_kind="x_sem", graph_method=True)[0] == "a"


def test_baseline_control_uses_native_retrieval_order() -> None:
    qwen, frontier = fixture_rows()
    assert posterior_rank(qwen, frontier, alpha_qwen=0.25, prior_kind="retrieval_order", graph_method=False)[0] == "a"
