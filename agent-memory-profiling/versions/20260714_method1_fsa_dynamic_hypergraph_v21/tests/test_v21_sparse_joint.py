from __future__ import annotations

import sys
from pathlib import Path

import torch


VERSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VERSION_ROOT / "src"))

from v21.sparse_joint import (  # noqa: E402
    BudgetedPairwiseFactorGraph,
    build_semantic_bundle_beam,
    construct_sparse_frontier,
    pairwise_greedy_select,
)
from v21.sparse_semantic_index import SparseSemanticEdge  # noqa: E402


def edges():
    return [
        SparseSemanticEdge(0, (0, 1), (0.9, 0.8), 2),
        SparseSemanticEdge(1, (2, 3), (0.9, 0.7), 2),
        SparseSemanticEdge(2, (4, 5), (0.8, 0.7), 2),
        SparseSemanticEdge(3, (6, 7), (0.8, 0.6), 2),
    ]


def test_semantic_bundle_beam_uses_distinct_edges_and_members() -> None:
    bundles, audit = build_semantic_bundle_beam(edges(), {0: 0.4, 1: 0.3, 2: 0.2, 3: 0.1}, {i: 1.0 - i * 0.01 for i in range(8)}, arity=3, beam_width=4)
    assert bundles and len(bundles) <= 4 and audit["bundle_hyperedges_generated"] > 0
    assert all(len(bundle.prototype_ids) == len(set(bundle.prototype_ids)) == 3 for bundle in bundles)
    assert all(len(bundle.memory_positions) == len(set(bundle.memory_positions)) == 3 for bundle in bundles)


def test_frontier_fallback_only_uses_returned_edge_members_and_makes_budget_feasible() -> None:
    bundles, _audit = build_semantic_bundle_beam(edges(), {0: 0.4, 1: 0.3, 2: 0.2, 3: 0.1}, {i: float(10 - i) for i in range(8)}, arity=3, beam_width=1)
    frontier = construct_sparse_frontier([9], bundles, edges(), {i: float(10 - i) for i in range(10)}, maximum_budget=6)
    assert len(frontier.memory_positions) == 6 and not frontier.budget_infeasible
    assert set(frontier.memory_positions).difference({0, 1, 2, 3, 4, 5, 6, 7, 9}) == set()
    assert frontier.fallback_members_appended > 0


def test_pairwise_graph_preserves_protected_prefix_and_budget() -> None:
    torch.manual_seed(9)
    model = BudgetedPairwiseFactorGraph(3, 8)
    features = torch.rand(7, 3)
    selected = pairwise_greedy_select(model, features, [0, 2], 5)
    assert selected[:2] == [0, 2]
    assert len(selected) == 5 and len(set(selected)) == 5


if __name__ == "__main__":
    tests = sorted((name, value) for name, value in globals().items() if name.startswith("test_") and callable(value))
    for name, test in tests:
        test()
        print(f"PASS {name}")
    print(f"{len(tests)} passed")
