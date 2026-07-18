from __future__ import annotations

import numpy as np
from scipy import sparse

from v24.diffusion import diffuse
from v24.frontier_selector import fuse_relevance, select_frontier
from v24.history_hypergraph import HistoryHypergraph
from v24.semantic_hypergraph import SemanticHypergraph, fit_spherical_soft


def test_spherical_soft_incidence_is_finite_and_sparse() -> None:
    rng = np.random.default_rng(4)
    values = np.vstack([rng.normal(size=(20, 4)) + np.array([4, 0, 0, 0]), rng.normal(size=(20, 4)) + np.array([-4, 0, 0, 0])]).astype(np.float32)
    result = fit_spherical_soft(values, n_prototypes=4, top_memberships=2, temperature=0.2, seed=4)
    assert result.incidence.shape == (40, 4)
    assert result.incidence.nnz <= 80
    assert np.isfinite(result.prototypes).all()
    assert np.isfinite(result.incidence.data).all()
    assert np.allclose(np.asarray(result.incidence.sum(axis=1)).reshape(-1), 1.0, atol=1e-5)


def test_query_conditioned_semantic_activation_uses_seed_incidence() -> None:
    prototypes = np.eye(3, dtype=np.float32)
    incidence = sparse.csr_matrix(np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32))
    graph = SemanticHypergraph(prototypes, incidence, ["m0", "m1", "m2"])
    active, _ = graph.active_edges(np.asarray([1, 0, 0], dtype=np.float32), np.asarray([0]), np.asarray([0.9], dtype=np.float32), top_edges=3)
    assert [edge["edge_id"] for edge in active] == [0]


def test_diffusion_restart_keeps_seed_mass_and_is_finite() -> None:
    restart = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    result = diffuse(3, [np.asarray([0, 1]), np.asarray([1, 2])], [1.0, 1.0], restart, alpha=0.8, steps=2)
    assert np.isfinite(result).all()
    assert np.isclose(result.sum(), 1.0)
    assert result[0] > result[2]


def test_frontier_is_dense_safe_and_deterministic() -> None:
    ids = ["m2", "m1", "m3"]
    fused, components = fuse_relevance(ids, np.asarray([0.9, 0.9, 0.1], dtype=np.float32), np.asarray([0.0, 1.0, 0.0], dtype=np.float32), np.zeros(3, dtype=np.float32), beta_dense=0.9)
    selected, rows = select_frontier(ids, fused, cap=2, components=components)
    assert len(selected) == 2
    assert rows[0]["memory_id"] == "m1"
    assert len({row["memory_id"] for row in rows}) == 2
