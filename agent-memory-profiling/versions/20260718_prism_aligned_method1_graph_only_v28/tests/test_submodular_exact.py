from __future__ import annotations

import itertools

import numpy as np

from src.graph_only_selector import ObjectiveWeights, exact_optimum_value, incidence_similarity


def test_incidence_similarity_is_nonnegative() -> None:
    incidence = np.asarray([[1, 0], [0.5, 0.5], [0, 1]], dtype=np.float32)
    similarity = incidence_similarity(incidence)
    assert np.all(similarity >= 0)
    assert np.allclose(np.diag(similarity), 1.0)


def test_exact_objective_audit_has_finite_optimum() -> None:
    relevance = np.asarray([1.0, 0.8, 0.7, 0.3], dtype=np.float32)
    incidence = np.asarray([[1, 0], [1, 0], [0, 1], [0.5, 0.5]], dtype=np.float32)
    value, choice = exact_optimum_value(relevance, incidence, np.asarray([0.6, 0.4]), 2, ObjectiveWeights())
    assert value > 0
    assert len(choice) == 2
    assert choice in set(itertools.combinations(range(4), 2))
