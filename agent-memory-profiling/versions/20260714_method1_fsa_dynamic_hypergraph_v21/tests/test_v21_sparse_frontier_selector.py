from __future__ import annotations

import torch

import numpy as np

from v21.sparse_frontier_selector import (
    SparseFrontierUtilityMLP,
    StructureAwareMarginalSelector,
    group_target_cross_entropy,
    structure_marginal_features,
)


def test_sparse_frontier_model_scores_each_candidate() -> None:
    model = SparseFrontierUtilityMLP(4, 8)
    assert model(torch.zeros(3, 7, 4)).shape == (3, 7)


def test_group_target_loss_excludes_protected_candidates() -> None:
    scores = torch.tensor([[10.0, 0.0, 2.0]], requires_grad=True)
    target = torch.tensor([[0.0, 0.0, 1.0]])
    protected = torch.tensor([[True, False, False]])
    loss = group_target_cross_entropy(scores, target, protected)
    loss.backward()
    assert float(loss) > 0.0
    assert scores.grad is not None
    assert float(scores.grad[0, 0]) == 0.0


def test_structure_aware_model_keeps_unary_and_marginal_axes_aligned() -> None:
    model = StructureAwareMarginalSelector(4, 5, 8)
    assert model(torch.zeros(3, 7, 4), torch.zeros(3, 7, 5)).shape == (3, 7)


def test_structure_marginals_expose_completion_novelty_and_diversity() -> None:
    selected = np.asarray([True, True, False, False])
    bundles = np.asarray([[True, True, True, False], [False, True, False, True]])
    bundle_weights = np.asarray([0.75, 0.25], dtype=np.float32)
    prototypes = np.asarray([[True, False, True, False], [False, False, False, True]])
    prototype_weights = np.asarray([0.6, 0.4], dtype=np.float32)
    similarity = np.eye(4, dtype=np.float32)
    features = structure_marginal_features(
        selected, bundles, bundle_weights, prototypes, prototype_weights, similarity,
    )
    assert features.shape == (4, 5)
    assert np.isclose(features[2, 0], 0.75)
    assert np.isclose(features[2, 2], 0.0)
    assert np.isclose(features[3, 2], 0.4)
    assert np.isclose(features[2, 4], 1.0)
    assert np.all(features[selected] == 0.0)


if __name__ == "__main__":
    tests = sorted((name, value) for name, value in globals().items() if name.startswith("test_") and callable(value))
    for name, test in tests:
        test()
        print(f"PASS {name}")
    print(f"{len(tests)} passed")
