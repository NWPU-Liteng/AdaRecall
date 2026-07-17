from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


VERSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VERSION_ROOT / "src"))

from v21.qesa import (  # noqa: E402
    LatentAspectQESA,
    QESAInferenceExample,
    alternative_group_affinity,
    group_coverage_loss,
    select_protected_frontier,
)


def test_qesa_distributions_are_nonnegative_and_normalized() -> None:
    model = LatentAspectQESA(6, 4, 3, 5)
    latent, aspect_map, prototype = model(torch.randn(2, 6))
    assert (latent >= 0).all() and torch.allclose(latent.sum(1), torch.ones(2), atol=1e-6)
    assert (aspect_map >= 0).all() and torch.allclose(aspect_map.sum(1), torch.ones(3), atol=1e-6)
    assert (prototype >= 0).all() and torch.allclose(prototype.sum(1), torch.ones(2), atol=1e-6)


def test_qesa_group_affinity_uses_alternative_or_semantics() -> None:
    members = torch.tensor([[0.8, 0.2, 0.0], [0.1, 0.7, 0.2]])
    affinity = alternative_group_affinity(members)
    assert torch.allclose(affinity, torch.tensor([1.0, 0.875, 0.25]))
    loss = group_coverage_loss(torch.tensor([[0.5, 0.5, 0.0]]), affinity[None, :], torch.tensor([0]))
    assert torch.isfinite(loss)


def test_qesa_inference_rejects_gold_group_fields() -> None:
    clean = {
        "query_embedding": np.ones(4, dtype=np.float32),
        "protected_ann_seed_ids": ["a"],
        "eligible_memory_ids": ["a", "b"],
    }
    assert QESAInferenceExample.from_mapping(clean).protected_ann_seed_ids == ("a",)
    try:
        QESAInferenceExample.from_mapping({**clean, "evidence_groups": [{"acceptable_memory_ids": ["a"]}]})
    except ValueError as error:
        assert "supervision-only" in str(error)
    else:
        raise AssertionError("QESA inference accepted evidence labels")


def test_qesa_frontier_preserves_seeds_and_budget() -> None:
    scores = np.asarray([0.1, 0.8, 0.7, 0.5])
    selected = select_protected_frontier(scores, [0, 1, 2, 3], [0, 3], 3)
    assert selected[:2] == [0, 3]
    assert selected == [0, 3, 1]


def test_qesa_query_id_permutation_does_not_change_predictions() -> None:
    torch.manual_seed(20260714)
    model = LatentAspectQESA(6, 4, 3, 5).eval()
    query_ids = ["query-a", "query-b", "query-c"]
    embeddings = torch.randn(3, 6)
    with torch.no_grad():
        original = model(embeddings)[2]
    permutation = [2, 0, 1]
    with torch.no_grad():
        permuted = model(embeddings[permutation])[2]
    by_id = {query_ids[index]: original[index] for index in range(len(query_ids))}
    permuted_by_id = {query_ids[index]: permuted[position] for position, index in enumerate(permutation)}
    assert all(torch.equal(by_id[query_id], permuted_by_id[query_id]) for query_id in query_ids)


if __name__ == "__main__":
    tests = sorted((name, value) for name, value in globals().items() if name.startswith("test_") and callable(value))
    for name, test in tests:
        test()
        print(f"PASS {name}")
    print(f"{len(tests)} passed")
