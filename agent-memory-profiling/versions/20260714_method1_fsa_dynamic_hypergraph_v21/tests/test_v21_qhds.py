from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


VERSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VERSION_ROOT / "src"))

from v21.qhds import MonotoneSubmodularQHDS, QHDSInferenceExample, build_features, greedy_select  # noqa: E402


def fixture():
    torch.manual_seed(20260714)
    model = MonotoneSubmodularQHDS(3, 8)
    cosine = torch.tensor([0.8, 0.7, 0.4, 0.2])
    association = torch.tensor([0.6, 0.5, 0.7, 0.1])
    features = build_features(cosine, association)
    incidence = torch.tensor([[0.8, 0.2], [0.7, 0.3], [0.1, 0.9], [0.4, 0.6]])
    aspects = torch.tensor([0.5, 0.5])
    relevance = model.relevance(features)
    return model, features, incidence, aspects, relevance


def test_qhds_utility_is_monotone() -> None:
    model, _features, incidence, aspects, relevance = fixture()
    empty = torch.zeros(4, dtype=torch.bool)
    one = empty.clone(); one[0] = True
    two = one.clone(); two[2] = True
    assert model.utility(empty, aspects, incidence, relevance) <= model.utility(one, aspects, incidence, relevance)
    assert model.utility(one, aspects, incidence, relevance) <= model.utility(two, aspects, incidence, relevance)


def test_qhds_has_diminishing_marginal_returns() -> None:
    model, _features, incidence, aspects, relevance = fixture()
    empty = torch.zeros(4, dtype=torch.bool)
    covered = empty.clone(); covered[0] = True
    empty_gain = model.marginal_gains(empty, aspects, incidence, relevance)[1]
    covered_gain = model.marginal_gains(covered, aspects, incidence, relevance)[1]
    assert covered_gain <= empty_gain


def test_qhds_greedy_preserves_protected_seeds_and_budget() -> None:
    model, features, incidence, aspects, _relevance = fixture()
    selected = greedy_select(model, features, incidence, aspects, [0, 1], 3)
    assert selected[:2] == [0, 1]
    assert len(selected) == 3 and len(set(selected)) == 3


def test_qhds_budget_policies_are_nested_prefixes() -> None:
    model, features, incidence, aspects, _relevance = fixture()
    order = greedy_select(model, features, incidence, aspects, [0], 4)
    assert set(order[:2]).issubset(order[:3])
    assert set(order[:3]).issubset(order[:4])


def test_qhds_inference_rejects_gold_fields() -> None:
    clean = {"query_embedding": np.ones(4, dtype=np.float32), "protected_ann_seed_ids": ["a"], "eligible_memory_ids": ["a", "b"]}
    assert QHDSInferenceExample.from_mapping(clean).protected_ann_seed_ids == ("a",)
    try:
        QHDSInferenceExample.from_mapping({**clean, "acceptable_member_ids": ["a"]})
    except ValueError as error:
        assert "supervision-only" in str(error)
    else:
        raise AssertionError("QHDS inference accepted supervision")


if __name__ == "__main__":
    tests = sorted((name, value) for name, value in globals().items() if name.startswith("test_") and callable(value))
    for name, test in tests:
        test()
        print(f"PASS {name}")
    print(f"{len(tests)} passed")
