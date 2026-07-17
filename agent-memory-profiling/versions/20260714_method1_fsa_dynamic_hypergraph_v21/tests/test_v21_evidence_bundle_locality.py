from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


VERSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VERSION_ROOT / "src"))

from v21.evidence_bundle_locality import (  # noqa: E402
    BundleCandidate,
    BundleInferenceExample,
    PairwiseOnlyBundleScorer,
    PermutationInvariantBundleScorer,
    build_semantic_bundle_candidates,
    ranked_bundle_metrics,
)


def test_deepsets_score_is_member_permutation_invariant() -> None:
    torch.manual_seed(7)
    scorer = PermutationInvariantBundleScorer(4, 3, 8).eval()
    query = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    members = torch.tensor([[[0.8, 0.2, 0.6], [0.7, 0.4, 0.1], [0.5, 0.9, 0.3]]])
    assert torch.allclose(scorer(query, members), scorer(query, members[:, [2, 0, 1]]), atol=1e-7)


def test_inference_requires_predicted_cardinality_and_disallows_gold_fields() -> None:
    candidate = BundleCandidate(("m1", "m2"), (3, 8), np.ones((2, 3), dtype=np.float32))
    clean = {
        "query_embedding": np.ones(4, dtype=np.float32), "eligible_memory_ids": ["m1", "m2"],
        "predicted_aspect_count": 2, "candidates": [candidate],
    }
    assert BundleInferenceExample.from_mapping(clean).predicted_aspect_count == 2
    try:
        BundleInferenceExample.from_mapping({**clean, "acceptable_memory_ids": ["m1"]})
    except ValueError as error:
        assert "supervision-only" in str(error)
    else:
        raise AssertionError("inference accepted acceptable evidence members")
    try:
        BundleInferenceExample.from_mapping({**clean, "predicted_aspect_count": 3})
    except ValueError as error:
        assert "cardinality" in str(error)
    else:
        raise AssertionError("inference accepted a cardinality mismatch")


def test_bundle_rejects_duplicate_member_across_semantic_aspects() -> None:
    try:
        BundleCandidate(("m1", "m1"), (3, 8), np.ones((2, 3), dtype=np.float32))
    except ValueError as error:
        assert "twice" in str(error)
    else:
        raise AssertionError("bundle accepted one memory for two evidence aspects")


def test_pairwise_comparator_is_neutral_for_a_predicted_singleton() -> None:
    scorer = PairwiseOnlyBundleScorer(4, 3, 8)
    values = scorer(torch.ones((2, 4)), torch.ones((2, 1, 3)))
    assert torch.equal(values, torch.zeros(2))


def test_candidate_builder_uses_distinct_semantic_aspects_and_bounded_beam() -> None:
    features = np.array([[0.9, 0.1, 0.2], [0.8, 0.2, 0.3], [0.2, 0.8, 0.7], [0.1, 0.9, 0.8]], dtype=np.float32)
    incidence = np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]], dtype=np.float32)
    candidates, audit = build_semantic_bundle_candidates(
        eligible_memory_ids=["a", "b", "c", "d"], eligible_member_features=features,
        eligible_incidence=incidence, aspect_prediction=np.array([0.6, 0.4], dtype=np.float32),
        predicted_aspect_count=2, activated_aspects=2, members_per_aspect=2, beam_width=3,
    )
    assert candidates and len(candidates) <= 3 and audit["activated_semantic_edges"] == 2
    assert all(len(set(candidate.semantic_aspect_ids)) == 2 for candidate in candidates)


def test_ranked_metrics_distinguish_top_one_from_beam_hit() -> None:
    bad = BundleCandidate(("a", "x"), (0, 1), np.ones((2, 3), dtype=np.float32))
    good = BundleCandidate(("a", "b"), (0, 1), np.ones((2, 3), dtype=np.float32))
    metrics = ranked_bundle_metrics([bad, good], [["a"], ["b"]], evaluation_beam=2)
    assert metrics["gold_compatible_bundle_hit_rate"] == 1.0
    assert metrics["all_groups_covered_rate"] == 0.0


if __name__ == "__main__":
    tests = sorted((name, value) for name, value in globals().items() if name.startswith("test_") and callable(value))
    for name, test in tests:
        test()
        print(f"PASS {name}")
    print(f"{len(tests)} passed")
