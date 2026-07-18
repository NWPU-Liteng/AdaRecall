from __future__ import annotations

import numpy as np

from v26.core import adaptive_features, oracle_tail_label, rank_percentile, select_tail


def test_rank_percentile_is_deterministic_on_ties() -> None:
    ranks = rank_percentile(np.asarray([1.0, 1.0, 0.0]), ["b", "a", "c"])
    assert ranks.tolist() == [1.0, 0.5, 0.0]


def test_tail_replacement_protects_dense_prefix_and_is_unique() -> None:
    ids = [f"m{index}" for index in range(40)]
    dense = np.arange(40, 0, -1, dtype=np.float32)
    lexical = np.arange(40, dtype=np.float32)
    selected = select_tail(ids, dense, lexical, 4, frontier_c=32)
    assert len(selected) == len(set(selected)) == 32
    assert selected[:28] == list(range(28))
    assert set(selected[-4:]).isdisjoint(set(range(32)))


def test_adaptive_features_detect_exact_signals() -> None:
    values = adaptive_features(
        "Which 20 mg medication was started in March 2025?",
        np.asarray([0.9, 0.8, 0.7]),
        np.asarray([0.0, 0.1, 0.8]),
        ["a", "b", "c"],
        frontier_c=2,
    )
    assert values[6] == 1.0
    assert values[7] == 1.0
    assert values[8] > 0.0


def test_oracle_tail_label_prefers_fewer_equal_options() -> None:
    metrics = {tail: {"candidate_recall": 0.5, "evidence_group_coverage": 0.5} for tail in (0, 2, 4, 8, 12)}
    assert oracle_tail_label(metrics) == 0
