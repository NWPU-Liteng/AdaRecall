from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from v25.residual_frontier import ResidualConfig, calibrated_edge_weight, select_residual_frontier


def test_broad_single_seed_edge_is_suppressed() -> None:
    assert calibrated_edge_weight(1.0, edge_size=32, visible_count=1000, seed_overlap=1) == 0.0
    assert calibrated_edge_weight(1.0, edge_size=4, visible_count=1000, seed_overlap=3) > 0.0


def test_zero_replacements_is_exact_dense_anchor() -> None:
    ids = ["a", "b", "c", "d"]
    dense = np.asarray([0.9, 0.8, 0.7, 0.6])
    selected, _ = select_residual_frontier(ids, dense, dense[::-1], dense[::-1], cap=2, config=ResidualConfig(max_replacements=0))
    assert selected == [0, 1]


def test_supported_challenger_only_replaces_tail() -> None:
    ids = ["a", "b", "c", "d"]
    dense = np.asarray([0.90, 0.80, 0.79, 0.78])
    semantic = np.asarray([0.10, 0.00, 1.00, 0.20])
    history = np.zeros(4)
    config = ResidualConfig(max_replacements=1, protected_prefix=1, graph_margin=0.1, max_dense_drop=0.02, history_weight=0.0, minimum_graph_rank=0.7)
    selected, rows = select_residual_frontier(ids, dense, semantic, history, cap=2, config=config)
    assert selected == [0, 2]
    assert sum(row["residual_replacement"] for row in rows) == 1
