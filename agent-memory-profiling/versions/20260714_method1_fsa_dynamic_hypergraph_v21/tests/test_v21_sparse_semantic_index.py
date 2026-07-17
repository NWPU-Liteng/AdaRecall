from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


VERSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VERSION_ROOT / "src"))

from v21.sparse_semantic_index import (  # noqa: E402
    SparseSemanticInvertedIndex,
    SparseSemanticLookupRequest,
    build_prototype_rank_index,
)


def fixture():
    incidence = np.array([[0.7, 0.1], [0.9, 0.3], [0.8, 0.9], [0.2, 0.8]], dtype=np.float32)
    return incidence, SparseSemanticInvertedIndex(build_prototype_rank_index(incidence), incidence)


def test_lookup_reads_only_bounded_rank_prefix_until_eligible_members_found() -> None:
    _incidence, index = fixture()
    request = SparseSemanticLookupRequest.from_mapping({
        "active_prototype_ids": [0, 1], "eligible_memory_positions": [0, 2, 3], "members_per_edge": 2,
    }, index.prototype_count, index.memory_count)
    result = index.lookup(request)
    assert result.activated_semantic_edges == 2
    assert result.edges[0].memory_positions == (2, 0)
    assert result.edges[0].rank_entries_visited == 3
    assert result.edges[1].memory_positions == (2, 3)
    assert result.incidence_entries_visited == 5
    assert result.metadata_bytes == 2 * 4 * np.dtype(np.int32).itemsize


def test_lookup_rejects_supervision_fields() -> None:
    _incidence, index = fixture()
    try:
        SparseSemanticLookupRequest.from_mapping({
            "active_prototype_ids": [0], "eligible_memory_positions": [0], "members_per_edge": 1,
            "acceptable_memory_ids": ["m0"],
        }, index.prototype_count, index.memory_count)
    except ValueError as error:
        assert "supervision-only" in str(error)
    else:
        raise AssertionError("sparse lookup accepted evidence supervision")


def test_rank_index_ties_are_deterministic_by_memory_position() -> None:
    incidence = np.array([[0.5], [0.5], [0.4]], dtype=np.float32)
    assert build_prototype_rank_index(incidence)[0].tolist() == [0, 1, 2]


if __name__ == "__main__":
    tests = sorted((name, value) for name, value in globals().items() if name.startswith("test_") and callable(value))
    for name, test in tests:
        test()
        print(f"PASS {name}")
    print(f"{len(tests)} passed")
