from __future__ import annotations

from src.streaming_fuzzy_hierarchy import dynamic_coarse_k


def test_dynamic_k_boundaries() -> None:
    assert dynamic_coarse_k(0) == 0
    assert dynamic_coarse_k(1) == 1
    assert dynamic_coarse_k(17) == 17
    assert dynamic_coarse_k(32) == 32
    assert dynamic_coarse_k(1024) == 32
    assert dynamic_coarse_k(2500) == 50
    assert dynamic_coarse_k(10000) == 64
