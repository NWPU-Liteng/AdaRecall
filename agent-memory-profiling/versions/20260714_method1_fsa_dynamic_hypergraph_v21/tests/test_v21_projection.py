from __future__ import annotations

import sys
from pathlib import Path

import torch


VERSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VERSION_ROOT / "src"))

from v21.semantic_projection import ResidualProjectionAdapter, multi_positive_info_nce  # noqa: E402


def test_v21_multi_positive_projection_has_finite_gradients() -> None:
    torch.manual_seed(20260714)
    adapter = ResidualProjectionAdapter(dimension=8, bottleneck=4)
    anchors = torch.randn(3, 8)
    candidates = torch.randn(5, 8)
    mask = torch.tensor([
        [1, 1, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0],
    ], dtype=torch.bool)
    loss = multi_positive_info_nce(adapter, anchors, candidates, mask, temperature=0.07)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in adapter.parameters())


if __name__ == "__main__":
    test_v21_multi_positive_projection_has_finite_gradients()
    print("PASS test_v21_multi_positive_projection_has_finite_gradients")
