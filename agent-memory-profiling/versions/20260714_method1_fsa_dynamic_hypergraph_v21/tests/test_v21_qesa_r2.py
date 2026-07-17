from __future__ import annotations

import sys
from pathlib import Path

import torch


VERSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VERSION_ROOT / "src"))

from v21.qesa_r2 import LatentAspectQESAR2, qesa_r2_training_loss  # noqa: E402


def test_qesa_r2_distributions_are_finite_nonnegative_and_normalized() -> None:
    model = LatentAspectQESAR2(6, 8, 4, 7, 0.25)
    latent, aspect_map, prototype = model(torch.randn(5, 6))
    for probabilities in (latent, aspect_map, prototype):
        assert torch.isfinite(probabilities).all()
        assert (probabilities >= 0).all()
        assert torch.allclose(probabilities.sum(dim=1), torch.ones(len(probabilities)), atol=1e-6)


def test_qesa_r2_soft_target_and_anticollapse_loss_has_finite_gradients() -> None:
    torch.manual_seed(20260714)
    model = LatentAspectQESAR2(5, 8, 3, 6, 0.25)
    query = torch.randn(4, 5)
    affinities = torch.rand(6, 6)
    owners = torch.tensor([0, 0, 1, 2, 3, 3])
    targets = torch.rand(4, 6)
    targets /= targets.sum(dim=1, keepdim=True)
    loss, diagnostics = qesa_r2_training_loss(
        model,
        query,
        affinities,
        owners,
        targets,
        coverage_weight=0.25,
        target_ce_weight=1.0,
        diversity_weight=0.1,
        map_entropy_weight=0.01,
        aspect_usage_weight=0.01,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert set(diagnostics) == {
        "coverage_loss",
        "target_cross_entropy",
        "aspect_diversity_penalty",
        "aspect_map_entropy",
        "aspect_usage_kl",
    }
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())


if __name__ == "__main__":
    tests = sorted((name, value) for name, value in globals().items() if name.startswith("test_") and callable(value))
    for name, test in tests:
        test()
        print(f"PASS {name}")
    print(f"{len(tests)} passed")
