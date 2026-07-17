from __future__ import annotations

import runpy
import sys
from pathlib import Path

import numpy as np
import torch


VERSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VERSION_ROOT / "src"))

from v21.io import load_config, prepare_attempt_directory  # noqa: E402
from v21.uot_recipes import array_sha256, build_preregistered_recipes  # noqa: E402
from v21.uot_semantic import diagnostics, log_unbalanced_sinkhorn, normalize_semantic_incidence, normalize_semantic_log_weights, primary_convergence_flag  # noqa: E402


def _transport(cost: torch.Tensor, *, tau: float = 0.25):
    a = torch.full((cost.shape[0],), 1.0 / cost.shape[0], dtype=torch.float32)
    b = torch.full((cost.shape[1],), 1.0 / cost.shape[1], dtype=torch.float32)
    return log_unbalanced_sinkhorn(
        cost,
        a,
        b,
        epsilon=0.2,
        tau_memory=tau,
        tau_prototype=tau,
        max_iterations=300,
        tolerance=1e-6,
    ), a, b


def test_uot_plan_is_finite_and_nonnegative() -> None:
    transport, _, _ = _transport(torch.tensor([[0.0, 1.0, 0.5], [1.0, 0.0, 0.5]]))
    assert torch.isfinite(transport.plan).all()
    assert (transport.plan >= 0).all()


def test_uot_incidence_normalization() -> None:
    transport, _, _ = _transport(torch.tensor([[0.0, 1.0, 0.5], [1.0, 0.0, 0.5]]))
    incidence, full = normalize_semantic_incidence(transport.plan, prototype_count=2)
    assert torch.allclose(full.sum(dim=1), torch.ones(2), atol=1e-6)
    assert torch.allclose(incidence.sum(dim=1), torch.ones(2), atol=1e-6)
    stable_incidence, stable_full = normalize_semantic_log_weights(
        torch.tensor([[-1000.0, -1001.0, 0.0], [-1002.0, -1000.0, 0.0]]),
        prototype_count=2,
    )
    assert torch.allclose(stable_incidence.sum(dim=1), torch.ones(2), atol=1e-6)
    assert torch.allclose(stable_full.sum(dim=1), torch.ones(2), atol=1e-6)


def test_uot_primary_convergence_flag() -> None:
    assert primary_convergence_flag(1e-6, 1e-6, 300, 300)
    assert not primary_convergence_flag(1.000001e-6, 1e-6, 300, 300)
    assert not primary_convergence_flag(1e-6, 1e-6, 301, 300)


def test_uot_unbalanced_marginals_are_not_forced_equal() -> None:
    cost = torch.tensor([[0.0, 5.0], [5.0, 0.0]], dtype=torch.float32)
    transport, a, b = _transport(cost, tau=0.01)
    row_l1 = (transport.plan.sum(dim=1) - a).abs().sum().item()
    column_l1 = (transport.plan.sum(dim=0) - b).abs().sum().item()
    assert row_l1 > 1e-5 or column_l1 > 1e-5


def test_uot_prototype_collapse_metrics() -> None:
    cost = torch.tensor([[0.0, 0.0, 1.0, 0.5], [0.0, 0.0, 1.0, 0.5]], dtype=torch.float32)
    transport, a, b = _transport(cost)
    incidence, full = normalize_semantic_incidence(transport.plan, prototype_count=3)
    prototypes = np.asarray([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    values = diagnostics(
        prototypes,
        incidence.numpy(),
        full.numpy(),
        transport,
        cost,
        a,
        b,
        0.2,
        0.25,
        0.25,
        0.0,
    )
    assert values["prototype_pairwise_cosine_max"] >= 0.9999
    assert values["prototype_collapse_flag"] is True
    assert values["finite_ratio"] == 1.0


def test_uot_recipe_manifest_exactly_covers_1_to_27() -> None:
    recipes = build_preregistered_recipes(load_config("v21_uot_semantic.json"))
    assert [recipe["recipe_id"] for recipe in recipes] == list(range(1, 28))
    assert len({recipe["recipe_key"] for recipe in recipes}) == 27


def test_uot_attempt_directory_allows_only_current_log(tmp_path: Path) -> None:
    attempt = tmp_path / "attempt_001"
    attempt.mkdir()
    log_name = "v21_uot_recipe_01_attempt_001.log"
    (attempt / log_name).write_text("launcher log\n", encoding="utf-8")
    assert prepare_attempt_directory(attempt, allowed_existing_names={log_name}) == attempt

    artifact = attempt / "v21_uot_recipe_01_attempt_001.json"
    artifact.write_text("{}\n", encoding="utf-8")
    try:
        prepare_attempt_directory(attempt, allowed_existing_names={log_name})
    except FileExistsError as error:
        assert artifact.name in str(error)
    else:
        raise AssertionError("an existing result artifact must block an attempt rerun")


def test_uot_merge_allows_signed_prototypes_but_rejects_negative_incidence(tmp_path: Path) -> None:
    validate = runpy.run_path(
        str(VERSION_ROOT / "scripts" / "85_merge_v21_uot_recipe_shards.py"),
        run_name="v21_merge_test",
    )["_validate_artifact_array"]
    prototype = np.asarray([[1.0, -1.0], [-0.5, 0.5]], dtype=np.float32)
    prototype_path = tmp_path / "prototype.npy"
    np.save(prototype_path, prototype, allow_pickle=False)
    assert validate(prototype_path, array_sha256(prototype), prototype.shape, incidence=False) == []

    incidence = np.asarray([[1.1, -0.1], [0.5, 0.5]], dtype=np.float32)
    incidence_path = tmp_path / "incidence.npy"
    np.save(incidence_path, incidence, allow_pickle=False)
    errors = validate(incidence_path, array_sha256(incidence), incidence.shape, incidence=True)
    assert any("negative" in error for error in errors)
