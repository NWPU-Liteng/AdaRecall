from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

import numpy as np


REQUIRED_NUMERICAL_DIAGNOSTICS = (
    "transport_plan_delta",
    "primal_objective",
    "dual_surrogate",
    "marginal_mass_residual",
    "finite_ratio",
    "minimum_plan_value",
    "maximum_plan_value",
    "effective_transported_mass",
)


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(repr(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def build_preregistered_recipes(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return the fixed 3 x 3 x 3 numerical screen in deterministic order."""
    uot = config["uot"]
    diversity = [("none", 0.0)] + [
        ("logdet_proximal", float(step)) for step in uot["prototype_diversity"]["step_sizes"]
    ]
    coupled_taus = list(zip(uot["tau_memory"], uot["tau_prototype"], strict=True))
    recipes: list[dict[str, Any]] = []
    recipe_id = 1
    for diversity_mode, diversity_step_size in diversity:
        for epsilon_mult in uot["epsilon_cost_scale_multipliers"]:
            for tau_memory, tau_prototype in coupled_taus:
                recipes.append({
                    "recipe_id": recipe_id,
                    "recipe_key": f"screen_{recipe_id:02d}",
                    "diversity_mode": diversity_mode,
                    "diversity_step_size": float(diversity_step_size),
                    "epsilon_mult": float(epsilon_mult),
                    "tau_memory": float(tau_memory),
                    "tau_prototype": float(tau_prototype),
                })
                recipe_id += 1
    if [recipe["recipe_id"] for recipe in recipes] != list(range(1, 28)):
        raise RuntimeError("v21 preregistered UOT grid must contain recipe IDs 1..27 exactly")
    return recipes


def recipe_for_id(config: Mapping[str, Any], recipe_id: int) -> dict[str, Any]:
    recipes = build_preregistered_recipes(config)
    if recipe_id < 1 or recipe_id > len(recipes):
        raise ValueError(f"recipe_id must be in 1..{len(recipes)}")
    return recipes[recipe_id - 1]


def recipe_matches(expected: Mapping[str, Any], observed: Mapping[str, Any]) -> bool:
    keys = (
        "recipe_id",
        "recipe_key",
        "diversity_mode",
        "diversity_step_size",
        "epsilon_mult",
        "tau_memory",
        "tau_prototype",
    )
    for key in keys:
        if key not in observed:
            return False
        if key in {"diversity_step_size", "epsilon_mult", "tau_memory", "tau_prototype"}:
            if float(expected[key]) != float(observed[key]):
                return False
        elif expected[key] != observed[key]:
            return False
    return True
