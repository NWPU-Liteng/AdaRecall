#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


VERSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VERSION_ROOT / "src"))

from v21.io import (  # noqa: E402
    iter_jsonl,
    load_config,
    prepare_attempt_directory,
    read_json,
    resolve,
    sha256_file,
    write_csv_new,
    write_json_new,
)
from v21.uot_recipes import array_sha256, build_preregistered_recipes  # noqa: E402
from v21.uot_semantic import farthest_point_prototypes, fit_uot, robust_cost_scale, sparse_memberships  # noqa: E402


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe_output_directory(path: str | Path) -> Path:
    output = Path(path).resolve()
    root = VERSION_ROOT.resolve()
    if not output.is_relative_to(root):
        raise RuntimeError(f"v21 UOT artifacts must remain under {root}")
    return output


def _write_npy_new(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.save(handle, value, allow_pickle=False)


def load_groups(path: str) -> list[list[str]]:
    groups = []
    for row in iter_jsonl(path):
        for group in row.get("evidence_groups", []):
            members = list(dict.fromkeys(map(str, group.get("acceptable_memory_ids", []))))
            if members:
                groups.append(members)
    return groups


def colocation(groups: list[list[str]], membership: dict[str, set[int]]) -> float:
    values = []
    for group in groups:
        assigned = [membership[memory_id] for memory_id in group if memory_id in membership]
        values.append(float(len(assigned) == len(group) and bool(set.intersection(*assigned))))
    return float(np.mean(values)) if values else 0.0


def _recipes_for_args(config: dict, args: argparse.Namespace) -> list[dict]:
    if args.smoke:
        return [
            {
                "recipe_id": 0,
                "recipe_key": "smoke_none",
                "diversity_mode": "none",
                "diversity_step_size": 0.0,
                "epsilon_mult": 0.02,
                "tau_memory": 1.0,
                "tau_prototype": 1.0,
            },
            {
                "recipe_id": 0,
                "recipe_key": "smoke_logdet_002",
                "diversity_mode": "logdet_proximal",
                "diversity_step_size": 0.002,
                "epsilon_mult": 0.02,
                "tau_memory": 1.0,
                "tau_prototype": 1.0,
            },
            {
                "recipe_id": 0,
                "recipe_key": "smoke_logdet_010",
                "diversity_mode": "logdet_proximal",
                "diversity_step_size": 0.01,
                "epsilon_mult": 0.01,
                "tau_memory": 0.25,
                "tau_prototype": 0.25,
            },
        ]
    recipes = build_preregistered_recipes(config)
    recipe_stop = len(recipes) if args.recipe_stop == 0 else args.recipe_stop
    if args.recipe_start < 1 or recipe_stop < args.recipe_start or recipe_stop > len(recipes):
        raise RuntimeError(f"invalid recipe interval [{args.recipe_start}, {recipe_stop}] for {len(recipes)} recipes")
    return recipes[args.recipe_start - 1 : recipe_stop]


def _default_output_directory(run_id: str, recipe: dict, attempt: int) -> Path:
    return (
        VERSION_ROOT
        / "artifacts"
        / "v21_uot"
        / run_id
        / "shards"
        / f"recipe_{int(recipe['recipe_id']):02d}_{recipe['recipe_key']}"
        / "attempts"
        / f"attempt_{attempt:03d}"
    )


def _run_one_recipe(
    *,
    recipe: dict,
    args: argparse.Namespace,
    data: dict,
    config: dict,
    vectors: np.ndarray,
    ids: list[str],
    groups: list[list[str]],
    config_hash: str,
    embedding_hash: str,
) -> dict:
    if args.output_dir:
        output_dir = _safe_output_directory(args.output_dir)
    else:
        output_dir = _default_output_directory(args.run_id, recipe, args.attempt)
    base_name = f"v21_uot_recipe_{int(recipe['recipe_id']):02d}_attempt_{args.attempt:03d}"
    prepare_attempt_directory(output_dir, allowed_existing_names={f"{base_name}.log"})
    prototype_count = 64
    seed = int(config["fixed_seeds"][0])
    initial_prototypes = farthest_point_prototypes(vectors, prototype_count, seed, args.device)
    scale = robust_cost_scale(vectors, initial_prototypes, args.device)
    prototypes, incidence, numerical = fit_uot(
        vectors,
        initial_prototypes,
        epsilon=max(1e-5, scale * float(recipe["epsilon_mult"])),
        tau_memory=float(recipe["tau_memory"]),
        tau_prototype=float(recipe["tau_prototype"]),
        dustbin_prior_mass=float(config["uot"]["dustbin_prior_mass"]),
        dustbin_cost=scale * 0.5,
        outer_iterations=2,
        sinkhorn_iterations=int(config["uot"]["sinkhorn_iterations"]),
        sinkhorn_tolerance=float(config["uot"]["sinkhorn_tolerance"]),
        diversity_mode=str(recipe["diversity_mode"]),
        diversity_step_size=float(recipe["diversity_step_size"]),
        diversity_steps=int(config["uot"]["prototype_diversity"]["proximal_steps"]),
        device=args.device,
    )
    memberships = {
        memory_id: set(values)
        for memory_id, values in zip(
            ids,
            sparse_memberships(
                incidence,
                float(config["uot"]["membership"]["cumulative_mass"]),
                int(config["uot"]["membership"]["max_memberships"]),
            ),
            strict=True,
        )
    }
    numerical["train_evidence_group_colocation_diagnostic_only"] = colocation(groups, memberships)
    prototype_path = output_dir / f"{base_name}_prototypes.npy"
    incidence_path = output_dir / f"{base_name}_semantic_incidence.npy"
    csv_path = output_dir / f"{base_name}.csv"
    json_path = output_dir / f"{base_name}.json"
    _write_npy_new(prototype_path, prototypes)
    _write_npy_new(incidence_path, incidence)
    result = {
        "schema_version": "v21-uot-recipe-shard-2",
        "execution_status": "completed",
        "numerical_status": "primary_converged" if numerical["transport_converged"] else "primary_nonconverged",
        "screen_kind": "smoke" if args.smoke else "full_preregistered",
        "run_id": args.run_id,
        "attempt": int(args.attempt),
        "created_at_utc": _utc_now(),
        "device": args.device,
        "recipe": recipe,
        "config_path": str((VERSION_ROOT / "configs" / args.config_name).relative_to(VERSION_ROOT)),
        "config_hash": config_hash,
        "input_embedding_path": str(resolve(data["memory"]["embedding_vectors"]).relative_to(VERSION_ROOT.parent)),
        "input_embedding_hash": embedding_hash,
        "input_embedding_ids_hash": sha256_file(resolve(data["memory"]["embedding_ids"])),
        "input_initial_prototype_hash": array_sha256(initial_prototypes),
        "prototype_hash": array_sha256(prototypes),
        "semantic_incidence_hash": array_sha256(incidence),
        "input_vector_shape": list(vectors.shape),
        "train_evidence_groups": len(groups),
        "artifacts": {
            "prototypes": prototype_path.name,
            "semantic_incidence": incidence_path.name,
            "csv": csv_path.name,
        },
        "diagnostics": numerical,
        "calibration_read": False,
        "formal_test_read": False,
        "selection_prohibition": "numerical diagnostics cannot select a final model; nested session-blocked train-only OOF Pareto selection remains required",
    }
    row = {
        "run_id": args.run_id,
        "attempt": int(args.attempt),
        **recipe,
        "execution_status": result["execution_status"],
        "numerical_status": result["numerical_status"],
        "config_hash": config_hash,
        "input_embedding_hash": embedding_hash,
        "input_initial_prototype_hash": result["input_initial_prototype_hash"],
        "prototype_hash": result["prototype_hash"],
        "semantic_incidence_hash": result["semantic_incidence_hash"],
        **numerical,
    }
    write_csv_new(str(csv_path), [row])
    write_json_new(str(json_path), result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an isolated v21 train-only UOT numerical recipe shard.")
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--smoke", action="store_true", help="Run retained implementation smoke recipes only.")
    parser.add_argument("--full-preregistered-screen", action="store_true", help="Run recipes from the fixed 1..27 UOT screen.")
    parser.add_argument("--recipe-start", type=int, default=1, help="One-based inclusive recipe position.")
    parser.add_argument("--recipe-stop", type=int, default=0, help="One-based inclusive recipe position; 0 selects through recipe 27.")
    parser.add_argument("--output-dir", default="", help="New v21 attempt directory; never an existing directory.")
    parser.add_argument("--run-id", default="", help="Local run revision recorded in the shard manifest.")
    parser.add_argument("--attempt", type=int, default=1, help="Monotone attempt number for a recipe within one run.")
    parser.add_argument("--config-name", default="v21_uot_semantic.json")
    parser.add_argument("--input-embedding-hash", default="", help="Hash supplied by the launcher manifest.")
    args = parser.parse_args()
    if args.smoke == args.full_preregistered_screen:
        raise RuntimeError("select exactly one of --smoke or --full-preregistered-screen")
    if args.attempt < 1:
        raise RuntimeError("attempt must be positive")
    args.run_id = args.run_id or f"v21_uot_manual_{_utc_now()}"
    data = load_config("v21_data.json")
    config = load_config(args.config_name)
    recipes = _recipes_for_args(config, args)
    if len(recipes) != 1:
        raise RuntimeError("the v21 worker executes exactly one recipe; use scripts/84_launch_v21_uot_recipe_shards.py for ranges")
    ids = list(map(str, read_json(data["memory"]["embedding_ids"])))
    vectors = np.load(resolve(data["memory"]["embedding_vectors"]), mmap_mode="r").astype(np.float32)
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    groups = load_groups(data["split"]["train_query_plan"])
    config_hash = sha256_file(VERSION_ROOT / "configs" / args.config_name)
    embedding_hash = args.input_embedding_hash or sha256_file(resolve(data["memory"]["embedding_vectors"]))
    result = _run_one_recipe(
        recipe=recipes[0],
        args=args,
        data=data,
        config=config,
        vectors=vectors,
        ids=ids,
        groups=groups,
        config_hash=config_hash,
        embedding_hash=embedding_hash,
    )
    print(json.dumps({
        "run_id": result["run_id"],
        "recipe_id": result["recipe"]["recipe_id"],
        "numerical_status": result["numerical_status"],
        "transport_iterations": result["diagnostics"]["transport_iterations"],
        "transport_potential_delta": result["diagnostics"]["transport_potential_delta"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
