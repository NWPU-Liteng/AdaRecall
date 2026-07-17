#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np
import torch


VERSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VERSION_ROOT / "src"))

from v21.evidence_sets import fold_for  # noqa: E402
from v21.io import iter_jsonl, load_config, read_json, resolve, sha256_file, write_csv_new, write_json_new  # noqa: E402
from v21.semantic_projection import ResidualProjectionAdapter, multi_positive_info_nce, project_matrix  # noqa: E402
from v21.uot_recipes import array_sha256, build_preregistered_recipes  # noqa: E402
from v21.uot_semantic import farthest_point_prototypes, fit_uot, robust_cost_scale, sparse_memberships  # noqa: E402


def load_query_vectors(data: dict) -> dict[str, np.ndarray]:
    matrix = np.load(resolve(data["query_embeddings"]["vectors"]), mmap_mode="r")
    index = read_json(data["query_embeddings"]["index"])["records"]
    return {
        str(row["query_id"]): np.array(matrix[position], dtype=np.float32, copy=True)
        for position, row in enumerate(index)
        if row.get("dataset") == data["query_embeddings"]["train_key"]
    }


def nearest_negative(index: faiss.Index, query: np.ndarray, prohibited: set[int]) -> int:
    _scores, positions = index.search(np.ascontiguousarray(query[None, :]), 64)
    for position in positions[0]:
        if int(position) >= 0 and int(position) not in prohibited:
            return int(position)
    raise RuntimeError("hard-negative lookup exhausted")


def load_records(data: dict, vectors: np.ndarray, ids: list[str], folds: int) -> tuple[dict[int, list[dict]], dict[int, list[dict]]]:
    query_vectors = load_query_vectors(data)
    index_by_id = {memory_id: position for position, memory_id in enumerate(ids)}
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(np.ascontiguousarray(vectors))
    groups_by_fold: dict[int, list[dict]] = defaultdict(list)
    queries_by_fold: dict[int, list[dict]] = defaultdict(list)
    for row in iter_jsonl(data["split"]["train_query_plan"]):
        query_id = str(row["query_id"])
        fold = fold_for(str(row["persona_id"]), int(row["query_session_id"]), folds)
        groups = [
            list(dict.fromkeys(map(str, group.get("acceptable_memory_ids", []))))
            for group in row.get("evidence_groups", [])
        ]
        groups = [group for group in groups if group]
        queries_by_fold[fold].append({
            "query_id": query_id,
            "persona_id": str(row["persona_id"]),
            "query_session_id": int(row["query_session_id"]),
            "groups": groups,
        })
        for group in groups:
            members = [index_by_id[memory_id] for memory_id in group]
            groups_by_fold[fold].append({
                "query": query_vectors[query_id],
                "members": members,
                "hard_negative": nearest_negative(index, query_vectors[query_id], set(members)),
            })
    return groups_by_fold, queries_by_fold


def batch_loss(adapter, memory, records, temperature, device):
    candidate_ids = list(dict.fromkeys(index for record in records for index in [*record["members"], record["hard_negative"]]))
    position = {index: offset for offset, index in enumerate(candidate_ids)}
    anchors, masks = [], []
    for record in records:
        mask = torch.zeros(len(candidate_ids), dtype=torch.bool, device=device)
        mask[[position[index] for index in record["members"]]] = True
        anchors.append(torch.as_tensor(record["query"], dtype=torch.float32, device=device))
        masks.append(mask)
        for member in record["members"] if len(record["members"]) >= 2 else []:
            anchors.append(memory[member])
            member_mask = mask.clone()
            member_mask[position[member]] = False
            masks.append(member_mask)
    return multi_positive_info_nce(
        adapter,
        torch.stack(anchors),
        memory[torch.as_tensor(candidate_ids, dtype=torch.long, device=device)],
        torch.stack(masks),
        temperature,
    )


def train_adapter(memory, train, *, seed, bottleneck, epochs, learning_rate, config, device):
    torch.manual_seed(seed)
    adapter = ResidualProjectionAdapter(dimension=memory.shape[1], bottleneck=bottleneck).to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=learning_rate, weight_decay=float(config["adapter"]["weight_decay"]))
    rng = np.random.default_rng(seed)
    for _epoch in range(epochs):
        adapter.train()
        order = rng.permutation(len(train))
        for start in range(0, len(train), 64):
            batch = [train[index] for index in order[start : start + 64]]
            optimizer.zero_grad(set_to_none=True)
            loss = batch_loss(adapter, memory, batch, float(config["adapter"]["temperature"]), device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            optimizer.step()
    return adapter


@torch.no_grad()
def validation_loss(adapter, memory, records, temperature, device):
    adapter.eval()
    losses = [batch_loss(adapter, memory, records[start : start + 64], temperature, device) for start in range(0, len(records), 64)]
    return float(torch.stack(losses).mean().item())


def group_colocation(groups, memberships):
    values = []
    for group in groups:
        assigned = [memberships[memory_id] for memory_id in group]
        values.append(float(bool(set.intersection(*assigned))))
    return float(np.mean(values)) if values else 0.0


def knn_colocation(groups, index_by_id, vectors, index):
    values = []
    for group in groups:
        positions = [index_by_id[memory_id] for memory_id in group]
        _scores, neighbors = index.search(np.ascontiguousarray(vectors[positions]), 33)
        target = set(positions)
        values.append(float(any(target <= set(map(int, row)) for row in neighbors)))
    return float(np.mean(values)) if values else 0.0


def fit_recipe(projected, recipe, seed, uot_config, device):
    initial = farthest_point_prototypes(projected, 64, seed, device)
    scale = robust_cost_scale(projected, initial, device)
    prototypes, incidence, diagnostics = fit_uot(
        projected,
        initial,
        epsilon=max(1e-5, scale * float(recipe["epsilon_mult"])),
        tau_memory=float(recipe["tau_memory"]),
        tau_prototype=float(recipe["tau_prototype"]),
        dustbin_prior_mass=float(uot_config["uot"]["dustbin_prior_mass"]),
        dustbin_cost=scale * 0.5,
        outer_iterations=2,
        sinkhorn_iterations=int(uot_config["uot"]["sinkhorn_iterations"]),
        sinkhorn_tolerance=float(uot_config["uot"]["sinkhorn_tolerance"]),
        diversity_mode=str(recipe["diversity_mode"]),
        diversity_step_size=float(recipe["diversity_step_size"]),
        diversity_steps=int(uot_config["uot"]["prototype_diversity"]["proximal_steps"]),
        device=device,
    )
    return prototypes, incidence, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one v21 projection/UOT outer OOF unit.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--run-id", default="v21_phase1_oof_r1_20260714")
    args = parser.parse_args()
    phase = load_config("v21_phase1_oof.json")
    data = load_config("v21_data.json")
    uot_config = load_config("v21_uot_semantic.json")
    if args.seed not in phase["fixed_seeds"] or not 0 <= args.fold < int(phase["outer_folds"]):
        raise ValueError("seed/fold is outside the fixed v21 OOF design")
    run_dir = VERSION_ROOT / "artifacts" / "v21_phase1" / args.run_id
    unit_dir = run_dir / f"seed_{args.seed}" / f"fold_{args.fold}"
    unit_dir.mkdir(parents=True, exist_ok=False)
    random.seed(args.seed + args.fold)
    np.random.seed(args.seed + args.fold)
    torch.manual_seed(args.seed + args.fold)

    ids = list(map(str, read_json(data["memory"]["embedding_ids"])))
    vectors = np.load(resolve(data["memory"]["embedding_vectors"]), mmap_mode="r").astype(np.float32)
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    groups_by_fold, queries_by_fold = load_records(data, vectors, ids, int(phase["outer_folds"]))
    outer_train_folds = [fold for fold in range(5) if fold != args.fold]
    inner_validation_fold = (args.fold + int(phase["inner_validation_fold_offset"])) % int(phase["outer_folds"])
    if inner_validation_fold not in outer_train_folds:
        raise RuntimeError("inner validation fold must differ from the outer heldout fold")
    inner_train = [record for fold in outer_train_folds if fold != inner_validation_fold for record in groups_by_fold[fold]]
    inner_validation = groups_by_fold[inner_validation_fold]
    memory = torch.as_tensor(vectors, dtype=torch.float32, device=args.device)

    adapter_trials = []
    best = None
    for bottleneck in phase["adapter_grid"]["bottlenecks"]:
        for epochs in phase["adapter_grid"]["epochs"]:
            for learning_rate in phase["adapter_grid"]["learning_rates"]:
                trial_seed = args.seed + args.fold * 1000 + int(bottleneck) + int(epochs)
                adapter = train_adapter(
                    memory, inner_train, seed=trial_seed, bottleneck=int(bottleneck), epochs=int(epochs),
                    learning_rate=float(learning_rate), config=uot_config, device=args.device,
                )
                loss = validation_loss(adapter, memory, inner_validation, float(uot_config["adapter"]["temperature"]), args.device)
                row = {"bottleneck": int(bottleneck), "epochs": int(epochs), "learning_rate": float(learning_rate), "inner_validation_loss": loss}
                adapter_trials.append(row)
                if best is None or (loss, int(bottleneck), int(epochs), float(learning_rate)) < best[0]:
                    best = ((loss, int(bottleneck), int(epochs), float(learning_rate)), adapter, row)
    assert best is not None
    selected_inner_adapter = best[1]
    projected_inner = project_matrix(selected_inner_adapter, memory).detach().cpu().numpy()

    screen_path = VERSION_ROOT / "aggregate" / "v21_uot_full_numerical_screen_v21_uot_screen_r7_20260714_cuda2.csv"
    with screen_path.open(encoding="utf-8") as handle:
        screen = {int(row["recipe_id"]): row for row in csv.DictReader(handle)}
    recipes = build_preregistered_recipes(uot_config)
    eligible_recipes = [
        recipe for recipe in recipes
        if str(screen[recipe["recipe_id"]]["transport_converged"]).lower() == "true"
        and str(screen[recipe["recipe_id"]]["prototype_collapse_flag"]).lower() == "false"
    ]
    inner_queries = queries_by_fold[inner_validation_fold]
    inner_groups = [group for query in inner_queries for group in query["groups"]]
    recipe_trials = []
    for recipe in eligible_recipes:
        prototypes, incidence, diagnostics = fit_recipe(
            projected_inner, recipe, args.seed + args.fold, uot_config, args.device
        )
        memberships = {
            memory_id: set(values)
            for memory_id, values in zip(ids, sparse_memberships(incidence, 0.9, 4), strict=True)
        }
        colocation = group_colocation(inner_groups, memberships)
        recipe_trials.append({"recipe": recipe, "inner_group_colocation": colocation, "diagnostics": diagnostics})
    viable = [trial for trial in recipe_trials if trial["diagnostics"]["transport_converged"] and not trial["diagnostics"]["prototype_collapse_flag"]]
    if not viable:
        raise RuntimeError("no numerically viable UOT recipe in nested inner validation")
    selected_recipe_trial = min(viable, key=lambda trial: (
        -trial["inner_group_colocation"],
        trial["diagnostics"]["single_prototype_domination"],
        -trial["diagnostics"]["prototype_effective_count"],
        trial["diagnostics"]["build_ms"],
        trial["recipe"]["recipe_id"],
    ))

    selected = best[2]
    outer_train = [record for fold in outer_train_folds for record in groups_by_fold[fold]]
    final_adapter = train_adapter(
        memory, outer_train, seed=args.seed + args.fold * 10000, bottleneck=selected["bottleneck"],
        epochs=selected["epochs"], learning_rate=selected["learning_rate"], config=uot_config, device=args.device,
    )
    projected = project_matrix(final_adapter, memory).detach().cpu().numpy()
    prototypes, incidence, diagnostics = fit_recipe(
        projected, selected_recipe_trial["recipe"], args.seed + args.fold, uot_config, args.device
    )
    if not diagnostics["transport_converged"] or diagnostics["prototype_collapse_flag"]:
        raise RuntimeError(f"outer unit transport invalid: {diagnostics}")
    memberships = {
        memory_id: set(values)
        for memory_id, values in zip(ids, sparse_memberships(incidence, 0.9, 4), strict=True)
    }
    index_by_id = {memory_id: position for position, memory_id in enumerate(ids)}
    raw_index = faiss.IndexFlatIP(vectors.shape[1])
    raw_index.add(np.ascontiguousarray(vectors))
    query_rows = []
    for query in queries_by_fold[args.fold]:
        query_rows.append({
            "seed": args.seed,
            "fold": args.fold,
            "query_id": query["query_id"],
            "persona_id": query["persona_id"],
            "query_session_id": query["query_session_id"],
            "group_count": len(query["groups"]),
            "uot_fsa_group_colocation": group_colocation(query["groups"], memberships),
            "cosine_knn_group_colocation": knn_colocation(query["groups"], index_by_id, vectors, raw_index),
            "recipe_id": selected_recipe_trial["recipe"]["recipe_id"],
        })

    checkpoint_path = unit_dir / "v21_projection.pt"
    prototype_path = unit_dir / "v21_uot_prototypes.npy"
    incidence_path = unit_dir / "v21_uot_semantic_incidence.npy"
    metrics_path = unit_dir / "v21_query_metrics.csv"
    torch.save({
        "state_dict": {key: value.detach().cpu() for key, value in final_adapter.state_dict().items()},
        "seed": args.seed, "fold": args.fold, "selected_adapter": selected,
    }, checkpoint_path)
    with prototype_path.open("xb") as handle:
        np.save(handle, prototypes, allow_pickle=False)
    with incidence_path.open("xb") as handle:
        np.save(handle, incidence, allow_pickle=False)
    write_csv_new(metrics_path, query_rows)
    result = {
        "schema_version": "v21-phase1-oof-unit-1",
        "status": "completed",
        "run_id": args.run_id,
        "seed": args.seed,
        "fold": args.fold,
        "inner_validation_fold": inner_validation_fold,
        "adapter_trials": adapter_trials,
        "selected_adapter": selected,
        "eligible_recipe_ids": [recipe["recipe_id"] for recipe in eligible_recipes],
        "recipe_trials": recipe_trials,
        "selected_recipe": selected_recipe_trial["recipe"],
        "outer_transport_diagnostics": diagnostics,
        "outer_query_count": len(query_rows),
        "artifacts": {
            "projection_checkpoint": str(checkpoint_path.relative_to(VERSION_ROOT)),
            "prototype": str(prototype_path.relative_to(VERSION_ROOT)),
            "incidence": str(incidence_path.relative_to(VERSION_ROOT)),
            "query_metrics": str(metrics_path.relative_to(VERSION_ROOT)),
        },
        "hashes": {
            "projection_checkpoint": sha256_file(checkpoint_path),
            "prototype": sha256_file(prototype_path),
            "incidence": sha256_file(incidence_path),
            "query_metrics": sha256_file(metrics_path),
            "prototype_array": array_sha256(prototypes),
            "incidence_array": array_sha256(incidence),
            "phase_config": sha256_file(VERSION_ROOT / "configs" / "v21_phase1_oof.json"),
            "numerical_screen": sha256_file(screen_path),
        },
        "calibration_read": False,
        "formal_test_read": False,
    }
    write_json_new(unit_dir / "v21_unit_manifest.json", result)
    print(json.dumps({"seed": args.seed, "fold": args.fold, "queries": len(query_rows), "recipe_id": result["selected_recipe"]["recipe_id"]}, sort_keys=True))


if __name__ == "__main__":
    main()
