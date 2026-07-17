#!/usr/bin/env python3
"""Train a stateful hypergraph-aware selector on retained bundle provenance."""
from __future__ import annotations

import argparse
import csv
import json
import re
import runpy
import sys
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import torch


VERSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VERSION_ROOT / "src"))

from v21.io import iter_jsonl, load_config, read_json, resolve, sha256_file, write_csv_new, write_json_new  # noqa: E402
from v21.sparse_frontier_selector import StructureAwareMarginalSelector, group_target_cross_entropy, structure_marginal_features  # noqa: E402
from v21.uot_recipes import array_sha256  # noqa: E402


WORKER = runpy.run_path(str(VERSION_ROOT / "scripts/128_train_v21_sparse_joint_tuned_oof_r2.py"), run_name="v21_structure_selector_helpers")
ATTEMPT_RE = re.compile(r"^attempt_[0-9]{3,}$")


def load_provenance(config: dict, target_seed: int) -> tuple[dict[str, dict], list[Path], list[dict]]:
    values: dict[str, dict] = {}
    paths: list[Path] = []
    manifests: list[dict] = []
    for seed in (int(target_seed),):
        for fold in range(int(config["outer_folds"])):
            unit = VERSION_ROOT / "artifacts/v21_sparse_frontier_provenance" / config["provenance_run_id"] / f"seed_{seed}" / f"fold_{fold}" / "attempts" / config["provenance_attempt"]
            manifest_path = unit / "v21_sparse_frontier_provenance_manifest.json"
            path = unit / "v21_sparse_frontier_provenance.jsonl"
            manifest = read_json(manifest_path)
            if manifest.get("status") != "completed" or manifest.get("exact_frontier_equivalence") is not True:
                raise ValueError(f"invalid provenance unit {manifest_path}")
            if manifest.get("calibration_read") is not False or manifest.get("formal_test_read") is not False or manifest.get("minimum_effect_threshold") is not None:
                raise ValueError(f"provenance contract violation {manifest_path}")
            if sha256_file(path) != manifest["hashes"]["provenance"]:
                raise ValueError(f"provenance hash mismatch {path}")
            for row in iter_jsonl(path):
                query_id = str(row["query_id"])
                if query_id in values:
                    raise ValueError(f"duplicate provenance query {query_id}")
                values[query_id] = row
            paths.append(path)
            manifests.append(manifest)
    return values, paths, manifests


def matrices(provenance: dict, ids: tuple[str, ...], raw: np.ndarray, incidence: np.ndarray, aspect: np.ndarray, query: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    frontier_ids = tuple(map(str, provenance["frontier_memory_ids"]))
    positions = np.asarray([ids.index(memory_id) for memory_id in frontier_ids], dtype=np.int64)
    cosine = raw[positions] @ query
    association = incidence[positions] @ aspect
    prior = 1.0 - np.arange(len(positions), dtype=np.float32) / max(len(positions) - 1, 1)
    unary = np.stack((cosine, association, cosine * association, prior), axis=1).astype(np.float32)
    bundles = provenance["retained_semantic_bundles"]
    bundle_incidence = np.zeros((len(bundles), len(frontier_ids)), dtype=bool)
    bundle_weights = np.zeros(len(bundles), dtype=np.float32)
    local = {memory_id: index for index, memory_id in enumerate(frontier_ids)}
    for bundle_index, bundle in enumerate(bundles):
        bundle_weights[bundle_index] = float(bundle["score"])
        for memory_id in bundle["memory_ids"]:
            if memory_id in local:
                bundle_incidence[bundle_index, local[memory_id]] = True
    if len(bundle_weights):
        shifted = bundle_weights - float(bundle_weights.max())
        bundle_weights = np.exp(shifted).astype(np.float32)
        bundle_weights /= max(float(bundle_weights.sum()), 1e-12)
    active = provenance["active_prototypes"]
    prototype_ids = [int(item["prototype_id"]) for item in active]
    prototype_incidence = np.zeros((len(prototype_ids), len(frontier_ids)), dtype=bool)
    prototype_weights = np.asarray([max(float(item["weight"]), 0.0) for item in active], dtype=np.float32)
    prototype_index = {prototype: index for index, prototype in enumerate(prototype_ids)}
    for edge in provenance["semantic_edges"]:
        prototype = int(edge["prototype_id"])
        if prototype not in prototype_index:
            continue
        for memory_id in edge["memory_ids"]:
            if memory_id in local:
                prototype_incidence[prototype_index[prototype], local[memory_id]] = True
    prototype_weights /= max(float(prototype_weights.sum()), 1e-12)
    similarity = (raw[positions] @ raw[positions].T).astype(np.float32)
    protected_count = len(provenance["protected_ann_memory_ids"])
    return unary, bundle_incidence, bundle_weights, prototype_incidence, prototype_weights, similarity, {"protected_count": protected_count, "frontier_size": len(frontier_ids)}


def quality(selected_ids: list[str], groups: list[list[str]]) -> dict[str, float]:
    selected = set(selected_ids)
    covered = [bool(selected.intersection(group)) for group in groups]
    relevant = {memory_id for group in groups for memory_id in group}
    hits = len(selected.intersection(relevant))
    return {
        "group_recall": float(np.mean(covered)) if covered else 0.0,
        "all_groups_covered": float(all(covered)) if covered else 0.0,
        "evidence_member_recall": hits / max(len(relevant), 1),
        "evidence_member_precision": hits / max(len(selected_ids), 1),
    }


def run(args: argparse.Namespace) -> None:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    config = load_config(args.config)
    data = load_config("v21_data.json")
    qesa_config = load_config("v21_qesa_r2.json")
    if args.run_id != config["run_id"] or args.seed not in config["fixed_seeds"] or not 0 <= args.fold < int(config["outer_folds"]):
        raise ValueError("run, seed, or fold outside selector contract")
    if not ATTEMPT_RE.fullmatch(args.attempt) or config.get("minimum_effect_threshold") is not None or config.get("calibration_read") is not False or config.get("formal_test_read") is not False:
        raise ValueError("attempt or locked split contract violation")
    unit_dir = VERSION_ROOT / "artifacts/v21_structure_aware_selector" / args.run_id / f"seed_{args.seed}" / f"fold_{args.fold}" / "attempts" / args.attempt
    unit_dir.mkdir(parents=True, exist_ok=False)

    ids = tuple(map(str, read_json(data["memory"]["embedding_ids"])))
    index_by_id = {memory_id: position for position, memory_id in enumerate(ids)}
    raw = np.load(resolve(data["memory"]["embedding_vectors"]), mmap_mode="r").astype(np.float32)
    raw /= np.maximum(np.linalg.norm(raw, axis=1, keepdims=True), 1e-12)
    query_vectors = WORKER["QESA_HELPERS"]["load_query_vectors"](data)
    rows_by_fold = WORKER["QESA_HELPERS"]["load_rows"](data, query_vectors, int(config["outer_folds"]))
    all_rows = [row for fold in range(int(config["outer_folds"])) for row in rows_by_fold[fold]]
    row_by_id = {str(row["query_id"]): row for row in all_rows}
    train_rows = [row for fold in range(int(config["outer_folds"])) if fold != args.fold for row in rows_by_fold[fold]]
    heldout_rows = rows_by_fold[args.fold]
    provenance, provenance_paths, provenance_manifests = load_provenance(config, args.seed)
    incidence_path = VERSION_ROOT / "artifacts/v21_phase1" / config["phase1_run_id"] / f"seed_{args.seed}" / f"fold_{args.fold}" / "v21_uot_semantic_incidence.npy"
    incidence = np.load(incidence_path, mmap_mode="r")
    qesa_path = VERSION_ROOT / "artifacts/v21_qesa" / config["qesa_run_id"] / f"seed_{args.seed}" / f"fold_{args.fold}" / "v21_qesa_r2_models.pt"
    qesa = WORKER["qesa_model"](torch.load(qesa_path, map_location="cpu"), raw.shape[1], int(qesa_config["hidden_dimension"]), args.device)
    aspects = WORKER["predict_qesa_batch"](qesa, np.stack([row["query_embedding"] for row in all_rows]), args.device)
    aspect_by_id = {str(row["query_id"]): aspect for row, aspect in zip(all_rows, aspects, strict=True)}

    # Build one fixed feature normalization from outer-train queries only.
    train_mats = {str(row["query_id"]): matrices(provenance[str(row["query_id"])], ids, raw, incidence, aspect_by_id[str(row["query_id"])], row["query_embedding"]) for row in train_rows}
    unary_stack = np.concatenate([item[0] for item in train_mats.values()], axis=0)
    unary_mean = unary_stack.mean(axis=0).astype(np.float32)
    unary_std = np.maximum(unary_stack.std(axis=0), 1e-6).astype(np.float32)

    example_unary, example_structural, example_targets, example_protected = [], [], [], []
    rollout_steps = int(config["training_rollout_steps"])
    for row in train_rows:
        query_id = str(row["query_id"])
        unary, bundle_inc, bundle_w, proto_inc, proto_w, similarity, info = train_mats[query_id]
        unary = (unary - unary_mean) / unary_std
        selected = np.zeros(len(unary), dtype=bool)
        selected[: info["protected_count"]] = True
        position_by_id = {memory_id: position for position, memory_id in enumerate(provenance[query_id]["frontier_memory_ids"])}
        for _step in range(rollout_steps):
            remaining_groups = []
            for group in row["groups"]:
                positions = [position_by_id[memory_id] for memory_id in group if memory_id in position_by_id and not selected[position_by_id[memory_id]]]
                if positions and not any(selected[position_by_id[memory_id]] for memory_id in group if memory_id in position_by_id):
                    remaining_groups.append(positions)
            if not remaining_groups:
                break
            target = np.zeros(len(unary), dtype=np.float32)
            for positions in remaining_groups:
                target[positions] += 1.0 / len(positions) / len(remaining_groups)
            structural = structure_marginal_features(selected, bundle_inc, bundle_w, proto_inc, proto_w, similarity)
            example_unary.append(unary.copy())
            example_structural.append(structural)
            example_targets.append(target)
            example_protected.append(selected.copy())
            chosen = int(np.argmax(np.where(target > 0, unary[:, 0] + target, -np.inf)))
            selected[chosen] = True
    if not example_unary:
        raise ValueError("no structure-aware training examples")
    train_unary = torch.as_tensor(np.stack(example_unary), dtype=torch.float32, device=args.device)
    train_structural = torch.as_tensor(np.stack(example_structural), dtype=torch.float32, device=args.device)
    train_targets = torch.as_tensor(np.stack(example_targets), dtype=torch.float32, device=args.device)
    train_protected = torch.as_tensor(np.stack(example_protected), dtype=torch.bool, device=args.device)
    torch.manual_seed(args.seed + args.fold * 10000)
    model = StructureAwareMarginalSelector(4, 5, int(config["hidden_dimension"])).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    rng = np.random.default_rng(args.seed + args.fold)
    losses = []
    for _epoch in range(int(config["epochs"])):
        order = rng.permutation(len(train_unary))
        batch_losses = []
        for start in range(0, len(order), int(config["batch_size"])):
            batch = torch.as_tensor(order[start:start + int(config["batch_size"])], dtype=torch.long, device=args.device)
            optimizer.zero_grad(set_to_none=True)
            scores = model(train_unary[batch], train_structural[batch])
            loss = group_target_cross_entropy(scores, train_targets[batch], train_protected[batch])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu().item()))
        losses.append(float(np.mean(batch_losses)))

    source_selection_path = VERSION_ROOT / "artifacts/v21_sparse_joint" / config["source_sparse_run_id"] / f"seed_{args.seed}" / f"fold_{args.fold}" / "attempts" / config["source_sparse_attempt"] / "v21_sparse_joint_selections.jsonl"
    baseline: dict[tuple[str, int], tuple[str, ...]] = {}
    for row in iter_jsonl(source_selection_path):
        if str(row["method"]) == "sparse_joint_semantic_bundle_hypergraph" and int(row["activated_prototypes"]) == 16 and int(row["members_per_semantic_edge"]) == 32 and int(row["semantic_bundle_beam"]) == 32 and int(row["budget"]) in set(map(int, config["budgets"])):
            baseline[(str(row["query_id"]), int(row["budget"]))] = tuple(map(str, row["selected_memory_ids"]))
    tile_path = VERSION_ROOT / "artifacts/v21_effect_closure/v21_neutral_memory_to_tile.csv"
    with tile_path.open(newline="", encoding="utf-8") as handle:
        tile = {str(row["memory_id"]): str(row["tile_id"]) for row in csv.DictReader(handle)}
    payload = {str(row["memory_id"]): int(row["payload_bytes"]) for row in iter_jsonl(data["memory"]["items"])}
    model = model.to("cpu").eval()
    metrics, selections = [], []
    for row in heldout_rows:
        query_id = str(row["query_id"])
        unary, bundle_inc, bundle_w, proto_inc, proto_w, similarity, info = matrices(provenance[query_id], ids, raw, incidence, aspect_by_id[query_id], row["query_embedding"])
        unary = (unary - unary_mean) / unary_std
        selected = np.zeros(len(unary), dtype=bool)
        selected[: info["protected_count"]] = True
        order = list(range(info["protected_count"]))
        start = perf_counter_ns()
        while len(order) < max(map(int, config["budgets"])):
            structural = structure_marginal_features(selected, bundle_inc, bundle_w, proto_inc, proto_w, similarity)
            with torch.no_grad():
                scores = model(torch.as_tensor(unary, dtype=torch.float32), torch.as_tensor(structural, dtype=torch.float32)).numpy()
            scores[selected] = -np.inf
            chosen = int(np.argmax(scores))
            selected[chosen] = True
            order.append(chosen)
        selector_ms = (perf_counter_ns() - start) / 1_000_000.0
        frontier_ids = list(map(str, provenance[query_id]["frontier_memory_ids"]))
        ann_order = list(range(info["protected_count"])) + sorted(range(info["protected_count"], len(frontier_ids)), key=lambda position: (-float(unary[position, 0]), position))
        for method, method_order in (("same_frontier_cosine", ann_order), ("existing_qhds", [frontier_ids.index(memory_id) for memory_id in baseline[(query_id, 32)]] + []), ("structure_aware_selector", order)):
            if method == "existing_qhds":
                method_order = [frontier_ids.index(memory_id) for memory_id in baseline[(query_id, 32)]]
            for budget in map(int, config["budgets"]):
                selected_ids = [frontier_ids[position] for position in method_order[:budget]]
                values = quality(selected_ids, row["groups"])
                values.update({"seed": args.seed, "fold": args.fold, "query_id": query_id, "method": method, "budget": budget, "selected_payload_tile_activations": len({tile[memory_id] for memory_id in selected_ids}), "payload_bytes": sum(payload[memory_id] for memory_id in selected_ids), "selector_ms": selector_ms if method == "structure_aware_selector" else 0.0})
                metrics.append(values)
                selections.append({"seed": args.seed, "fold": args.fold, "query_id": query_id, "method": method, "budget": budget, "selected_memory_ids": selected_ids})

    metrics_path = unit_dir / "v21_structure_aware_selector_metrics.csv"
    selections_path = unit_dir / "v21_structure_aware_selector_selections.jsonl"
    model_path = unit_dir / "v21_structure_aware_selector_model.pt"
    write_csv_new(metrics_path, metrics)
    with selections_path.open("x", encoding="utf-8") as handle:
        for row in selections:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    torch.save({"state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()}, "unary_mean": unary_mean, "unary_std": unary_std, "training_examples": len(example_unary), "training_loss_first": losses[0], "training_loss_last": losses[-1]}, model_path)
    manifest = {
        "schema_version": "v21-structure-aware-selector-oof-unit-1", "status": "completed", "run_id": args.run_id, "attempt": args.attempt, "seed": args.seed, "fold": args.fold,
        "outer_train_queries": len(train_rows), "outer_heldout_queries": len(heldout_rows), "training_examples": len(example_unary), "metric_rows": len(metrics), "selection_rows": len(selections),
        "architecture": "separate unary and stateful hypergraph marginal networks; bundle completion/progress, prototype coverage, and embedding diversity",
        "target_fold_excluded": True, "calibration_read": False, "formal_test_read": False, "minimum_effect_threshold": None,
        "artifacts": {"metrics": str(metrics_path.relative_to(VERSION_ROOT)), "selections": str(selections_path.relative_to(VERSION_ROOT)), "model": str(model_path.relative_to(VERSION_ROOT))},
        "hashes": {"config": sha256_file(VERSION_ROOT / "configs" / args.config), "incidence": array_sha256(incidence), "qesa_checkpoint": sha256_file(qesa_path), "source_selections": sha256_file(source_selection_path), "metrics": sha256_file(metrics_path), "selections": sha256_file(selections_path), "model": sha256_file(model_path), **{f"provenance_{index}": sha256_file(path) for index, path in enumerate(provenance_paths)}},
        "provenance_manifests": [{"seed": manifest["seed"], "fold": manifest["fold"], "hash": manifest["hashes"]["provenance"]} for manifest in provenance_manifests],
        "training_loss_first": losses[0], "training_loss_last": losses[-1],
    }
    write_json_new(unit_dir / "v21_structure_aware_selector_manifest.json", manifest)
    print(json.dumps({"status": "completed", "seed": args.seed, "fold": args.fold, "examples": len(example_unary), "loss_first": losses[0], "loss_last": losses[-1]}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="v21_structure_aware_selector_oof_r1.json")
    parser.add_argument("--run-id", default="v21_structure_aware_selector_oof_r1_20260715")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--attempt", default="attempt_001")
    parser.add_argument("--device", default="cpu")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
