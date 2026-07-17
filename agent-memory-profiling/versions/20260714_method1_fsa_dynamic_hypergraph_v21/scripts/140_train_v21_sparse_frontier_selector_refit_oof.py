#!/usr/bin/env python3
"""Train one candidate utility selector on actual outer-train sparse frontiers."""
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
from v21.sparse_frontier_selector import SparseFrontierUtilityMLP, group_target_cross_entropy  # noqa: E402
from v21.uot_recipes import array_sha256  # noqa: E402


WORKER = runpy.run_path(str(VERSION_ROOT / "scripts/128_train_v21_sparse_joint_tuned_oof_r2.py"), run_name="v21_sparse_refit_helpers")
QESA_HELPERS = WORKER["QESA_HELPERS"]
ATTEMPT_RE = re.compile(r"^attempt_[0-9]{3,}$")


def load_frontiers(config: dict, seed: int) -> tuple[dict[str, tuple[str, ...]], list[Path]]:
    output: dict[str, tuple[str, ...]] = {}
    paths: list[Path] = []
    for fold in range(int(config["outer_folds"])):
        path = VERSION_ROOT / "artifacts/v21_fixed_sparse_frontiers" / config["frontier_run_id"] / f"seed_{seed}" / f"fold_{fold}" / "attempts" / config["frontier_attempt"] / "v21_fixed_sparse_frontiers.jsonl"
        paths.append(path)
        for row in iter_jsonl(path):
            query_id = str(row["query_id"])
            if query_id in output:
                raise ValueError(f"duplicate fixed frontier {query_id}")
            output[query_id] = tuple(map(str, row["frontier_memory_ids"]))
    if len(output) != 1355:
        raise ValueError("fixed sparse frontiers are incomplete")
    return output, paths


def quality(selected_ids: tuple[str, ...], groups: list[list[str]]) -> dict[str, float]:
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
    config = load_config(args.config)
    data = load_config("v21_data.json")
    qesa_config = load_config("v21_qesa_r2.json")
    if args.run_id != config["run_id"] or args.seed not in config["fixed_seeds"] or not 0 <= args.fold < int(config["outer_folds"]):
        raise ValueError("run, seed, or fold outside refit contract")
    if not ATTEMPT_RE.fullmatch(args.attempt):
        raise ValueError("attempt must have the form attempt_NNN")
    if config.get("minimum_effect_threshold") is not None or config.get("calibration_read") is not False or config.get("formal_test_read") is not False:
        raise ValueError("thresholds and locked split reads are forbidden")
    unit_dir = VERSION_ROOT / "artifacts/v21_sparse_frontier_selector_refit" / args.run_id / f"seed_{args.seed}" / f"fold_{args.fold}" / "attempts" / args.attempt
    unit_dir.mkdir(parents=True, exist_ok=False)

    ids = tuple(map(str, read_json(data["memory"]["embedding_ids"])))
    index_by_id = {memory_id: position for position, memory_id in enumerate(ids)}
    raw = np.load(resolve(data["memory"]["embedding_vectors"]), mmap_mode="r").astype(np.float32)
    raw /= np.maximum(np.linalg.norm(raw, axis=1, keepdims=True), 1e-12)
    query_vectors = QESA_HELPERS["load_query_vectors"](data)
    rows_by_fold = QESA_HELPERS["load_rows"](data, query_vectors, int(config["outer_folds"]))
    all_rows = [row for fold in range(int(config["outer_folds"])) for row in rows_by_fold[fold]]
    row_by_id = {str(row["query_id"]): row for row in all_rows}
    train_rows = [row for fold in range(int(config["outer_folds"])) if fold != args.fold for row in rows_by_fold[fold]]
    heldout_rows = rows_by_fold[args.fold]
    frontiers, frontier_paths = load_frontiers(config, args.seed)
    incidence_path = VERSION_ROOT / "artifacts/v21_phase1" / config["phase1_run_id"] / f"seed_{args.seed}" / f"fold_{args.fold}" / "v21_uot_semantic_incidence.npy"
    incidence = np.load(incidence_path, mmap_mode="r")
    qesa_path = VERSION_ROOT / "artifacts/v21_qesa" / config["qesa_run_id"] / f"seed_{args.seed}" / f"fold_{args.fold}" / "v21_qesa_r2_models.pt"
    checkpoint = torch.load(qesa_path, map_location="cpu")
    qesa = WORKER["qesa_model"](checkpoint, raw.shape[1], int(qesa_config["hidden_dimension"]), args.device)
    predictions = WORKER["predict_qesa_batch"](qesa, np.stack([row["query_embedding"] for row in all_rows]), args.device)
    prediction_by_id = {str(row["query_id"]): prediction for row, prediction in zip(all_rows, predictions, strict=True)}

    def features_for(row: dict) -> np.ndarray:
        frontier = frontiers[str(row["query_id"])]
        positions = np.asarray([index_by_id[memory_id] for memory_id in frontier], dtype=np.int64)
        cosine = raw[positions] @ row["query_embedding"]
        association = incidence[positions] @ prediction_by_id[str(row["query_id"])]
        position_prior = 1.0 - np.arange(len(frontier), dtype=np.float32) / max(len(frontier) - 1, 1)
        return np.stack((cosine, association, cosine * association, position_prior), axis=1).astype(np.float32)

    train_features = np.stack([features_for(row) for row in train_rows])
    feature_mean = train_features.reshape(-1, train_features.shape[-1]).mean(axis=0)
    feature_std = train_features.reshape(-1, train_features.shape[-1]).std(axis=0)
    feature_std = np.maximum(feature_std, 1e-6)
    train_features = (train_features - feature_mean) / feature_std
    targets = np.zeros(train_features.shape[:2], dtype=np.float32)
    protected_mask = np.zeros(train_features.shape[:2], dtype=bool)
    protected_mask[:, : int(config["protected_ann_seeds"])] = True
    valid_examples = 0
    for query_index, row in enumerate(train_rows):
        frontier = frontiers[str(row["query_id"])]
        position_by_id = {memory_id: position for position, memory_id in enumerate(frontier)}
        protected = set(frontier[: int(config["protected_ann_seeds"])])
        group_targets = []
        for group in row["groups"]:
            if protected.intersection(group):
                continue
            positions = [position_by_id[memory_id] for memory_id in group if memory_id in position_by_id and position_by_id[memory_id] >= int(config["protected_ann_seeds"])]
            if positions:
                distribution = np.zeros(len(frontier), dtype=np.float32)
                distribution[positions] = 1.0 / len(positions)
                group_targets.append(distribution)
        if group_targets:
            targets[query_index] = np.mean(group_targets, axis=0)
            targets[query_index] /= max(float(targets[query_index].sum()), 1e-12)
            valid_examples += 1
    if valid_examples == 0:
        raise ValueError("no sparse-frontier residual supervision")

    train_x = torch.as_tensor(train_features, device=args.device)
    train_y = torch.as_tensor(targets, device=args.device)
    train_protected = torch.as_tensor(protected_mask, device=args.device)
    torch.manual_seed(args.seed + args.fold * 10000)
    model = SparseFrontierUtilityMLP(4, int(config["hidden_dimension"])).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    rng = np.random.default_rng(args.seed + args.fold)
    losses = []
    for _epoch in range(int(config["epochs"])):
        order = rng.permutation(len(train_rows))
        running = []
        for start in range(0, len(order), int(config["batch_size"])):
            batch = torch.as_tensor(order[start : start + int(config["batch_size"])], device=args.device)
            optimizer.zero_grad(set_to_none=True)
            scores = model(train_x[batch])
            loss = group_target_cross_entropy(scores, train_y[batch], train_protected[batch])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running.append(float(loss.detach().item()))
        losses.append(float(np.mean(running)))

    source_selection_path = VERSION_ROOT / "artifacts/v21_sparse_joint" / config["source_sparse_run_id"] / f"seed_{args.seed}" / f"fold_{args.fold}" / "attempts" / config["source_sparse_attempt"] / "v21_sparse_joint_selections.jsonl"
    baseline: dict[tuple[str, str, int], tuple[str, ...]] = {}
    for row in iter_jsonl(source_selection_path):
        if (
            str(row["method"]) in {"ann_flat", "sparse_joint_semantic_bundle_hypergraph"}
            and int(row["activated_prototypes"]) == 16
            and int(row["members_per_semantic_edge"]) == 32
            and int(row["semantic_bundle_beam"]) == 32
            and int(row["budget"]) in set(map(int, config["budgets"]))
        ):
            baseline[(str(row["query_id"]), str(row["method"]), int(row["budget"]))] = tuple(map(str, row["selected_memory_ids"]))
    neutral_path = VERSION_ROOT / "artifacts/v21_effect_closure/v21_neutral_memory_to_tile.csv"
    with neutral_path.open(newline="", encoding="utf-8") as handle:
        tile = {str(row["memory_id"]): str(row["tile_id"]) for row in csv.DictReader(handle)}
    payload = {str(row["memory_id"]): int(row["payload_bytes"]) for row in iter_jsonl(data["memory"]["items"])}

    model = model.to("cpu").eval()
    heldout_features = {str(row["query_id"]): (features_for(row) - feature_mean) / feature_std for row in heldout_rows}
    for row in heldout_rows[: min(50, len(heldout_rows))]:
        _ = model(torch.as_tensor(heldout_features[str(row["query_id"])], dtype=torch.float32))
    metrics: list[dict] = []
    selections: list[dict] = []
    for row in heldout_rows:
        query_id = str(row["query_id"])
        frontier = frontiers[query_id]
        start = perf_counter_ns()
        with torch.no_grad():
            scores = model(torch.as_tensor(heldout_features[query_id], dtype=torch.float32)).numpy()
        remaining = list(range(int(config["protected_ann_seeds"]), len(frontier)))
        order = list(range(int(config["protected_ann_seeds"]))) + sorted(remaining, key=lambda position: (-float(scores[position]), position))
        selector_ms = (perf_counter_ns() - start) / 1_000_000.0
        methods = {
            "ann_flat": {budget: baseline[(query_id, "ann_flat", budget)] for budget in map(int, config["budgets"])},
            "existing_qhds": {budget: baseline[(query_id, "sparse_joint_semantic_bundle_hypergraph", budget)] for budget in map(int, config["budgets"])},
            "sparse_frontier_refit_mlp": {budget: tuple(frontier[position] for position in order[:budget]) for budget in map(int, config["budgets"])},
        }
        for method_name, by_budget in methods.items():
            for budget, selected in by_budget.items():
                values = quality(selected, row["groups"])
                metrics.append({
                    "seed": args.seed, "fold": args.fold, "query_id": query_id, "method": method_name, "budget": budget,
                    **values,
                    "selected_payload_tile_activations": len({tile[memory_id] for memory_id in selected}),
                    "payload_bytes": sum(payload[memory_id] for memory_id in selected),
                    "selector_ms": selector_ms if method_name == "sparse_frontier_refit_mlp" else 0.0,
                })
                selections.append({"seed": args.seed, "fold": args.fold, "query_id": query_id, "method": method_name, "budget": budget, "selected_memory_ids": list(selected)})

    metrics_path = unit_dir / "v21_sparse_frontier_selector_refit_metrics.csv"
    selections_path = unit_dir / "v21_sparse_frontier_selector_refit_selections.jsonl"
    model_path = unit_dir / "v21_sparse_frontier_selector_refit_model.pt"
    write_csv_new(metrics_path, metrics)
    with selections_path.open("x", encoding="utf-8") as handle:
        for row in selections:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    torch.save({
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "training_loss_first": losses[0],
        "training_loss_last": losses[-1],
    }, model_path)
    manifest = {
        "schema_version": "v21-sparse-frontier-selector-refit-oof-unit-1",
        "status": "completed",
        "run_id": args.run_id,
        "attempt": args.attempt,
        "seed": args.seed,
        "fold": args.fold,
        "outer_train_queries": len(train_rows),
        "outer_heldout_queries": len(heldout_rows),
        "valid_training_queries": valid_examples,
        "metric_rows": len(metrics),
        "selection_rows": len(selections),
        "training_loss_first": losses[0],
        "training_loss_last": losses[-1],
        "actual_sparse_frontier_training": True,
        "target_fold_excluded": True,
        "calibration_read": False,
        "formal_test_read": False,
        "artifacts": {"metrics": str(metrics_path.relative_to(VERSION_ROOT)), "selections": str(selections_path.relative_to(VERSION_ROOT)), "model": str(model_path.relative_to(VERSION_ROOT))},
        "hashes": {
            "config": sha256_file(VERSION_ROOT / "configs" / args.config),
            "incidence": array_sha256(incidence),
            "qesa_checkpoint": sha256_file(qesa_path),
            "source_selections": sha256_file(source_selection_path),
            **{f"frontier_fold_{fold}": sha256_file(path) for fold, path in enumerate(frontier_paths)},
            "metrics": sha256_file(metrics_path), "selections": sha256_file(selections_path), "model": sha256_file(model_path),
        },
    }
    write_json_new(unit_dir / "v21_sparse_frontier_selector_refit_manifest.json", manifest)
    b32 = [row for row in metrics if row["budget"] == 32]
    pooled = {method: float(np.mean([row["group_recall"] for row in b32 if row["method"] == method])) for method in ("ann_flat", "existing_qhds", "sparse_frontier_refit_mlp")}
    print(json.dumps({"status": "completed", "seed": args.seed, "fold": args.fold, "B32_group_recall": pooled, "loss_first": losses[0], "loss_last": losses[-1]}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="v21_sparse_frontier_selector_refit_oof_r1.json")
    parser.add_argument("--run-id", default="v21_sparse_frontier_selector_refit_oof_r1_20260715")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--attempt", default="attempt_001")
    parser.add_argument("--device", default="cuda:0")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
