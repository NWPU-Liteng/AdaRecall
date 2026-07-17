#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path

import numpy as np
import torch


VERSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VERSION_ROOT / "src"))

from v21.io import load_config, read_json, resolve, sha256_file, write_csv_new, write_json_new  # noqa: E402
from v21.qesa_r2 import LatentAspectQESAR2  # noqa: E402
from v21.qhds import (  # noqa: E402
    MonotoneSubmodularQHDS,
    QHDSInferenceExample,
    build_features,
    greedy_select,
    group_contrastive_loss,
)
from v21.uot_recipes import array_sha256  # noqa: E402


QESA_HELPERS = runpy.run_path(str(VERSION_ROOT / "scripts" / "90_train_v21_qesa_oof.py"), run_name="v21_qesa_helpers")


@torch.no_grad()
def qesa_predictions(checkpoint: dict, queries: np.ndarray, hidden: int, device: str) -> np.ndarray:
    latent = int(checkpoint["selected_latent_aspects"])
    state = checkpoint["qesa"]
    prototypes = int(state["aspect_prototype_logits"].shape[1])
    model = LatentAspectQESAR2(
        queries.shape[1], hidden, latent, prototypes, float(checkpoint["selected_regime"]["aspect_temperature"])
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model(torch.as_tensor(queries, device=device))[2].cpu().numpy()


def prepare_training_examples(rows, predictions, incidence, raw, index_by_id, config, seed, device):
    examples = []
    rng = np.random.default_rng(seed)
    for row, aspect in zip(rows, predictions, strict=True):
        eligible = np.asarray([index_by_id[memory_id] for memory_id in row["eligible_memory_ids"]], dtype=np.int64)
        cosine = raw[eligible] @ row["query_embedding"]
        association = incidence[eligible] @ aspect
        ann_count = min(int(config["training_ann_candidates"]), len(eligible))
        assoc_count = min(int(config["training_association_candidates"]), len(eligible))
        positions = set(np.argsort(-cosine, kind="stable")[:ann_count].tolist())
        positions.update(np.argsort(-association, kind="stable")[:assoc_count].tolist())
        positive_global = {index_by_id[memory_id] for group in row["groups"] for memory_id in group}
        position_by_global = {int(global_index): position for position, global_index in enumerate(eligible)}
        positions.update(position_by_global[index] for index in positive_global)
        remaining = np.asarray(sorted(set(range(len(eligible))).difference(positions)), dtype=np.int64)
        random_count = min(int(config["training_random_candidates"]), len(remaining))
        if random_count:
            positions.update(rng.choice(remaining, size=random_count, replace=False).tolist())
        protected_positions = np.argsort(-cosine, kind="stable")[: int(config["protected_ann_seeds"])].tolist()
        positions.update(protected_positions)
        candidate_positions = np.asarray(sorted(positions), dtype=np.int64)
        candidate_global = eligible[candidate_positions]
        local_by_global = {int(index): local for local, index in enumerate(candidate_global)}
        selected = torch.zeros(len(candidate_global), dtype=torch.bool, device=device)
        selected[[local_by_global[int(eligible[position])] for position in protected_positions]] = True
        positive_masks = []
        for group in row["groups"]:
            mask = torch.zeros(len(candidate_global), dtype=torch.bool, device=device)
            for memory_id in group:
                mask[local_by_global[index_by_id[memory_id]]] = True
            positive_masks.append(mask)
        examples.append({
            "features": build_features(
                torch.as_tensor(cosine[candidate_positions], device=device),
                torch.as_tensor(association[candidate_positions], device=device),
            ),
            "incidence": torch.as_tensor(incidence[candidate_global], device=device),
            "aspects": torch.as_tensor(aspect, device=device),
            "selected": selected,
            "positive_masks": positive_masks,
        })
    return examples


def train_model(examples, config, seed, device):
    torch.manual_seed(seed)
    model = MonotoneSubmodularQHDS(3, int(config["hidden_dimension"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    rng = np.random.default_rng(seed)
    epoch_losses = []
    for _epoch in range(int(config["epochs"])):
        order = rng.permutation(len(examples))
        running = []
        optimizer.zero_grad(set_to_none=True)
        for count, position in enumerate(order, start=1):
            example = examples[int(position)]
            loss = group_contrastive_loss(
                model, example["features"], example["incidence"], example["aspects"], example["selected"],
                example["positive_masks"], float(config["contrastive_temperature"]), float(config["scale_regularization"]),
            )
            (loss / 16.0).backward()
            running.append(float(loss.detach().item()))
            if count % 16 == 0 or count == len(order):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        epoch_losses.append(float(np.mean(running)))
    structural, modular = model.positive_scales()
    return model, {
        "epoch_loss_first": epoch_losses[0],
        "epoch_loss_last": epoch_losses[-1],
        "structural_scale": float(structural.detach().item()),
        "modular_scale": float(modular.detach().item()),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }


def evidence_metrics(selected_ids, protected_ids, groups):
    selected, protected = set(selected_ids), set(protected_ids)
    covered = [bool(selected.intersection(group)) for group in groups]
    protected_covered = [bool(protected.intersection(group)) for group in groups]
    both = [len(selected.intersection(group)) >= 2 for group in groups]
    gain = sum(covered) - sum(protected_covered)
    added = max(0, len(selected_ids) - len(protected_ids))
    return {
        "group_coverage": float(np.mean(covered)),
        "all_groups_covered": float(all(covered)),
        "both_alternatives_selected_rate": float(np.mean(both)),
        "residual_uncovered_groups_after_protected": int(len(groups) - sum(protected_covered)),
        "coverage_gain_after_protected": float(gain / len(groups)),
        "unique_groups_covered_per_added_candidate": float(gain / added) if added else 0.0,
        "final_top5_group_coverage": float(np.mean(protected_covered)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one v21 QHDS outer OOF unit.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--run-id", default="v21_qhds_oof_r1_20260714")
    args = parser.parse_args()
    config = load_config("v21_qhds.json")
    qesa_config = load_config("v21_qesa_r2.json")
    data = load_config("v21_data.json")
    if args.run_id != config["run_id"] or args.seed not in config["fixed_seeds"] or not 0 <= args.fold < 5:
        raise ValueError("run ID, seed, or fold outside QHDS preregistration")
    unit_dir = VERSION_ROOT / "artifacts" / "v21_qhds" / args.run_id / f"seed_{args.seed}" / f"fold_{args.fold}"
    unit_dir.mkdir(parents=True, exist_ok=False)
    ids = list(map(str, read_json(data["memory"]["embedding_ids"])))
    index_by_id = {memory_id: position for position, memory_id in enumerate(ids)}
    raw = np.load(resolve(data["memory"]["embedding_vectors"]), mmap_mode="r").astype(np.float32)
    raw /= np.maximum(np.linalg.norm(raw, axis=1, keepdims=True), 1e-12)
    queries = QESA_HELPERS["load_query_vectors"](data)
    rows_by_fold = QESA_HELPERS["load_rows"](data, queries, 5)
    train_rows = [row for fold in range(5) if fold != args.fold for row in rows_by_fold[fold]]
    heldout_rows = rows_by_fold[args.fold]
    phase1_unit = VERSION_ROOT / "artifacts" / "v21_phase1" / config["phase1_run_id"] / f"seed_{args.seed}" / f"fold_{args.fold}"
    incidence_path = phase1_unit / "v21_uot_semantic_incidence.npy"
    incidence = np.load(incidence_path, allow_pickle=False)
    qesa_unit = VERSION_ROOT / "artifacts" / "v21_qesa" / config["qesa_run_id"] / f"seed_{args.seed}" / f"fold_{args.fold}"
    qesa_checkpoint_path = qesa_unit / "v21_qesa_r2_models.pt"
    checkpoint = torch.load(qesa_checkpoint_path, map_location="cpu")
    train_matrix = np.stack([row["query_embedding"] for row in train_rows])
    heldout_matrix = np.stack([row["query_embedding"] for row in heldout_rows])
    train_predictions = qesa_predictions(checkpoint, train_matrix, int(qesa_config["hidden_dimension"]), args.device)
    heldout_predictions = qesa_predictions(checkpoint, heldout_matrix, int(qesa_config["hidden_dimension"]), args.device)
    _q, _a, _o, train_targets = QESA_HELPERS["supervision"](train_rows, incidence, index_by_id)
    prior = train_targets.mean(axis=0); prior /= max(float(prior.sum()), 1e-12)
    examples = prepare_training_examples(train_rows, train_predictions, incidence, raw, index_by_id, config, args.seed + args.fold, args.device)
    model, training = train_model(examples, config, args.seed + args.fold * 10000, args.device)
    model.eval()

    metrics, selections = [], []
    max_budget = max(map(int, config["budgets"]))
    for row, aspect in zip(heldout_rows, heldout_predictions, strict=True):
        eligible = np.asarray([index_by_id[memory_id] for memory_id in row["eligible_memory_ids"]], dtype=np.int64)
        cosine = raw[eligible] @ row["query_embedding"]
        association = incidence[eligible] @ aspect
        protected_local = np.argsort(-cosine, kind="stable")[: int(config["protected_ann_seeds"])].tolist()
        protected_ids = [ids[int(eligible[position])] for position in protected_local]
        QHDSInferenceExample.from_mapping({
            "query_embedding": row["query_embedding"], "protected_ann_seed_ids": protected_ids,
            "eligible_memory_ids": row["eligible_memory_ids"],
        })
        incidence_tensor = torch.as_tensor(incidence[eligible], device=args.device)
        aspect_tensor = torch.as_tensor(aspect, device=args.device)
        prior_tensor = torch.as_tensor(prior, device=args.device)
        features = build_features(torch.as_tensor(cosine, device=args.device), torch.as_tensor(association, device=args.device))
        no_uot_features = build_features(torch.as_tensor(cosine, device=args.device), torch.zeros(len(eligible), device=args.device))
        orders = {
            "qhds": greedy_select(model, features, incidence_tensor, aspect_tensor, protected_local, max_budget),
            "qhds_without_latent_aspects": greedy_select(model, features, incidence_tensor, prior_tensor, protected_local, max_budget),
            "qhds_without_uot_soft_incidence": greedy_select(model, no_uot_features, incidence_tensor, aspect_tensor, protected_local, max_budget, structural=False),
        }
        protected_set = set(protected_local)
        orders["flat_cosine"] = protected_local + [int(position) for position in np.argsort(-cosine, kind="stable") if int(position) not in protected_set][: max_budget - len(protected_local)]
        orders["qesa_association"] = protected_local + [int(position) for position in np.argsort(-association, kind="stable") if int(position) not in protected_set][: max_budget - len(protected_local)]
        for method, order in orders.items():
            for budget in map(int, config["budgets"]):
                selected_ids = [ids[int(eligible[position])] for position in order[:budget]]
                values = evidence_metrics(selected_ids, protected_ids, row["groups"])
                metrics.append({
                    "query_id": row["query_id"], "persona_id": row["persona_id"],
                    "query_session_id": row["query_session_id"], "group_count": len(row["groups"]),
                    "seed": args.seed, "fold": args.fold, "method": method, "budget": budget,
                    "candidate_count": len(selected_ids), **values,
                })
                selections.append({
                    "query_id": row["query_id"], "seed": args.seed, "fold": args.fold,
                    "method": method, "budget": budget, "selected_memory_ids": selected_ids,
                })
    metrics_path = unit_dir / "v21_qhds_query_metrics.csv"
    selections_path = unit_dir / "v21_qhds_selections.jsonl"
    model_path = unit_dir / "v21_qhds_model.pt"
    write_csv_new(metrics_path, metrics)
    with selections_path.open("x", encoding="utf-8") as handle:
        for row in selections:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    torch.save({"state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()}, "training": training}, model_path)
    manifest = {
        "schema_version": "v21-qhds-oof-unit-1", "status": "completed", "run_id": args.run_id,
        "seed": args.seed, "fold": args.fold, "outer_queries": len(heldout_rows),
        "metric_rows": len(metrics), "selection_rows": len(selections), "training": training,
        "nested_budget_policy": True, "protected_seed_count": int(config["protected_ann_seeds"]),
        "artifacts": {"metrics": str(metrics_path.relative_to(VERSION_ROOT)), "selections": str(selections_path.relative_to(VERSION_ROOT)), "model": str(model_path.relative_to(VERSION_ROOT))},
        "hashes": {
            "metrics": sha256_file(metrics_path), "selections": sha256_file(selections_path), "model": sha256_file(model_path),
            "config": sha256_file(VERSION_ROOT / "configs" / "v21_qhds.json"),
            "qesa_checkpoint": sha256_file(qesa_checkpoint_path), "incidence_array": array_sha256(incidence),
        },
        "calibration_read": False, "formal_test_read": False,
    }
    write_json_new(unit_dir / "v21_qhds_unit_manifest.json", manifest)
    print(json.dumps({"seed": args.seed, "fold": args.fold, "metrics": len(metrics), "training": training}, sort_keys=True))


if __name__ == "__main__":
    main()
