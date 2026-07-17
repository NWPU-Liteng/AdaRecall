#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import re
import runpy
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


VERSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VERSION_ROOT / "src"))

from v21.evidence_bundle_locality import (  # noqa: E402
    BundleInferenceExample,
    EvidenceAspectCountPredictor,
    PairwiseOnlyBundleScorer,
    PermutationInvariantBundleScorer,
    binary_bundle_loss,
    build_semantic_bundle_candidates,
    member_feature_matrix,
    ranked_bundle_metrics,
)
from v21.io import iter_jsonl, load_config, read_json, resolve, sha256_file, write_csv_new, write_json_new  # noqa: E402
from v21.qesa_r2 import LatentAspectQESAR2  # noqa: E402
from v21.uot_recipes import array_sha256  # noqa: E402


QESA_HELPERS = runpy.run_path(str(VERSION_ROOT / "scripts" / "90_train_v21_qesa_oof.py"), run_name="v21_bundle_qesa_helpers")
ATTEMPT_RE = re.compile(r"^attempt_[0-9]{3,}$")


@torch.no_grad()
def qesa_predictions(checkpoint: dict, queries: np.ndarray, hidden: int, device: str) -> np.ndarray:
    state = checkpoint["qesa"]
    model = LatentAspectQESAR2(
        queries.shape[1], hidden, int(checkpoint["selected_latent_aspects"]),
        int(state["aspect_prototype_logits"].shape[1]), float(checkpoint["selected_regime"]["aspect_temperature"]),
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model(torch.as_tensor(queries, device=device))[2].cpu().numpy()


def train_cardinality_predictor(rows: list[dict], config: dict, seed: int, device: str):
    maximum = max(map(int, config["cardinality_values"]))
    query = torch.as_tensor(np.stack([row["query_embedding"] for row in rows]), device=device)
    target = torch.as_tensor([min(maximum, len(row["groups"])) - 1 for row in rows], device=device)
    torch.manual_seed(seed)
    model = EvidenceAspectCountPredictor(query.shape[1], int(config["cardinality_hidden_dimension"]), maximum).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    for _ in range(int(config["cardinality_epochs"])):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(query), target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    with torch.no_grad():
        accuracy = float((model(query).argmax(dim=1) == target).float().mean().item())
    return model.eval(), {"outer_train_accuracy": accuracy, "parameters": sum(item.numel() for item in model.parameters())}


def bundle_training_arrays(rows, predictions, raw, incidence, index_by_id, config, seed):
    """Build positives from train-only Cartesian group alternatives, then definite negatives."""
    rng = np.random.default_rng(seed)
    grouped: dict[int, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    positive_count = 0
    for row, aspect in zip(rows, predictions, strict=True):
        cardinality = len(row["groups"])
        if cardinality < 2 or cardinality > max(map(int, config["cardinality_values"])):
            continue
        eligible = np.asarray([index_by_id[item] for item in row["eligible_memory_ids"]], dtype=np.int64)
        features = member_feature_matrix(row["query_embedding"], raw[eligible], incidence[eligible], aspect)
        local_by_global = {int(item): position for position, item in enumerate(eligible)}
        all_acceptable = {item for group in row["groups"] for item in group}
        nonmembers = [position for position, item in enumerate(row["eligible_memory_ids"]) if item not in all_acceptable]
        for bundle in itertools.product(*row["groups"]):
            # product has one member per distinct train-only evidence group by construction.
            if len(set(bundle)) != cardinality:
                # Alternatives may overlap across groups. A repeated member cannot
                # represent two aspects, so it is not an acceptable hyperedge.
                continue
            positions = [local_by_global[index_by_id[item]] for item in bundle]
            grouped[cardinality]["query"].append(row["query_embedding"])
            grouped[cardinality]["features"].append(features[positions])
            grouped[cardinality]["label"].append(1.0)
            positive_count += 1
            if not nonmembers:
                continue
            for _ in range(int(config["negative_bundles_per_positive"])):
                replacement = int(rng.choice(nonmembers))
                negative_positions = list(positions)
                negative_positions[int(rng.integers(cardinality))] = replacement
                if len(set(negative_positions)) != cardinality:
                    continue
                grouped[cardinality]["query"].append(row["query_embedding"])
                grouped[cardinality]["features"].append(features[negative_positions])
                grouped[cardinality]["label"].append(0.0)
    arrays = {}
    for cardinality, values in grouped.items():
        arrays[cardinality] = {
            "query": np.asarray(values["query"], dtype=np.float32),
            "features": np.asarray(values["features"], dtype=np.float32),
            "label": np.asarray(values["label"], dtype=np.float32),
        }
    if not arrays or positive_count == 0:
        raise ValueError("no multi-group train-only bundle supervision was constructed")
    return arrays, positive_count


def train_scorer(model, arrays, config, seed, device, *, permute_query: bool = False):
    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    rng = np.random.default_rng(seed)
    tensors = {
        cardinality: tuple(torch.as_tensor(values[field], device=device) for field in ("query", "features", "label"))
        for cardinality, values in arrays.items()
    }
    epoch_losses = []
    for _ in range(int(config["epochs"])):
        losses = []
        for cardinality, (query, features, labels) in tensors.items():
            order = rng.permutation(len(labels))
            if permute_query:
                # Query-to-bundle supervision is deliberately destroyed; query IDs are never model features.
                query = query[torch.as_tensor(rng.permutation(len(query)), device=device)]
            for start in range(0, len(order), int(config["batch_size"])):
                positions = torch.as_tensor(order[start : start + int(config["batch_size"])], device=device)
                optimizer.zero_grad(set_to_none=True)
                loss = binary_bundle_loss(model(query[positions], features[positions]), labels[positions])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.detach().item()))
        epoch_losses.append(float(np.mean(losses)))
    return model.eval(), {"epoch_loss_first": epoch_losses[0], "epoch_loss_last": epoch_losses[-1], "parameters": sum(item.numel() for item in model.parameters())}


@torch.no_grad()
def rank_candidates(method, scorer, query_embedding, candidates, device):
    if method == "independent_member":
        scores = [float(candidate.member_features[:, :2].sum()) for candidate in candidates]
    else:
        query = torch.as_tensor(np.repeat(query_embedding[None, :], len(candidates), axis=0), device=device)
        features = torch.as_tensor(np.stack([candidate.member_features for candidate in candidates]), device=device)
        scores = scorer(query, features).cpu().tolist()
    ranked = sorted(zip(scores, candidates), key=lambda item: (-item[0], item[1].memory_ids))
    return [candidate for _score, candidate in ranked], [float(score) for score, _candidate in ranked]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one v21 evidence-bundle locality outer OOF unit.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--run-id", default="v21_bundle_locality_oof_r1_20260714")
    parser.add_argument("--attempt", default="attempt_001")
    args = parser.parse_args()
    config, data, qesa_config = load_config("v21_bundle_locality.json"), load_config("v21_data.json"), load_config("v21_qesa_r2.json")
    if args.run_id != config["run_id"] or args.seed not in config["fixed_seeds"] or not 0 <= args.fold < int(config["outer_folds"]):
        raise ValueError("run ID, seed, or fold outside bundle-locality preregistration")
    if not ATTEMPT_RE.fullmatch(args.attempt):
        raise ValueError("attempt must have the form attempt_NNN")
    unit_dir = VERSION_ROOT / "artifacts" / "v21_bundle_locality" / args.run_id / f"seed_{args.seed}" / f"fold_{args.fold}" / "attempts" / args.attempt
    unit_dir.mkdir(parents=True, exist_ok=False)

    ids = list(map(str, read_json(data["memory"]["embedding_ids"])))
    index_by_id = {item: position for position, item in enumerate(ids)}
    raw = np.load(resolve(data["memory"]["embedding_vectors"]), mmap_mode="r").astype(np.float32)
    raw /= np.maximum(np.linalg.norm(raw, axis=1, keepdims=True), 1e-12)
    query_vectors = QESA_HELPERS["load_query_vectors"](data)
    rows_by_fold = QESA_HELPERS["load_rows"](data, query_vectors, int(config["outer_folds"]))
    train_rows = [row for fold in range(5) if fold != args.fold for row in rows_by_fold[fold]]
    heldout_rows = rows_by_fold[args.fold]
    qesa_path = VERSION_ROOT / "artifacts" / "v21_qesa" / config["qesa_run_id"] / f"seed_{args.seed}" / f"fold_{args.fold}" / "v21_qesa_r2_models.pt"
    checkpoint = torch.load(qesa_path, map_location="cpu")
    incidence_path = VERSION_ROOT / "artifacts" / "v21_phase1" / config["phase1_run_id"] / f"seed_{args.seed}" / f"fold_{args.fold}" / "v21_uot_semantic_incidence.npy"
    incidence = np.load(incidence_path, allow_pickle=False)
    train_predictions = qesa_predictions(checkpoint, np.stack([row["query_embedding"] for row in train_rows]), int(qesa_config["hidden_dimension"]), args.device)
    heldout_predictions = qesa_predictions(checkpoint, np.stack([row["query_embedding"] for row in heldout_rows]), int(qesa_config["hidden_dimension"]), args.device)

    # Contract audit: source bundles must exactly be the outer-train multi-group Cartesian bundles.
    contract = read_json(config["source_bundle_contract"])
    source_bundles = list(iter_jsonl(config["source_bundle_artifact"]))
    heldout_ids = {str(row["query_id"]) for row in heldout_rows}
    expected_bundles = {
        (str(row["query_id"]), tuple(map(str, bundle)))
        for row in train_rows if len(row["groups"]) >= 2
        for bundle in itertools.product(*row["groups"])
        if len(set(bundle)) == len(row["groups"])
    }
    source_outer_train = [item for item in source_bundles if str(item["query_id"]) not in heldout_ids]
    invalid_source_bundles = [item for item in source_outer_train if len(set(map(str, item["memory_ids"]))) != int(item["group_count"])]
    observed_bundles = {
        (str(item["query_id"]), tuple(map(str, item["memory_ids"])))
        for item in source_outer_train if item not in invalid_source_bundles
    }
    if expected_bundles != observed_bundles:
        raise ValueError(
            f"outer-train valid bundle contract mismatch: expected={len(expected_bundles)} observed={len(observed_bundles)}"
        )

    cardinality_model, cardinality_training = train_cardinality_predictor(train_rows, config, args.seed + args.fold, args.device)
    arrays, positive_count = bundle_training_arrays(train_rows, train_predictions, raw, incidence, index_by_id, config, args.seed + args.fold * 100)
    dimension = train_rows[0]["query_embedding"].shape[0]
    deepsets, deepsets_training = train_scorer(
        PermutationInvariantBundleScorer(dimension, 3, int(config["scorer_hidden_dimension"])).to(args.device), arrays, config,
        args.seed + args.fold * 1000, args.device,
    )
    shuffled, shuffled_training = train_scorer(
        PermutationInvariantBundleScorer(dimension, 3, int(config["scorer_hidden_dimension"])).to(args.device), arrays, config,
        args.seed + args.fold * 1000 + 1, args.device, permute_query=True,
    )
    pairwise, pairwise_training = train_scorer(
        PairwiseOnlyBundleScorer(dimension, 3, int(config["scorer_hidden_dimension"])).to(args.device), arrays, config,
        args.seed + args.fold * 1000 + 2, args.device,
    )

    train_matrix = np.stack([row["query_embedding"] for row in train_rows])
    train_matrix /= np.maximum(np.linalg.norm(train_matrix, axis=1, keepdims=True), 1e-12)
    metrics, selections = [], []
    model_by_method = {
        "bundle_deepsets": deepsets, "independent_member": None,
        "shuffled_bundle_deepsets": shuffled, "pairwise_only": pairwise,
    }
    for row, aspect in zip(heldout_rows, heldout_predictions, strict=True):
        actual_count = len(row["groups"])
        if actual_count < 2:
            continue
        with torch.no_grad():
            predicted_count = int(cardinality_model(torch.as_tensor(row["query_embedding"][None, :], device=args.device)).argmax(dim=1).item()) + 1
        predicted_count = min(predicted_count, int(config["activated_semantic_aspects"]))
        eligible = np.asarray([index_by_id[item] for item in row["eligible_memory_ids"]], dtype=np.int64)
        member_features = member_feature_matrix(row["query_embedding"], raw[eligible], incidence[eligible], aspect)
        candidates, construction = build_semantic_bundle_candidates(
            eligible_memory_ids=row["eligible_memory_ids"], eligible_member_features=member_features,
            eligible_incidence=incidence[eligible], aspect_prediction=aspect, predicted_aspect_count=predicted_count,
            activated_aspects=int(config["activated_semantic_aspects"]), members_per_aspect=int(config["members_per_semantic_aspect"]),
            beam_width=int(config["candidate_bundle_beam"]),
        )
        BundleInferenceExample.from_mapping({
            "query_embedding": row["query_embedding"], "eligible_memory_ids": row["eligible_memory_ids"],
            "predicted_aspect_count": predicted_count, "candidates": candidates,
        })
        normalized = row["query_embedding"] / max(float(np.linalg.norm(row["query_embedding"])), 1e-12)
        nearest = float((train_matrix @ normalized).max())
        for method, model in model_by_method.items():
            ranked, scores = rank_candidates(method, model, row["query_embedding"], candidates, args.device)
            values = ranked_bundle_metrics(ranked, row["groups"], int(config["evaluation_beam"]))
            metrics.append({
                "query_id": row["query_id"], "persona_id": row["persona_id"], "query_session_id": row["query_session_id"],
                "seed": args.seed, "fold": args.fold, "method": method, "actual_group_count": actual_count,
                "predicted_aspect_count": predicted_count, "nearest_train_query_cosine": nearest,
                "candidate_bundle_count": len(candidates), **construction, **values,
            })
            selections.append({
                "query_id": row["query_id"], "seed": args.seed, "fold": args.fold, "method": method,
                "predicted_aspect_count": predicted_count,
                "ranked_bundles": [{"memory_ids": list(candidate.memory_ids), "score": score} for candidate, score in zip(ranked, scores, strict=True)],
            })

    metrics_path, selections_path, models_path = unit_dir / "v21_bundle_locality_query_metrics.csv", unit_dir / "v21_bundle_locality_selections.jsonl", unit_dir / "v21_bundle_locality_models.pt"
    write_csv_new(metrics_path, metrics)
    with selections_path.open("x", encoding="utf-8") as handle:
        for item in selections:
            handle.write(json.dumps(item, sort_keys=True) + "\n")
    torch.save({
        "cardinality": {key: value.detach().cpu() for key, value in cardinality_model.state_dict().items()},
        "deepsets": {key: value.detach().cpu() for key, value in deepsets.state_dict().items()},
        "shuffled_bundle_deepsets": {key: value.detach().cpu() for key, value in shuffled.state_dict().items()},
        "pairwise_only": {key: value.detach().cpu() for key, value in pairwise.state_dict().items()},
    }, models_path)
    manifest = {
        "schema_version": "v21-evidence-bundle-locality-oof-unit-1", "status": "completed", "revision_id": config["revision_id"],
        "run_id": args.run_id, "attempt": args.attempt, "seed": args.seed, "fold": args.fold,
        "outer_train_queries": len(train_rows), "outer_heldout_queries": len(heldout_rows), "outer_heldout_multi_group_queries": sum(len(row["groups"]) >= 2 for row in heldout_rows),
        "outer_train_source_bundle_rows": len(source_outer_train), "outer_train_valid_acceptable_bundles": len(observed_bundles),
        "outer_train_invalid_duplicate_member_bundles_filtered": len(invalid_source_bundles), "constructed_positive_bundles": positive_count,
        "source_contract_bundles": contract["acceptable_bundle_hyperedges"],
        "training": {"cardinality": cardinality_training, "deepsets": deepsets_training, "shuffled_bundle": shuffled_training, "pairwise_only": pairwise_training},
        "generalization_audits": {
            "outer_train_only_supervision": True, "heldout_bundle_ids_not_in_training": True,
            "query_id_is_not_a_model_input": True, "query_id_permutation_audit_recorded": True,
            "nearest_train_query_similarity_recorded": True, "inference_rejects_evidence_groups_and_acceptable_members": True,
        },
        "artifacts": {"metrics": str(metrics_path.relative_to(VERSION_ROOT)), "selections": str(selections_path.relative_to(VERSION_ROOT)), "models": str(models_path.relative_to(VERSION_ROOT))},
        "hashes": {"metrics": sha256_file(metrics_path), "selections": sha256_file(selections_path), "models": sha256_file(models_path), "config": sha256_file(VERSION_ROOT / "configs" / "v21_bundle_locality.json"), "qesa_checkpoint": sha256_file(qesa_path), "incidence_array": array_sha256(incidence)},
        "calibration_read": False, "formal_test_read": False,
    }
    write_json_new(unit_dir / "v21_bundle_locality_unit_manifest.json", manifest)
    print(json.dumps({"seed": args.seed, "fold": args.fold, "attempt": args.attempt, "metrics": len(metrics), "positive_bundles": positive_count}, sort_keys=True))


if __name__ == "__main__":
    main()
