#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import runpy
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


VERSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VERSION_ROOT / "src"))

from v21.evidence_sets import fold_for  # noqa: E402
from v21.io import iter_jsonl, load_config, read_json, resolve, sha256_file, write_csv_new, write_json_new  # noqa: E402
from v21.qesa import (  # noqa: E402
    FlatMLPRanker,
    LatentAspectQESA,
    LinearPrototypeProbe,
    QESAInferenceExample,
    evidence_metrics,
    qesa_training_loss,
    select_protected_frontier,
)
from v21.semantic_projection import ResidualProjectionAdapter, project_matrix  # noqa: E402
from v21.uot_recipes import array_sha256  # noqa: E402
from v21.uot_semantic import sparse_memberships  # noqa: E402


def load_query_vectors(data: dict) -> dict[str, np.ndarray]:
    matrix = np.load(resolve(data["query_embeddings"]["vectors"]), mmap_mode="r")
    records = read_json(data["query_embeddings"]["index"])["records"]
    return {
        str(row["query_id"]): np.array(matrix[position], dtype=np.float32, copy=True)
        for position, row in enumerate(records)
        if row.get("dataset") == data["query_embeddings"]["train_key"]
    }


def load_rows(data: dict, query_vectors: dict[str, np.ndarray], folds: int) -> dict[int, list[dict]]:
    by_fold: dict[int, list[dict]] = defaultdict(list)
    for row in iter_jsonl(data["split"]["train_query_plan"]):
        fold = fold_for(str(row["persona_id"]), int(row["query_session_id"]), folds)
        groups = [
            list(dict.fromkeys(map(str, group.get("acceptable_memory_ids", []))))
            for group in row.get("evidence_groups", [])
        ]
        by_fold[fold].append({
            "query_id": str(row["query_id"]),
            "persona_id": str(row["persona_id"]),
            "query_session_id": int(row["query_session_id"]),
            "query_embedding": query_vectors[str(row["query_id"])],
            "eligible_memory_ids": list(map(str, row["eligible_memory_ids"])),
            "groups": groups,
        })
    return by_fold


def supervision(rows: list[dict], incidence: np.ndarray, index_by_id: dict[str, int]):
    queries = np.stack([row["query_embedding"] for row in rows]).astype(np.float32)
    affinities, owners, targets = [], [], []
    for query_index, row in enumerate(rows):
        query_affinities = []
        for group in row["groups"]:
            member = incidence[[index_by_id[memory_id] for memory_id in group]]
            affinity = member.max(axis=0)
            affinity = affinity / max(float(affinity.max()), 1e-12)
            affinities.append(affinity)
            query_affinities.append(affinity)
            owners.append(query_index)
        target = np.mean(query_affinities, axis=0)
        targets.append(target / max(float(target.sum()), 1e-12))
    return queries, np.asarray(affinities, dtype=np.float32), np.asarray(owners, dtype=np.int64), np.asarray(targets, dtype=np.float32)


def train_qesa(rows, incidence, index_by_id, *, latent, seed, config, device, shuffled=False):
    queries, affinities, owners, _targets = supervision(rows, incidence, index_by_id)
    if shuffled:
        permutation = np.random.default_rng(seed).permutation(len(queries))
        queries = queries[permutation]
    query_tensor = torch.as_tensor(queries, device=device)
    affinity_tensor = torch.as_tensor(affinities, device=device)
    owner_tensor = torch.as_tensor(owners, device=device)
    torch.manual_seed(seed)
    model = LatentAspectQESA(
        queries.shape[1], int(config["hidden_dimension"]), latent, incidence.shape[1]
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    last = {}
    for _epoch in range(int(config["epochs"])):
        optimizer.zero_grad(set_to_none=True)
        loss, last = qesa_training_loss(
            model, query_tensor, affinity_tensor, owner_tensor,
            diversity_weight=float(config["aspect_diversity_weight"]),
            entropy_weight=float(config["prediction_entropy_weight"]),
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return model, {**last, "parameters": sum(parameter.numel() for parameter in model.parameters())}


def train_linear(rows, incidence, index_by_id, *, seed, config, device):
    queries, affinities, owners, _targets = supervision(rows, incidence, index_by_id)
    query_tensor = torch.as_tensor(queries, device=device)
    affinity_tensor = torch.as_tensor(affinities, device=device)
    owner_tensor = torch.as_tensor(owners, device=device)
    torch.manual_seed(seed)
    model = LinearPrototypeProbe(queries.shape[1], incidence.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    for _epoch in range(int(config["epochs"])):
        optimizer.zero_grad(set_to_none=True)
        probabilities = model(query_tensor)
        selected = probabilities[owner_tensor]
        coverage = (selected * affinity_tensor).sum(1).clamp(1e-8, 1.0)
        loss = -coverage.log().mean()
        loss.backward()
        optimizer.step()
    return model


def train_flat_mlp(rows, projected_memory, index_by_id, *, seed, config, device):
    queries = np.stack([row["query_embedding"] for row in rows]).astype(np.float32)
    targets = []
    for row in rows:
        positions = [index_by_id[memory_id] for group in row["groups"] for memory_id in group]
        target = projected_memory[positions].mean(axis=0)
        targets.append(target / max(float(np.linalg.norm(target)), 1e-12))
    queries_t = torch.as_tensor(queries, device=device)
    targets_t = torch.as_tensor(np.asarray(targets, dtype=np.float32), device=device)
    torch.manual_seed(seed)
    model = FlatMLPRanker(queries.shape[1], int(config["hidden_dimension"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    rng = np.random.default_rng(seed)
    for _epoch in range(int(config["epochs"])):
        order = rng.permutation(len(rows))
        for start in range(0, len(rows), int(config["batch_size"])):
            positions = torch.as_tensor(order[start : start + int(config["batch_size"])], device=device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(queries_t[positions])
            target = targets_t[positions]
            logits = prediction @ target.T / float(config["temperature"])
            loss = F.cross_entropy(logits, torch.arange(len(positions), device=device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return model


@torch.no_grad()
def prototype_predictions(model, rows, device):
    query = torch.as_tensor(np.stack([row["query_embedding"] for row in rows]), device=device)
    if isinstance(model, LatentAspectQESA):
        return model(query)[2].cpu().numpy()
    return model(query).cpu().numpy()


def evaluate_prototype_predictions(rows, predictions, incidence, raw_memory, ids, index_by_id, budgets, method):
    output = []
    for row, prediction in zip(rows, predictions, strict=True):
        eligible = [index_by_id[memory_id] for memory_id in row["eligible_memory_ids"]]
        query = row["query_embedding"]
        ann_scores = raw_memory[eligible] @ query
        seed_order = np.argsort(-ann_scores, kind="stable")[:5]
        protected = [eligible[position] for position in seed_order]
        inference = QESAInferenceExample.from_mapping({
            "query_embedding": query,
            "protected_ann_seed_ids": [ids[position] for position in protected],
            "eligible_memory_ids": row["eligible_memory_ids"],
        })
        assert len(inference.protected_ann_seed_ids) == 5
        scores = incidence @ prediction
        for budget in budgets:
            selected = select_protected_frontier(scores, eligible, protected, int(budget))
            metrics = evidence_metrics([ids[position] for position in selected], row["groups"])
            output.append({
                "query_id": row["query_id"], "persona_id": row["persona_id"],
                "query_session_id": row["query_session_id"], "group_count": len(row["groups"]),
                "method": method, "budget": int(budget), "candidate_count": len(selected), **metrics,
            })
    return output


def evaluate_vector_predictions(rows, predictions, memory, raw_memory, ids, index_by_id, budgets, method):
    output = []
    for row, prediction in zip(rows, predictions, strict=True):
        eligible = [index_by_id[memory_id] for memory_id in row["eligible_memory_ids"]]
        ann_scores = raw_memory[eligible] @ row["query_embedding"]
        protected = [eligible[position] for position in np.argsort(-ann_scores, kind="stable")[:5]]
        scores = memory @ prediction
        for budget in budgets:
            selected = select_protected_frontier(scores, eligible, protected, int(budget))
            metrics = evidence_metrics([ids[position] for position in selected], row["groups"])
            output.append({
                "query_id": row["query_id"], "persona_id": row["persona_id"],
                "query_session_id": row["query_session_id"], "group_count": len(row["groups"]),
                "method": method, "budget": int(budget), "candidate_count": len(selected), **metrics,
            })
    return output


def aggregate_selection(rows, metrics, budget):
    selected = [row for row in metrics if int(row["budget"]) == budget]
    multi = [row for row in selected if int(row["group_count"]) >= 2]
    return {
        "all_group_coverage": float(np.mean([row["group_coverage"] for row in selected])),
        "multi_group_coverage": float(np.mean([row["group_coverage"] for row in multi])),
        "multi_group_all_covered": float(np.mean([row["all_groups_covered"] for row in multi])),
        "queries": len(rows), "multi_group_queries": len(multi),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one nested v21 QESA outer OOF unit.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--run-id", default="v21_qesa_oof_r1_20260714")
    args = parser.parse_args()
    config = load_config("v21_qesa.json")
    phase = load_config("v21_phase1_oof.json")
    data = load_config("v21_data.json")
    if args.seed not in config["fixed_seeds"] or not 0 <= args.fold < int(config["outer_folds"]):
        raise ValueError("seed/fold outside fixed QESA OOF design")
    unit_dir = VERSION_ROOT / "artifacts" / "v21_qesa" / args.run_id / f"seed_{args.seed}" / f"fold_{args.fold}"
    unit_dir.mkdir(parents=True, exist_ok=False)

    ids = list(map(str, read_json(data["memory"]["embedding_ids"])))
    index_by_id = {memory_id: position for position, memory_id in enumerate(ids)}
    raw = np.load(resolve(data["memory"]["embedding_vectors"]), mmap_mode="r").astype(np.float32)
    raw /= np.maximum(np.linalg.norm(raw, axis=1, keepdims=True), 1e-12)
    query_vectors = load_query_vectors(data)
    rows_by_fold = load_rows(data, query_vectors, int(config["outer_folds"]))
    outer_train_folds = [fold for fold in range(5) if fold != args.fold]
    inner_validation_fold = (args.fold + int(phase["inner_validation_fold_offset"])) % 5
    inner_train_rows = [row for fold in outer_train_folds if fold != inner_validation_fold for row in rows_by_fold[fold]]
    inner_validation_rows = rows_by_fold[inner_validation_fold]
    outer_train_rows = [row for fold in outer_train_folds for row in rows_by_fold[fold]]
    heldout_rows = rows_by_fold[args.fold]

    phase1_unit = VERSION_ROOT / "artifacts" / "v21_phase1" / config["phase1_run_id"] / f"seed_{args.seed}" / f"fold_{args.fold}"
    phase1_manifest = json.loads((phase1_unit / "v21_unit_manifest.json").read_text(encoding="utf-8"))
    final_incidence = np.load(phase1_unit / "v21_uot_semantic_incidence.npy", allow_pickle=False)
    phase_helpers = runpy.run_path(str(VERSION_ROOT / "scripts" / "87_train_v21_projection_uot_oof.py"), run_name="v21_phase1_helpers")
    groups_by_fold, _queries_by_fold = phase_helpers["load_records"](data, raw, ids, 5)
    inner_groups = [record for fold in outer_train_folds if fold != inner_validation_fold for record in groups_by_fold[fold]]
    memory_tensor = torch.as_tensor(raw, device=args.device)
    selected_adapter = phase1_manifest["selected_adapter"]
    inner_adapter = phase_helpers["train_adapter"](
        memory_tensor, inner_groups,
        seed=args.seed + args.fold * 1000 + int(selected_adapter["bottleneck"]) + int(selected_adapter["epochs"]),
        bottleneck=int(selected_adapter["bottleneck"]), epochs=int(selected_adapter["epochs"]),
        learning_rate=float(selected_adapter["learning_rate"]), config=load_config("v21_uot_semantic.json"), device=args.device,
    )
    inner_projected = project_matrix(inner_adapter, memory_tensor).detach().cpu().numpy()
    _inner_prototypes, inner_incidence, inner_diagnostics = phase_helpers["fit_recipe"](
        inner_projected, phase1_manifest["selected_recipe"], args.seed + args.fold,
        load_config("v21_uot_semantic.json"), args.device,
    )
    if not inner_diagnostics["transport_converged"] or inner_diagnostics["prototype_collapse_flag"]:
        raise RuntimeError("reconstructed inner FSA is numerically invalid")

    inner_trials = []
    for latent in config["latent_aspect_counts"]:
        model, diagnostics = train_qesa(
            inner_train_rows, inner_incidence, index_by_id, latent=int(latent),
            seed=args.seed + args.fold * 100 + int(latent), config=config, device=args.device,
        )
        predictions = prototype_predictions(model, inner_validation_rows, args.device)
        metrics = evaluate_prototype_predictions(
            inner_validation_rows, predictions, inner_incidence, raw, ids, index_by_id,
            [int(config["inner_selection_budget"])], f"qesa_l{latent}",
        )
        summary = aggregate_selection(inner_validation_rows, metrics, int(config["inner_selection_budget"]))
        inner_trials.append({"latent_aspects": int(latent), "training": diagnostics, "selection": summary})
    selected_trial = min(inner_trials, key=lambda trial: (
        -trial["selection"]["multi_group_all_covered"],
        -trial["selection"]["multi_group_coverage"],
        -trial["selection"]["all_group_coverage"],
        trial["latent_aspects"],
    ))
    latent = int(selected_trial["latent_aspects"])

    final_qesa, qesa_diagnostics = train_qesa(
        outer_train_rows, final_incidence, index_by_id, latent=latent,
        seed=args.seed + args.fold * 10000, config=config, device=args.device,
    )
    shuffled_qesa, shuffled_diagnostics = train_qesa(
        outer_train_rows, final_incidence, index_by_id, latent=latent,
        seed=args.seed + args.fold * 10000 + 1, config=config, device=args.device, shuffled=True,
    )
    linear = train_linear(outer_train_rows, final_incidence, index_by_id, seed=args.seed + args.fold, config=config, device=args.device)
    checkpoint = torch.load(phase1_unit / "v21_projection.pt", map_location="cpu")
    final_adapter = ResidualProjectionAdapter(dimension=raw.shape[1], bottleneck=int(checkpoint["selected_adapter"]["bottleneck"])).to(args.device)
    final_adapter.load_state_dict(checkpoint["state_dict"])
    projected_memory = project_matrix(final_adapter, memory_tensor).detach().cpu().numpy()
    flat_mlp = train_flat_mlp(
        outer_train_rows, projected_memory, index_by_id, seed=args.seed + args.fold,
        config=config, device=args.device,
    )

    _q, _a, _o, train_targets = supervision(outer_train_rows, final_incidence, index_by_id)
    prior = train_targets.mean(axis=0)
    prior /= max(float(prior.sum()), 1e-12)
    heldout_query_matrix = np.stack([row["query_embedding"] for row in heldout_rows])
    train_query_matrix = np.stack([row["query_embedding"] for row in outer_train_rows])
    nearest_scores = heldout_query_matrix @ train_query_matrix.T
    nearest_positions = nearest_scores.argmax(axis=1)
    memorized = train_targets[nearest_positions]

    qesa_predictions = prototype_predictions(final_qesa, heldout_rows, args.device)
    shuffled_predictions = prototype_predictions(shuffled_qesa, heldout_rows, args.device)
    linear_predictions = prototype_predictions(linear, heldout_rows, args.device)
    prior_predictions = np.repeat(prior[None, :], len(heldout_rows), axis=0)
    with torch.no_grad():
        flat_predictions = flat_mlp(torch.as_tensor(heldout_query_matrix, device=args.device)).cpu().numpy()
    budgets = list(map(int, config["budgets"]))
    metrics = []
    for method, predictions in (
        ("qesa", qesa_predictions),
        ("query_independent_aspect_prior", prior_predictions),
        ("shuffled_group_qesa", shuffled_predictions),
        ("linear_probe", linear_predictions),
        ("nearest_train_query_memorization", memorized),
    ):
        rows = evaluate_prototype_predictions(heldout_rows, predictions, final_incidence, raw, ids, index_by_id, budgets, method)
        for row, nearest in zip(rows, np.repeat(nearest_scores.max(axis=1), len(budgets)), strict=True):
            row["nearest_train_query_cosine"] = float(nearest)
        metrics.extend(rows)
    metrics.extend(evaluate_vector_predictions(
        heldout_rows, heldout_query_matrix, raw, raw, ids, index_by_id, budgets, "flat_cosine"
    ))
    metrics.extend(evaluate_vector_predictions(
        heldout_rows, flat_predictions, projected_memory, raw, ids, index_by_id, budgets, "same_capacity_mlp_ranker"
    ))
    for row in metrics:
        row.update({"seed": args.seed, "fold": args.fold, "selected_latent_aspects": latent})

    metrics_path = unit_dir / "v21_qesa_query_metrics.csv"
    checkpoint_path = unit_dir / "v21_qesa_models.pt"
    write_csv_new(metrics_path, metrics)
    torch.save({
        "qesa": {key: value.detach().cpu() for key, value in final_qesa.state_dict().items()},
        "shuffled_qesa": {key: value.detach().cpu() for key, value in shuffled_qesa.state_dict().items()},
        "linear_probe": {key: value.detach().cpu() for key, value in linear.state_dict().items()},
        "flat_mlp": {key: value.detach().cpu() for key, value in flat_mlp.state_dict().items()},
        "selected_latent_aspects": latent,
    }, checkpoint_path)
    manifest = {
        "schema_version": "v21-qesa-oof-unit-1", "status": "completed", "run_id": args.run_id,
        "seed": args.seed, "fold": args.fold, "inner_validation_fold": inner_validation_fold,
        "inner_fsa_diagnostics": inner_diagnostics, "inner_trials": inner_trials,
        "selected_latent_aspects": latent, "qesa_training": qesa_diagnostics,
        "shuffled_training": shuffled_diagnostics,
        "parameter_counts": {
            "qesa": sum(parameter.numel() for parameter in final_qesa.parameters()),
            "same_capacity_mlp_ranker": sum(parameter.numel() for parameter in flat_mlp.parameters()),
            "linear_probe": sum(parameter.numel() for parameter in linear.parameters()),
        },
        "outer_queries": len(heldout_rows), "metric_rows": len(metrics),
        "generalization_audits": {
            "query_id_is_not_a_model_input": True,
            "query_id_permutation_invariant_by_interface": True,
            "heldout_session_nearest_query_similarity_recorded": True,
            "memorization_baseline_recorded": True,
            "shuffled_group_supervision_recorded": True,
            "frozen_embedding_linear_probe_recorded": True,
            "unseen_query_outer_oof": True,
        },
        "artifacts": {
            "metrics": str(metrics_path.relative_to(VERSION_ROOT)),
            "models": str(checkpoint_path.relative_to(VERSION_ROOT)),
        },
        "hashes": {
            "metrics": sha256_file(metrics_path), "models": sha256_file(checkpoint_path),
            "config": sha256_file(VERSION_ROOT / "configs" / "v21_qesa.json"),
            "final_incidence_array": array_sha256(final_incidence),
        },
        "calibration_read": False, "formal_test_read": False,
    }
    write_json_new(unit_dir / "v21_qesa_unit_manifest.json", manifest)
    print(json.dumps({"seed": args.seed, "fold": args.fold, "selected_latent_aspects": latent, "metrics": len(metrics)}, sort_keys=True))


if __name__ == "__main__":
    main()
