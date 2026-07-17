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

from v21.io import load_config, sha256_file, write_csv_new, write_json_new  # noqa: E402
from v21.qesa_r2 import LatentAspectQESAR2, qesa_r2_training_loss  # noqa: E402
from v21.semantic_projection import project_matrix  # noqa: E402
from v21.uot_recipes import array_sha256  # noqa: E402


R1_HELPERS = runpy.run_path(str(VERSION_ROOT / "scripts" / "90_train_v21_qesa_oof.py"), run_name="v21_qesa_r1_helpers")
PHASE1_HELPERS = runpy.run_path(str(VERSION_ROOT / "scripts" / "87_train_v21_projection_uot_oof.py"), run_name="v21_phase1_helpers")


def train_model(rows, incidence, index_by_id, *, latent, regime, seed, config, device, shuffled=False):
    queries, affinities, owners, targets = R1_HELPERS["supervision"](rows, incidence, index_by_id)
    if shuffled:
        queries = queries[np.random.default_rng(seed).permutation(len(queries))]
    query_tensor = torch.as_tensor(queries, device=device)
    affinity_tensor = torch.as_tensor(affinities, device=device)
    owner_tensor = torch.as_tensor(owners, device=device)
    target_tensor = torch.as_tensor(targets, device=device)
    torch.manual_seed(seed)
    model = LatentAspectQESAR2(
        queries.shape[1],
        int(config["hidden_dimension"]),
        int(latent),
        incidence.shape[1],
        float(regime["aspect_temperature"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    diagnostics = {}
    for _epoch in range(int(config["epochs"])):
        optimizer.zero_grad(set_to_none=True)
        loss, diagnostics = qesa_r2_training_loss(
            model,
            query_tensor,
            affinity_tensor,
            owner_tensor,
            target_tensor,
            coverage_weight=float(regime["coverage_weight"]),
            target_ce_weight=float(regime["target_ce_weight"]),
            diversity_weight=float(regime["diversity_weight"]),
            map_entropy_weight=float(regime["map_entropy_weight"]),
            aspect_usage_weight=float(regime["aspect_usage_weight"]),
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    with torch.no_grad():
        latent_probability, aspect_map, prototype = model(query_tensor)
        normalized = torch.nn.functional.normalize(aspect_map, dim=1, eps=1e-12)
        cosine = normalized @ normalized.T
        off_diagonal = cosine[~torch.eye(len(cosine), dtype=torch.bool, device=cosine.device)]
        diagnostics.update({
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "aspect_map_pairwise_cosine_mean": float(off_diagonal.mean().item()),
            "aspect_map_pairwise_cosine_max": float(off_diagonal.max().item()),
            "prototype_coordinate_std_mean": float(prototype.std(dim=0).mean().item()),
            "latent_max_probability_mean": float(latent_probability.max(dim=1).values.mean().item()),
        })
    return model, diagnostics


@torch.no_grad()
def predictions(model, rows, device):
    query = torch.as_tensor(np.stack([row["query_embedding"] for row in rows]), device=device)
    return model(query)[2].cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one nested v21 QESA r2 outer OOF unit.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--run-id", default="v21_qesa_oof_r2_20260714")
    args = parser.parse_args()
    config = load_config("v21_qesa_r2.json")
    phase = load_config("v21_phase1_oof.json")
    data = load_config("v21_data.json")
    if args.run_id != config["run_id"]:
        raise ValueError("QESA r2 run ID must match its preregistered config")
    if args.seed not in config["fixed_seeds"] or not 0 <= args.fold < int(config["outer_folds"]):
        raise ValueError("seed/fold outside fixed QESA r2 OOF design")
    unit_dir = VERSION_ROOT / "artifacts" / "v21_qesa" / args.run_id / f"seed_{args.seed}" / f"fold_{args.fold}"
    unit_dir.mkdir(parents=True, exist_ok=False)

    ids = list(map(str, R1_HELPERS["read_json"](data["memory"]["embedding_ids"])))
    index_by_id = {memory_id: position for position, memory_id in enumerate(ids)}
    raw = np.load(R1_HELPERS["resolve"](data["memory"]["embedding_vectors"]), mmap_mode="r").astype(np.float32)
    raw /= np.maximum(np.linalg.norm(raw, axis=1, keepdims=True), 1e-12)
    query_vectors = R1_HELPERS["load_query_vectors"](data)
    rows_by_fold = R1_HELPERS["load_rows"](data, query_vectors, int(config["outer_folds"]))
    outer_train_folds = [fold for fold in range(5) if fold != args.fold]
    inner_validation_fold = (args.fold + int(phase["inner_validation_fold_offset"])) % 5
    inner_train_rows = [row for fold in outer_train_folds if fold != inner_validation_fold for row in rows_by_fold[fold]]
    inner_validation_rows = rows_by_fold[inner_validation_fold]
    outer_train_rows = [row for fold in outer_train_folds for row in rows_by_fold[fold]]
    heldout_rows = rows_by_fold[args.fold]

    phase1_unit = VERSION_ROOT / "artifacts" / "v21_phase1" / config["phase1_run_id"] / f"seed_{args.seed}" / f"fold_{args.fold}"
    phase1_manifest = json.loads((phase1_unit / "v21_unit_manifest.json").read_text(encoding="utf-8"))
    final_incidence = np.load(phase1_unit / "v21_uot_semantic_incidence.npy", allow_pickle=False)
    groups_by_fold, _queries_by_fold = PHASE1_HELPERS["load_records"](data, raw, ids, 5)
    inner_groups = [record for fold in outer_train_folds if fold != inner_validation_fold for record in groups_by_fold[fold]]
    memory_tensor = torch.as_tensor(raw, device=args.device)
    selected_adapter = phase1_manifest["selected_adapter"]
    inner_adapter = PHASE1_HELPERS["train_adapter"](
        memory_tensor,
        inner_groups,
        seed=args.seed + args.fold * 1000 + int(selected_adapter["bottleneck"]) + int(selected_adapter["epochs"]),
        bottleneck=int(selected_adapter["bottleneck"]),
        epochs=int(selected_adapter["epochs"]),
        learning_rate=float(selected_adapter["learning_rate"]),
        config=load_config("v21_uot_semantic.json"),
        device=args.device,
    )
    inner_projected = project_matrix(inner_adapter, memory_tensor).detach().cpu().numpy()
    _prototypes, inner_incidence, inner_diagnostics = PHASE1_HELPERS["fit_recipe"](
        inner_projected,
        phase1_manifest["selected_recipe"],
        args.seed + args.fold,
        load_config("v21_uot_semantic.json"),
        args.device,
    )
    if not inner_diagnostics["transport_converged"] or inner_diagnostics["prototype_collapse_flag"]:
        raise RuntimeError("reconstructed inner FSA is numerically invalid")

    inner_trials = []
    for latent in config["latent_aspect_counts"]:
        for regime in config["regularization_regimes"]:
            trial_seed = args.seed + args.fold * 1000 + int(latent) + sum(map(ord, regime["regime_id"]))
            model, diagnostics = train_model(
                inner_train_rows,
                inner_incidence,
                index_by_id,
                latent=int(latent),
                regime=regime,
                seed=trial_seed,
                config=config,
                device=args.device,
            )
            metric_rows = R1_HELPERS["evaluate_prototype_predictions"](
                inner_validation_rows,
                predictions(model, inner_validation_rows, args.device),
                inner_incidence,
                raw,
                ids,
                index_by_id,
                [int(config["inner_selection_budget"])],
                f"qesa_r2_l{latent}_{regime['regime_id']}",
            )
            selection = R1_HELPERS["aggregate_selection"](
                inner_validation_rows, metric_rows, int(config["inner_selection_budget"])
            )
            inner_trials.append({
                "latent_aspects": int(latent),
                "regime": regime,
                "training": diagnostics,
                "selection": selection,
            })
    selected = min(inner_trials, key=lambda trial: (
        -trial["selection"]["multi_group_all_covered"],
        -trial["selection"]["multi_group_coverage"],
        -trial["selection"]["all_group_coverage"],
        trial["latent_aspects"],
        trial["regime"]["regime_id"],
    ))
    latent = int(selected["latent_aspects"])
    regime = selected["regime"]
    final_qesa, final_diagnostics = train_model(
        outer_train_rows,
        final_incidence,
        index_by_id,
        latent=latent,
        regime=regime,
        seed=args.seed + args.fold * 10000,
        config=config,
        device=args.device,
    )
    shuffled_qesa, shuffled_diagnostics = train_model(
        outer_train_rows,
        final_incidence,
        index_by_id,
        latent=latent,
        regime=regime,
        seed=args.seed + args.fold * 10000 + 1,
        config=config,
        device=args.device,
        shuffled=True,
    )
    budgets = list(map(int, config["budgets"]))
    metrics = []
    heldout_query_matrix = np.stack([row["query_embedding"] for row in heldout_rows])
    train_query_matrix = np.stack([row["query_embedding"] for row in outer_train_rows])
    nearest = (heldout_query_matrix @ train_query_matrix.T).max(axis=1)
    for method, model in (("qesa", final_qesa), ("shuffled_group_qesa", shuffled_qesa)):
        method_rows = R1_HELPERS["evaluate_prototype_predictions"](
            heldout_rows,
            predictions(model, heldout_rows, args.device),
            final_incidence,
            raw,
            ids,
            index_by_id,
            budgets,
            method,
        )
        for row, similarity in zip(method_rows, np.repeat(nearest, len(budgets)), strict=True):
            row.update({
                "nearest_train_query_cosine": float(similarity),
                "seed": args.seed,
                "fold": args.fold,
                "selected_latent_aspects": latent,
                "selected_regime_id": regime["regime_id"],
            })
        metrics.extend(method_rows)

    metrics_path = unit_dir / "v21_qesa_r2_query_metrics.csv"
    models_path = unit_dir / "v21_qesa_r2_models.pt"
    write_csv_new(metrics_path, metrics)
    torch.save({
        "qesa": {key: value.detach().cpu() for key, value in final_qesa.state_dict().items()},
        "shuffled_qesa": {key: value.detach().cpu() for key, value in shuffled_qesa.state_dict().items()},
        "selected_latent_aspects": latent,
        "selected_regime": regime,
    }, models_path)
    r1_metrics = VERSION_ROOT / "aggregate" / "v21_qesa_oof_metrics.csv"
    manifest = {
        "schema_version": "v21-qesa-r2-oof-unit-1",
        "status": "completed",
        "run_id": args.run_id,
        "seed": args.seed,
        "fold": args.fold,
        "inner_validation_fold": inner_validation_fold,
        "inner_fsa_diagnostics": inner_diagnostics,
        "inner_trials": inner_trials,
        "selected_latent_aspects": latent,
        "selected_regime": regime,
        "qesa_training": final_diagnostics,
        "shuffled_training": shuffled_diagnostics,
        "outer_queries": len(heldout_rows),
        "metric_rows": len(metrics),
        "artifacts": {
            "metrics": str(metrics_path.relative_to(VERSION_ROOT)),
            "models": str(models_path.relative_to(VERSION_ROOT)),
        },
        "hashes": {
            "metrics": sha256_file(metrics_path),
            "models": sha256_file(models_path),
            "config": sha256_file(VERSION_ROOT / "configs" / "v21_qesa_r2.json"),
            "final_incidence_array": array_sha256(final_incidence),
            "reused_r1_baseline_metrics": sha256_file(r1_metrics),
        },
        "generalization_audits": {
            "query_id_is_not_a_model_input": True,
            "shuffled_group_supervision_recorded": True,
            "unseen_query_outer_oof": True,
        },
        "calibration_read": False,
        "formal_test_read": False,
    }
    write_json_new(unit_dir / "v21_qesa_r2_unit_manifest.json", manifest)
    print(json.dumps({
        "seed": args.seed,
        "fold": args.fold,
        "selected_latent_aspects": latent,
        "selected_regime": regime["regime_id"],
        "metrics": len(metrics),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
