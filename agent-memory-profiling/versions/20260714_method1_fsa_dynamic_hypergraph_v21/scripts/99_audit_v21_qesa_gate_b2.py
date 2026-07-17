#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import runpy
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


VERSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VERSION_ROOT / "src"))

from v21.io import load_config, read_json, sha256_file, write_csv_new, write_json_new  # noqa: E402
from v21.qesa import LinearPrototypeProbe  # noqa: E402
from v21.qesa_r2 import LatentAspectQESAR2  # noqa: E402


R1 = runpy.run_path(str(VERSION_ROOT / "scripts" / "90_train_v21_qesa_oof.py"), run_name="v21_qesa_r1_helpers")
STATS = runpy.run_path(str(VERSION_ROOT / "scripts" / "92_aggregate_v21_qesa_gate.py"), run_name="v21_qesa_stats_helpers")


@torch.no_grad()
def predict_r2(state: dict, regime: dict, latent_count: int, queries: np.ndarray, hidden: int) -> np.ndarray:
    prototype_count = int(state["aspect_prototype_logits"].shape[1])
    model = LatentAspectQESAR2(
        queries.shape[1], hidden, latent_count, prototype_count, float(regime["aspect_temperature"])
    )
    model.load_state_dict(state)
    model.eval()
    return model(torch.as_tensor(queries))[2].numpy()


@torch.no_grad()
def predict_linear(state: dict, queries: np.ndarray, prototypes: int) -> np.ndarray:
    model = LinearPrototypeProbe(queries.shape[1], prototypes)
    model.load_state_dict(state)
    model.eval()
    return model(torch.as_tensor(queries)).numpy()


def query_metrics(targets: np.ndarray, affinities: np.ndarray, owners: np.ndarray, predictions: np.ndarray, recall_k: int) -> list[dict]:
    output = []
    for query_index, target in enumerate(targets):
        query_affinities = affinities[owners == query_index]
        prediction = predictions[query_index]
        target_top = set(np.argsort(-target, kind="stable")[:recall_k].tolist())
        predicted_top = set(np.argsort(-prediction, kind="stable")[:recall_k].tolist())
        output.append({
            "group_affinity_score": float((query_affinities @ prediction).mean()),
            "target_cross_entropy": float(-(target * np.log(np.maximum(prediction, 1e-12))).sum()),
            "prototype_recall_at_k": float(len(target_top.intersection(predicted_top)) / recall_k),
        })
    return output


def paired_effect(rows: list[dict], comparator: str, metric: str, subset: str) -> list[dict]:
    selected = [row for row in rows if row["method"] in {"qesa", comparator} and (subset == "all" or int(row["group_count"]) >= 2)]
    indexed = {(int(row["seed"]), row["query_id"], row["method"]): row for row in selected}
    keys = sorted({(int(row["seed"]), row["query_id"]) for row in selected})
    output = []
    for seed, query_id in keys:
        qesa = indexed[(seed, query_id, "qesa")]
        baseline = indexed[(seed, query_id, comparator)]
        if metric == "target_cross_entropy":
            delta = float(baseline[metric]) - float(qesa[metric])
        else:
            delta = float(qesa[metric]) - float(baseline[metric])
        output.append({
            "seed": seed,
            "query_id": query_id,
            "persona_id": qesa["persona_id"],
            "query_session_id": qesa["query_session_id"],
            "delta": delta,
        })
    return output


def summarize_effect(rows: list[dict], seed: int, config: dict) -> dict:
    values = np.asarray([row["delta"] for row in rows], dtype=np.float64)
    return {
        "paired_rows": len(rows),
        "effect_positive_favors_qesa": float(values.mean()),
        "paired_query_ci95": STATS["query_bootstrap"](rows, seed, int(config["bootstrap_samples"])),
        "session_clustered_ci95": STATS["cluster_bootstrap"](rows, seed + 10000, int(config["bootstrap_samples"])),
        "cluster_sign_flip_p_two_sided": STATS["sign_flip_p"](rows, seed + 20000, int(config["cluster_sign_flip_samples"])),
        "wins": int((values > 0).sum()),
        "losses": int((values < 0).sum()),
        "ties": int((values == 0).sum()),
    }


def main() -> None:
    gate = load_config("v21_qesa_gate_b2.json")
    r2_config = load_config("v21_qesa_r2.json")
    data = load_config("v21_data.json")
    ids = list(map(str, read_json(data["memory"]["embedding_ids"])))
    index_by_id = {memory_id: position for position, memory_id in enumerate(ids)}
    query_vectors = R1["load_query_vectors"](data)
    rows_by_fold = R1["load_rows"](data, query_vectors, int(r2_config["outer_folds"]))
    run_dir = VERSION_ROOT / "artifacts" / "v21_qesa" / gate["source_run_id"]
    r1_run_dir = VERSION_ROOT / "artifacts" / "v21_qesa" / r2_config["parent_negative_run_id"]
    metric_rows = []
    unit_manifests = []
    recall_k = int(gate["prototype_recall_k"])
    methods = ["qesa", "query_independent_aspect_prior", "shuffled_group_qesa", "frozen_embedding_linear_probe"]
    for seed in r2_config["fixed_seeds"]:
        for fold in range(int(r2_config["outer_folds"])):
            unit_dir = run_dir / f"seed_{seed}" / f"fold_{fold}"
            manifest = json.loads((unit_dir / "v21_qesa_r2_unit_manifest.json").read_text(encoding="utf-8"))
            checkpoint = torch.load(unit_dir / "v21_qesa_r2_models.pt", map_location="cpu")
            r1_checkpoint = torch.load(r1_run_dir / f"seed_{seed}" / f"fold_{fold}" / "v21_qesa_models.pt", map_location="cpu")
            phase1_unit = VERSION_ROOT / "artifacts" / "v21_phase1" / r2_config["phase1_run_id"] / f"seed_{seed}" / f"fold_{fold}"
            incidence = np.load(phase1_unit / "v21_uot_semantic_incidence.npy", allow_pickle=False)
            outer_train = [row for other_fold in range(5) if other_fold != fold for row in rows_by_fold[other_fold]]
            heldout = rows_by_fold[fold]
            _train_q, _train_a, _train_o, train_targets = R1["supervision"](outer_train, incidence, index_by_id)
            heldout_q, heldout_affinities, heldout_owners, heldout_targets = R1["supervision"](heldout, incidence, index_by_id)
            prior = train_targets.mean(axis=0)
            prior /= max(float(prior.sum()), 1e-12)
            predictions = {
                "qesa": predict_r2(
                    checkpoint["qesa"], checkpoint["selected_regime"], int(checkpoint["selected_latent_aspects"]),
                    heldout_q, int(r2_config["hidden_dimension"]),
                ),
                "query_independent_aspect_prior": np.repeat(prior[None, :], len(heldout), axis=0),
                "shuffled_group_qesa": predict_r2(
                    checkpoint["shuffled_qesa"], checkpoint["selected_regime"], int(checkpoint["selected_latent_aspects"]),
                    heldout_q, int(r2_config["hidden_dimension"]),
                ),
                "frozen_embedding_linear_probe": predict_linear(
                    r1_checkpoint["linear_probe"], heldout_q, incidence.shape[1]
                ),
            }
            per_method = {
                method: query_metrics(heldout_targets, heldout_affinities, heldout_owners, values, recall_k)
                for method, values in predictions.items()
            }
            for query_index, row in enumerate(heldout):
                for method in methods:
                    metric_rows.append({
                        "query_id": row["query_id"],
                        "persona_id": row["persona_id"],
                        "query_session_id": row["query_session_id"],
                        "group_count": len(row["groups"]),
                        "seed": int(seed),
                        "fold": fold,
                        "method": method,
                        "prototype_recall_k": recall_k,
                        **per_method[method][query_index],
                    })
            unit_manifests.append(manifest)

    expected_rows = len(r2_config["fixed_seeds"]) * 1355 * len(methods)
    keys = [(row["seed"], row["query_id"], row["method"]) for row in metric_rows]
    if len(metric_rows) != expected_rows or len(keys) != len(set(keys)):
        raise RuntimeError(f"Gate B2 metric coverage invalid: rows={len(metric_rows)} expected={expected_rows} unique={len(set(keys))}")

    comparisons, raw_p = {}, {}
    number = 0
    for metric in gate["primary_metrics"]:
        comparisons[metric] = {}
        for comparator in gate["required_comparators"]:
            all_rows = paired_effect(metric_rows, comparator, metric, "all")
            multi_rows = paired_effect(metric_rows, comparator, metric, "K>=2")
            seed_effects = {
                str(seed): float(np.mean([row["delta"] for row in all_rows if int(row["seed"]) == int(seed)]))
                for seed in r2_config["fixed_seeds"]
            }
            comparisons[metric][comparator] = {
                "all": summarize_effect(all_rows, 20260714 + number, gate),
                "K>=2": summarize_effect(multi_rows, 20261714 + number, gate),
                "seed_effects": seed_effects,
            }
            key = f"{metric}:{comparator}"
            raw_p[key] = comparisons[metric][comparator]["all"]["cluster_sign_flip_p_two_sided"]
            number += 1
    adjusted = STATS["holm"](raw_p)
    for key, value in adjusted.items():
        metric, comparator = key.split(":", 1)
        comparisons[metric][comparator]["all"]["holm_adjusted_p"] = value

    rule = gate["gate_rule"]
    effects_pass = all(
        comparisons[metric][comparator]["all"]["session_clustered_ci95"][0] > 0
        and comparisons[metric][comparator]["K>=2"]["session_clustered_ci95"][0] > 0
        and all(effect > 0 for effect in comparisons[metric][comparator]["seed_effects"].values())
        for metric in gate["primary_metrics"] for comparator in gate["required_comparators"]
    )
    cosine_values = [manifest["qesa_training"]["aspect_map_pairwise_cosine_mean"] for manifest in unit_manifests]
    std_values = [manifest["qesa_training"]["prototype_coordinate_std_mean"] for manifest in unit_manifests]
    collapse = {
        "mean_aspect_map_pairwise_cosine": float(np.mean(cosine_values)),
        "maximum_unit_aspect_map_pairwise_cosine_mean": float(np.max(cosine_values)),
        "mean_prototype_coordinate_std": float(np.mean(std_values)),
    }
    collapse_pass = (
        collapse["mean_aspect_map_pairwise_cosine"] <= float(rule["mean_aspect_map_pairwise_cosine_at_most"])
        and collapse["maximum_unit_aspect_map_pairwise_cosine_mean"] < float(rule["maximum_unit_aspect_map_pairwise_cosine_mean_below"])
        and collapse["mean_prototype_coordinate_std"] >= float(rule["mean_prototype_coordinate_std_at_least"])
    )
    gate_passed = effects_pass and collapse_pass
    output_csv = VERSION_ROOT / "aggregate" / "v21_qesa_gate_b2_representation_metrics.csv"
    write_csv_new(output_csv, metric_rows)
    result = {
        "schema_version": "v21-qesa-gate-b2-result-1",
        "status": "passed" if gate_passed else "not_passed",
        "source_run_id": gate["source_run_id"],
        "metric_rows": len(metric_rows),
        "comparisons": comparisons,
        "collapse_diagnostics": collapse,
        "effects_pass": effects_pass,
        "collapse_pass": collapse_pass,
        "role_boundary": gate["role_boundary"],
        "hashes": {
            "gate_config": sha256_file(VERSION_ROOT / "configs" / "v21_qesa_gate_b2.json"),
            "metrics": sha256_file(output_csv),
            "source_r2_summary": sha256_file(VERSION_ROOT / "aggregate" / "v21_qesa_oof_summary_r2.json"),
        },
        "calibration_read": False,
        "formal_test_read": False,
    }
    output_json = VERSION_ROOT / "aggregate" / "v21_qesa_gate_b2_summary.json"
    write_json_new(output_json, result)
    report = "# Method 1 v21 QESA Gate B2 Representation Validity\n\n"
    report += f"Gate B2 status: `{result['status']}`. This audit used frozen QESA r2 checkpoints on unseen session-blocked OOF queries and did not train or select a new model.\n\n"
    report += "| Metric | Comparator | All-query effect | Clustered 95% CI | K>=2 effect | K>=2 clustered 95% CI |\n"
    report += "| --- | --- | ---: | --- | ---: | --- |\n"
    for metric in gate["primary_metrics"]:
        for comparator in gate["required_comparators"]:
            all_result = comparisons[metric][comparator]["all"]
            multi = comparisons[metric][comparator]["K>=2"]
            report += (
                f"| {metric} | {comparator} | {all_result['effect_positive_favors_qesa']:+.6f} | "
                f"[{all_result['session_clustered_ci95'][0]:+.6f}, {all_result['session_clustered_ci95'][1]:+.6f}] | "
                f"{multi['effect_positive_favors_qesa']:+.6f} | "
                f"[{multi['session_clustered_ci95'][0]:+.6f}, {multi['session_clustered_ci95'][1]:+.6f}] |\n"
            )
    report += f"\nMean aspect-map cosine: `{collapse['mean_aspect_map_pairwise_cosine']:.6f}`; maximum unit mean cosine: `{collapse['maximum_unit_aspect_map_pairwise_cosine_mean']:.6f}`; mean prototype coordinate std: `{collapse['mean_prototype_coordinate_std']:.6f}`.\n\n"
    report += "Passing this representation gate permits QHDS development only. Direct flat/MLP retrieval and the matched pairwise factor graph remain mandatory end-to-end Gate C baselines. K>=2 denotes the multi-group/compositional subset. Calibration and formal test remained unread.\n"
    report_path = VERSION_ROOT / "reports" / "METHOD1_V21_QESA_GATE_B2_RESULTS.md"
    with report_path.open("x", encoding="utf-8") as handle:
        handle.write(report)
    print(json.dumps({"status": result["status"], "metric_rows": len(metric_rows), "effects_pass": effects_pass, "collapse_pass": collapse_pass}, sort_keys=True))


if __name__ == "__main__":
    main()
