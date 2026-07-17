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

from v21.io import load_config, read_json, sha256_file, write_json_new  # noqa: E402
from v21.qesa import LatentAspectQESA  # noqa: E402


def entropy(probabilities: np.ndarray) -> np.ndarray:
    return -(probabilities * np.log(np.maximum(probabilities, 1e-12))).sum(axis=1)


def metric_diagnostics(rows: list[dict], comparators: list[str]) -> dict:
    indexed = {
        (int(row["seed"]), row["query_id"], row["method"], int(row["budget"])): row
        for row in rows
    }
    output = {}
    for budget in (8, 16, 32):
        output[str(budget)] = {}
        qesa_rows = [row for row in rows if row["method"] == "qesa" and int(row["budget"]) == budget]
        for comparator in comparators:
            all_delta, multi_delta = [], []
            for row in qesa_rows:
                baseline = indexed[(int(row["seed"]), row["query_id"], comparator, budget)]
                delta = float(row["group_coverage"]) - float(baseline["group_coverage"])
                all_delta.append(delta)
                if int(row["group_count"]) >= 2:
                    multi_delta.append(delta)
            output[str(budget)][comparator] = {
                "all_query_effect": float(np.mean(all_delta)),
                "multi_group_effect": float(np.mean(multi_delta)),
            }
        output[str(budget)]["qesa_absolute"] = {
            "mean_group_coverage": float(np.mean([float(row["group_coverage"]) for row in qesa_rows])),
            "all_groups_covered_rate": float(np.mean([float(row["all_groups_covered"]) for row in qesa_rows])),
            "both_alternatives_selected_rate": float(np.mean([float(row["both_alternatives_selected_rate"]) for row in qesa_rows])),
        }
    return output


def main() -> None:
    config = load_config("v21_qesa.json")
    gate = load_config("v21_qesa_gate_b.json")
    data = load_config("v21_data.json")
    metrics_path = VERSION_ROOT / "aggregate" / "v21_qesa_oof_metrics.csv"
    with metrics_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    helpers = runpy.run_path(str(VERSION_ROOT / "scripts" / "90_train_v21_qesa_oof.py"), run_name="v21_qesa_helpers")
    query_vectors = helpers["load_query_vectors"](data)
    rows_by_fold = helpers["load_rows"](data, query_vectors, int(config["outer_folds"]))
    input_dimension = len(next(iter(query_vectors.values())))
    run_dir = VERSION_ROOT / "artifacts" / "v21_qesa" / gate["run_id"]
    unit_diagnostics = []
    pooled_latent = []
    pooled_prototype = []
    pooled_l1_from_unit_mean = []
    for seed in config["fixed_seeds"]:
        for fold in range(int(config["outer_folds"])):
            unit_dir = run_dir / f"seed_{seed}" / f"fold_{fold}"
            manifest = json.loads((unit_dir / "v21_qesa_unit_manifest.json").read_text(encoding="utf-8"))
            checkpoint = torch.load(unit_dir / "v21_qesa_models.pt", map_location="cpu")
            latent_count = int(checkpoint["selected_latent_aspects"])
            prototype_count = int(checkpoint["qesa"]["aspect_prototype_logits"].shape[1])
            model = LatentAspectQESA(
                input_dimension,
                int(config["hidden_dimension"]),
                latent_count,
                prototype_count,
            )
            model.load_state_dict(checkpoint["qesa"])
            model.eval()
            query = torch.as_tensor(np.stack([row["query_embedding"] for row in rows_by_fold[fold]]))
            with torch.no_grad():
                latent, aspect_map, prototype = model(query)
            latent_array = latent.numpy()
            prototype_array = prototype.numpy()
            aspect_array = aspect_map.numpy()
            unit_mean = prototype_array.mean(axis=0, keepdims=True)
            l1 = np.abs(prototype_array - unit_mean).sum(axis=1)
            normalized_aspects = aspect_array / np.maximum(np.linalg.norm(aspect_array, axis=1, keepdims=True), 1e-12)
            cosine = normalized_aspects @ normalized_aspects.T
            off_diagonal = cosine[~np.eye(latent_count, dtype=bool)]
            unit_diagnostics.append({
                "seed": int(seed),
                "fold": fold,
                "queries": len(query),
                "latent_aspects": latent_count,
                "latent_entropy_mean": float(entropy(latent_array).mean()),
                "latent_effective_count_mean": float(np.exp(entropy(latent_array)).mean()),
                "latent_max_probability_mean": float(latent_array.max(axis=1).mean()),
                "prototype_entropy_mean": float(entropy(prototype_array).mean()),
                "prototype_effective_count_mean": float(np.exp(entropy(prototype_array)).mean()),
                "prototype_max_probability_mean": float(prototype_array.max(axis=1).mean()),
                "prototype_coordinate_std_mean": float(prototype_array.std(axis=0).mean()),
                "prototype_l1_from_unit_mean_mean": float(l1.mean()),
                "prototype_l1_from_unit_mean_p95": float(np.quantile(l1, 0.95)),
                "aspect_map_pairwise_cosine_mean": float(off_diagonal.mean()),
                "aspect_map_pairwise_cosine_max": float(off_diagonal.max()),
                "inner_fsa_converged": bool(manifest["inner_fsa_diagnostics"]["transport_converged"]),
                "inner_fsa_collapse": bool(manifest["inner_fsa_diagnostics"]["prototype_collapse_flag"]),
            })
            pooled_latent.append(latent_array)
            pooled_prototype.append(prototype_array)
            pooled_l1_from_unit_mean.append(l1)

    latent_effective_ratio = [
        row["latent_effective_count_mean"] / row["latent_aspects"] for row in unit_diagnostics
    ]
    prototype_arrays = np.concatenate(pooled_prototype)
    l1_values = np.concatenate(pooled_l1_from_unit_mean)
    diagnostics = metric_diagnostics(rows, list(gate["required_comparators"]))
    result = {
        "schema_version": "v21-qesa-failure-diagnosis-1",
        "status": "completed_train_only_negative_result_diagnosis",
        "source_gate_b_status": read_json("aggregate/v21_qesa_oof_summary.json")["status"],
        "run_id": gate["run_id"],
        "metric_diagnostics_by_budget": diagnostics,
        "prediction_diagnostics": {
            "unit_rows": unit_diagnostics,
            "latent_effective_count_ratio_mean": float(np.mean(latent_effective_ratio)),
            "prototype_coordinate_std_mean_across_units": float(np.mean([row["prototype_coordinate_std_mean"] for row in unit_diagnostics])),
            "prototype_l1_from_unit_mean_mean": float(l1_values.mean()),
            "prototype_l1_from_unit_mean_p95": float(np.quantile(l1_values, 0.95)),
            "prototype_entropy_mean": float(entropy(prototype_arrays).mean()),
            "prototype_effective_count_mean": float(np.exp(entropy(prototype_arrays)).mean()),
        },
        "diagnostic_interpretation_rule": {
            "near_prior_warning_if_prototype_l1_from_unit_mean_mean_below": 0.10,
            "diffuse_latent_warning_if_effective_count_ratio_above": 0.75,
        },
        "hashes": {
            "source_metrics": sha256_file(metrics_path),
            "source_summary": sha256_file(VERSION_ROOT / "aggregate" / "v21_qesa_oof_summary.json"),
            "training_config": sha256_file(VERSION_ROOT / "configs" / "v21_qesa.json"),
            "gate_config": sha256_file(VERSION_ROOT / "configs" / "v21_qesa_gate_b.json"),
        },
        "calibration_read": False,
        "formal_test_read": False,
    }
    prediction = result["prediction_diagnostics"]
    result["warnings"] = {
        "near_query_independent_prototype_prediction": (
            prediction["prototype_l1_from_unit_mean_mean"]
            < result["diagnostic_interpretation_rule"]["near_prior_warning_if_prototype_l1_from_unit_mean_mean_below"]
        ),
        "diffuse_latent_distribution": (
            prediction["latent_effective_count_ratio_mean"]
            > result["diagnostic_interpretation_rule"]["diffuse_latent_warning_if_effective_count_ratio_above"]
        ),
    }
    output_path = VERSION_ROOT / "aggregate" / "v21_qesa_failure_diagnosis_r1.json"
    write_json_new(output_path, result)

    report = "# Method 1 v21 QESA Failure Diagnosis\n\n"
    report += "This is a train-only post-Gate diagnostic. It does not revise Gate B and does not read calibration or formal test data. The failed OOF artifacts remain immutable.\n\n"
    report += "| Budget | vs prior | vs flat cosine | vs shuffled QESA | vs same-capacity MLP |\n"
    report += "| ---: | ---: | ---: | ---: | ---: |\n"
    for budget in (8, 16, 32):
        row = diagnostics[str(budget)]
        report += (
            f"| {budget} | {row['query_independent_aspect_prior']['all_query_effect']:+.6f} | "
            f"{row['flat_cosine']['all_query_effect']:+.6f} | "
            f"{row['shuffled_group_qesa']['all_query_effect']:+.6f} | "
            f"{row['same_capacity_mlp_ranker']['all_query_effect']:+.6f} |\n"
        )
    report += "\nPrediction diagnostics:\n\n"
    report += f"- Mean latent effective-count ratio: `{prediction['latent_effective_count_ratio_mean']:.6f}`.\n"
    report += f"- Mean prototype L1 distance from its unit mean: `{prediction['prototype_l1_from_unit_mean_mean']:.6f}`.\n"
    report += f"- Mean prototype-coordinate standard deviation: `{prediction['prototype_coordinate_std_mean_across_units']:.6f}`.\n"
    report += f"- Near-query-independent warning: `{result['warnings']['near_query_independent_prototype_prediction']}`.\n"
    report += f"- Diffuse-latent warning: `{result['warnings']['diffuse_latent_distribution']}`.\n\n"
    report += "A revision is scientifically justified only if this audit identifies a predeclared representational failure. It must use a new run ID and preserve this negative result. QHDS remains blocked while Gate B is not passed.\n"
    report_path = VERSION_ROOT / "reports" / "METHOD1_V21_QESA_FAILURE_DIAGNOSIS.md"
    with report_path.open("x", encoding="utf-8") as handle:
        handle.write(report)
    print(json.dumps({"warnings": result["warnings"], "prediction_diagnostics": prediction}, sort_keys=True))


if __name__ == "__main__":
    main()
