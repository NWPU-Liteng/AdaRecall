#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import runpy
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


VERSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VERSION_ROOT / "src"))

from v21.io import load_config, sha256_file, write_csv_new, write_json_new  # noqa: E402


STATS = runpy.run_path(str(VERSION_ROOT / "scripts" / "92_aggregate_v21_qesa_gate.py"), run_name="v21_qesa_gate_helpers")


def main() -> None:
    config = load_config("v21_qesa_r2.json")
    gate = load_config("v21_qesa_gate_b_r2.json")
    run_dir = VERSION_ROOT / "artifacts" / "v21_qesa" / gate["run_id"]
    r1_metrics_path = VERSION_ROOT / "aggregate" / "v21_qesa_oof_metrics.csv"
    r1_metrics_hash = sha256_file(r1_metrics_path)
    r2_rows, manifests, errors = [], [], []
    for seed in config["fixed_seeds"]:
        for fold in range(int(config["outer_folds"])):
            unit_dir = run_dir / f"seed_{seed}" / f"fold_{fold}"
            manifest_path = unit_dir / "v21_qesa_r2_unit_manifest.json"
            if not manifest_path.is_file():
                errors.append(f"missing r2 unit manifest seed={seed} fold={fold}")
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            metrics_path = VERSION_ROOT / manifest["artifacts"]["metrics"]
            models_path = VERSION_ROOT / manifest["artifacts"]["models"]
            if sha256_file(metrics_path) != manifest["hashes"]["metrics"]:
                errors.append(f"metrics hash mismatch seed={seed} fold={fold}")
            if sha256_file(models_path) != manifest["hashes"]["models"]:
                errors.append(f"models hash mismatch seed={seed} fold={fold}")
            if sha256_file(VERSION_ROOT / "configs" / "v21_qesa_r2.json") != manifest["hashes"]["config"]:
                errors.append(f"config hash mismatch seed={seed} fold={fold}")
            if manifest["hashes"]["reused_r1_baseline_metrics"] != r1_metrics_hash:
                errors.append(f"reused baseline hash mismatch seed={seed} fold={fold}")
            inner = manifest["inner_fsa_diagnostics"]
            if not inner["transport_converged"] or inner["prototype_collapse_flag"]:
                errors.append(f"invalid inner FSA seed={seed} fold={fold}")
            if manifest.get("calibration_read") or manifest.get("formal_test_read"):
                errors.append(f"locked split access recorded seed={seed} fold={fold}")
            with metrics_path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            if len(rows) != int(manifest["metric_rows"]):
                errors.append(f"metric row count mismatch seed={seed} fold={fold}")
            keys = [(row["query_id"], row["method"], int(row["budget"])) for row in rows]
            if len(keys) != len(set(keys)):
                errors.append(f"duplicate r2 key seed={seed} fold={fold}")
            if set(row["method"] for row in rows) != {"qesa", "shuffled_group_qesa"}:
                errors.append(f"r2 method coverage mismatch seed={seed} fold={fold}")
            r2_rows.extend(rows)
            manifests.append(manifest)
    if errors:
        raise RuntimeError("v21 QESA r2 aggregation failed:\n- " + "\n- ".join(errors))

    with r1_metrics_path.open(encoding="utf-8", newline="") as handle:
        r1_rows = list(csv.DictReader(handle))
    reused_methods = {
        "query_independent_aspect_prior",
        "flat_cosine",
        "same_capacity_mlp_ranker",
        "linear_probe",
        "nearest_train_query_memorization",
    }
    combined = r2_rows + [row for row in r1_rows if row["method"] in reused_methods]
    expected_methods = {"qesa", "shuffled_group_qesa", *reused_methods}
    expected_budgets = set(map(int, config["budgets"]))
    expected_per_query = len(expected_methods) * len(expected_budgets)
    expected_total = len(config["fixed_seeds"]) * 1355 * expected_per_query
    keys = [(int(row["seed"]), row["query_id"], row["method"], int(row["budget"])) for row in combined]
    if len(combined) != expected_total or len(keys) != len(set(keys)):
        raise RuntimeError(f"combined exact coverage failed: rows={len(combined)} expected={expected_total} unique={len(set(keys))}")
    if set(row["method"] for row in combined) != expected_methods or set(int(row["budget"]) for row in combined) != expected_budgets:
        raise RuntimeError("combined method or budget coverage mismatch")
    for seed in config["fixed_seeds"]:
        query_counts = Counter(row["query_id"] for row in combined if int(row["seed"]) == int(seed))
        if len(query_counts) != 1355 or set(query_counts.values()) != {expected_per_query}:
            raise RuntimeError(f"seed {seed} lacks exact OOF coverage")

    comparisons, raw_p = {}, {}
    primary_budget = int(gate["primary_budget"])
    for number, comparator in enumerate(gate["required_comparators"]):
        all_pair = STATS["paired_rows"](combined, comparator, primary_budget, "all")
        multi_pair = STATS["paired_rows"](combined, comparator, primary_budget, "K>=2")
        seed_effects = {}
        for seed in config["fixed_seeds"]:
            values = [row["delta"] for row in all_pair if int(row["seed"]) == int(seed)]
            seed_effects[str(seed)] = float(np.mean(values))
        comparisons[comparator] = {
            "all": STATS["summarize"](all_pair, 20260714 + number, int(gate["bootstrap_samples"]), int(gate["cluster_sign_flip_samples"])),
            "K>=2": STATS["summarize"](multi_pair, 20261714 + number, int(gate["bootstrap_samples"]), int(gate["cluster_sign_flip_samples"])),
            "seed_effects": seed_effects,
        }
        raw_p[comparator] = comparisons[comparator]["all"]["cluster_sign_flip_p_two_sided"]
    for comparator, adjusted in STATS["holm"](raw_p).items():
        comparisons[comparator]["all"]["holm_adjusted_p"] = adjusted

    subset_rules = {
        "K=1": lambda count: count == 1,
        "K=2": lambda count: count == 2,
        "K=3": lambda count: count == 3,
        "K=4": lambda count: count == 4,
        "K>=2": lambda count: count >= 2,
    }
    subset_summary = {}
    for subset, predicate in subset_rules.items():
        subset_summary[subset] = {}
        for method in sorted(expected_methods):
            selected = [
                row for row in combined
                if row["method"] == method and int(row["budget"]) == primary_budget and predicate(int(row["group_count"]))
            ]
            subset_summary[subset][method] = {
                "rows": len(selected),
                "mean_group_coverage": float(np.mean([float(row["group_coverage"]) for row in selected])),
                "all_groups_covered_rate": float(np.mean([float(row["all_groups_covered"]) for row in selected])),
                "both_alternatives_selected_rate": float(np.mean([float(row["both_alternatives_selected_rate"]) for row in selected])),
            }

    gate_passed = all(
        comparisons[name]["all"]["session_clustered_ci95"][0] > 0
        and comparisons[name]["K>=2"]["session_clustered_ci95"][0] > 0
        and all(effect > 0 for effect in comparisons[name]["seed_effects"].values())
        for name in gate["required_comparators"]
    )
    latent_counts = Counter(int(manifest["selected_latent_aspects"]) for manifest in manifests)
    regime_counts = Counter(manifest["selected_regime"]["regime_id"] for manifest in manifests)
    collapse = {
        "aspect_map_pairwise_cosine_mean": float(np.mean([manifest["qesa_training"]["aspect_map_pairwise_cosine_mean"] for manifest in manifests])),
        "prototype_coordinate_std_mean": float(np.mean([manifest["qesa_training"]["prototype_coordinate_std_mean"] for manifest in manifests])),
    }
    output_csv = VERSION_ROOT / "aggregate" / "v21_qesa_oof_metrics_r2.csv"
    write_csv_new(output_csv, combined)
    result = {
        "schema_version": "v21-qesa-r2-gate-b-result-1",
        "status": "passed" if gate_passed else "not_passed",
        "revision_limit_reached": True,
        "run_id": gate["run_id"],
        "parent_negative_run_id": config["parent_negative_run_id"],
        "units": len(manifests),
        "metric_rows": len(combined),
        "primary_budget": primary_budget,
        "comparisons": comparisons,
        "subset_summary": subset_summary,
        "selected_latent_aspect_counts": {str(key): value for key, value in sorted(latent_counts.items())},
        "selected_regime_counts": dict(sorted(regime_counts.items())),
        "collapse_diagnostics": collapse,
        "gate_rule": gate["gate_rule"],
        "multiple_comparison_correction": gate["multiple_comparison_correction"],
        "hashes": {
            "r2_config": sha256_file(VERSION_ROOT / "configs" / "v21_qesa_r2.json"),
            "r2_gate_config": sha256_file(VERSION_ROOT / "configs" / "v21_qesa_gate_b_r2.json"),
            "reused_r1_metrics": r1_metrics_hash,
            "combined_metrics": sha256_file(output_csv),
        },
        "calibration_read": False,
        "formal_test_read": False,
    }
    output_json = VERSION_ROOT / "aggregate" / "v21_qesa_oof_summary_r2.json"
    write_json_new(output_json, result)
    report = "# Method 1 v21 QESA r2 Generalization Results\n\n"
    report += f"Gate B status: `{result['status']}`. This is the single preregistered mechanism-driven revision after the immutable r1 negative result.\n\n"
    report += "| Comparator (B=16) | All-query effect | Clustered 95% CI | K>=2 effect | K>=2 clustered 95% CI | Holm p |\n"
    report += "| --- | ---: | --- | ---: | --- | ---: |\n"
    for comparator in gate["required_comparators"]:
        all_result, multi = comparisons[comparator]["all"], comparisons[comparator]["K>=2"]
        report += (
            f"| {comparator} | {all_result['effect_qesa_minus_comparator']:+.6f} | "
            f"[{all_result['session_clustered_ci95'][0]:+.6f}, {all_result['session_clustered_ci95'][1]:+.6f}] | "
            f"{multi['effect_qesa_minus_comparator']:+.6f} | "
            f"[{multi['session_clustered_ci95'][0]:+.6f}, {multi['session_clustered_ci95'][1]:+.6f}] | "
            f"{all_result['holm_adjusted_p']:.6g} |\n"
        )
    report += f"\nMean aspect-map pairwise cosine: `{collapse['aspect_map_pairwise_cosine_mean']:.6f}`. Mean query-conditioned prototype coordinate standard deviation: `{collapse['prototype_coordinate_std_mean']:.6f}`.\n\n"
    report += "K>=2 is the multi-group/compositional subset; each inner group remains an OR set of two acceptable alternatives. Calibration and formal test remained locked and unread. If this revision does not pass, QESA and downstream QHDS promotion stop as a preserved negative result.\n"
    report_path = VERSION_ROOT / "reports" / "METHOD1_V21_QESA_R2_GENERALIZATION_RESULTS.md"
    with report_path.open("x", encoding="utf-8") as handle:
        handle.write(report)
    print(json.dumps({"status": result["status"], "units": len(manifests), "metric_rows": len(combined)}, sort_keys=True))


if __name__ == "__main__":
    main()
