#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


VERSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VERSION_ROOT / "src"))

from v21.io import load_config, sha256_file, write_csv_new, write_json_new  # noqa: E402


def quantile_ci(values: np.ndarray) -> list[float]:
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def paired_rows(rows: list[dict], comparator: str, budget: int, subset: str) -> list[dict]:
    selected = [
        row for row in rows
        if int(row["budget"]) == budget
        and row["method"] in {"qesa", comparator}
        and (subset == "all" or int(row["group_count"]) >= 2)
    ]
    indexed = {(int(row["seed"]), row["query_id"], row["method"]): row for row in selected}
    keys = sorted({(int(row["seed"]), row["query_id"]) for row in selected})
    output = []
    for seed, query_id in keys:
        qesa = indexed.get((seed, query_id, "qesa"))
        baseline = indexed.get((seed, query_id, comparator))
        if qesa is None or baseline is None:
            raise RuntimeError(f"unpaired QESA comparison seed={seed} query={query_id} comparator={comparator}")
        output.append({
            "seed": seed,
            "query_id": query_id,
            "persona_id": qesa["persona_id"],
            "query_session_id": qesa["query_session_id"],
            "delta": float(qesa["group_coverage"]) - float(baseline["group_coverage"]),
        })
    return output


def clustered_values(rows: list[dict]) -> np.ndarray:
    clusters: dict[tuple[int, str, str], list[float]] = defaultdict(list)
    for row in rows:
        key = (int(row["seed"]), str(row["persona_id"]), str(row["query_session_id"]))
        clusters[key].append(float(row["delta"]))
    return np.asarray([np.mean(clusters[key]) for key in sorted(clusters)], dtype=np.float64)


def query_bootstrap(rows: list[dict], seed: int, samples: int) -> list[float]:
    values = np.asarray([float(row["delta"]) for row in rows], dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 250):
        count = min(250, samples - start)
        positions = rng.integers(0, len(values), size=(count, len(values)))
        means[start : start + count] = values[positions].mean(axis=1)
    return quantile_ci(means)


def cluster_bootstrap(rows: list[dict], seed: int, samples: int) -> list[float]:
    clusters: dict[tuple[int, str, str], list[float]] = defaultdict(list)
    for row in rows:
        key = (int(row["seed"]), str(row["persona_id"]), str(row["query_session_id"]))
        clusters[key].append(float(row["delta"]))
    keys = sorted(clusters)
    sums = np.asarray([sum(clusters[key]) for key in keys], dtype=np.float64)
    counts = np.asarray([len(clusters[key]) for key in keys], dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for sample in range(samples):
        positions = rng.integers(0, len(keys), size=len(keys))
        means[sample] = sums[positions].sum() / counts[positions].sum()
    return quantile_ci(means)


def sign_flip_p(rows: list[dict], seed: int, samples: int) -> float:
    values = clustered_values(rows)
    observed = abs(float(values.mean()))
    rng = np.random.default_rng(seed)
    exceed = 0
    for start in range(0, samples, 1000):
        count = min(1000, samples - start)
        signs = rng.integers(0, 2, size=(count, len(values)), dtype=np.int8) * 2 - 1
        exceed += int((np.abs((signs * values).mean(axis=1)) >= observed).sum())
    return float((exceed + 1) / (samples + 1))


def holm(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=p_values.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, name in enumerate(ordered):
        running = max(running, (total - rank) * p_values[name])
        adjusted[name] = min(1.0, running)
    return adjusted


def summarize(rows: list[dict], seed: int, bootstrap_samples: int, sign_samples: int) -> dict:
    delta = np.asarray([float(row["delta"]) for row in rows], dtype=np.float64)
    return {
        "paired_rows": len(rows),
        "effect_qesa_minus_comparator": float(delta.mean()),
        "paired_query_ci95": query_bootstrap(rows, seed, bootstrap_samples),
        "session_clustered_ci95": cluster_bootstrap(rows, seed + 10000, bootstrap_samples),
        "cluster_sign_flip_p_two_sided": sign_flip_p(rows, seed + 20000, sign_samples),
        "wins": int((delta > 0).sum()),
        "losses": int((delta < 0).sum()),
        "ties": int((delta == 0).sum()),
    }


def main() -> None:
    training = load_config("v21_qesa.json")
    gate = load_config("v21_qesa_gate_b.json")
    run_id = str(gate["run_id"])
    run_dir = VERSION_ROOT / "artifacts" / "v21_qesa" / run_id
    all_rows: list[dict] = []
    manifests = []
    errors = []
    expected_methods = {"qesa", *training["required_baselines"]}
    expected_budgets = set(map(int, training["budgets"]))
    expected_queries_by_seed = 1355
    for seed in training["fixed_seeds"]:
        for fold in range(int(training["outer_folds"])):
            unit_dir = run_dir / f"seed_{seed}" / f"fold_{fold}"
            manifest_path = unit_dir / "v21_qesa_unit_manifest.json"
            if not manifest_path.is_file():
                errors.append(f"missing unit manifest seed={seed} fold={fold}")
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            metrics_path = VERSION_ROOT / manifest["artifacts"]["metrics"]
            models_path = VERSION_ROOT / manifest["artifacts"]["models"]
            if sha256_file(metrics_path) != manifest["hashes"]["metrics"]:
                errors.append(f"metrics hash mismatch seed={seed} fold={fold}")
            if sha256_file(models_path) != manifest["hashes"]["models"]:
                errors.append(f"model hash mismatch seed={seed} fold={fold}")
            if sha256_file(VERSION_ROOT / "configs" / "v21_qesa.json") != manifest["hashes"]["config"]:
                errors.append(f"training config hash mismatch seed={seed} fold={fold}")
            diagnostics = manifest["inner_fsa_diagnostics"]
            if not diagnostics["transport_converged"] or diagnostics["prototype_collapse_flag"]:
                errors.append(f"invalid inner FSA seed={seed} fold={fold}")
            if manifest.get("calibration_read") or manifest.get("formal_test_read"):
                errors.append(f"locked split access recorded seed={seed} fold={fold}")
            with metrics_path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            if len(rows) != int(manifest["metric_rows"]):
                errors.append(f"metric row count mismatch seed={seed} fold={fold}")
            keys = [(row["query_id"], row["method"], int(row["budget"])) for row in rows]
            if len(keys) != len(set(keys)):
                errors.append(f"duplicate query/method/budget key seed={seed} fold={fold}")
            if set(row["method"] for row in rows) != expected_methods:
                errors.append(f"method coverage mismatch seed={seed} fold={fold}")
            if set(int(row["budget"]) for row in rows) != expected_budgets:
                errors.append(f"budget coverage mismatch seed={seed} fold={fold}")
            all_rows.extend(rows)
            manifests.append(manifest)
    if errors:
        raise RuntimeError("v21 QESA aggregation failed:\n- " + "\n- ".join(errors))

    by_seed: dict[int, list[dict]] = defaultdict(list)
    for row in all_rows:
        by_seed[int(row["seed"])].append(row)
    expected_per_query = len(expected_methods) * len(expected_budgets)
    for seed in training["fixed_seeds"]:
        rows = by_seed[int(seed)]
        query_counts = Counter(row["query_id"] for row in rows)
        if len(query_counts) != expected_queries_by_seed or set(query_counts.values()) != {expected_per_query}:
            errors.append(f"seed {seed} lacks exact OOF query/method/budget coverage")
    expected_total = len(training["fixed_seeds"]) * expected_queries_by_seed * expected_per_query
    if len(all_rows) != expected_total:
        errors.append(f"expected {expected_total} total rows, found {len(all_rows)}")
    if errors:
        raise RuntimeError("v21 QESA exact coverage validation failed:\n- " + "\n- ".join(errors))

    bootstrap_samples = int(gate["bootstrap_samples"])
    sign_samples = int(gate["cluster_sign_flip_samples"])
    primary_budget = int(gate["primary_budget"])
    comparisons = {}
    raw_p = {}
    for number, comparator in enumerate(gate["required_comparators"]):
        all_pair = paired_rows(all_rows, comparator, primary_budget, "all")
        multi_pair = paired_rows(all_rows, comparator, primary_budget, "K>=2")
        seed_effects = {}
        for seed in training["fixed_seeds"]:
            seed_rows = [row for row in all_pair if int(row["seed"]) == int(seed)]
            seed_effects[str(seed)] = float(np.mean([row["delta"] for row in seed_rows]))
        comparisons[comparator] = {
            "all": summarize(all_pair, 20260714 + number, bootstrap_samples, sign_samples),
            "K>=2": summarize(multi_pair, 20261714 + number, bootstrap_samples, sign_samples),
            "seed_effects": seed_effects,
        }
        raw_p[comparator] = comparisons[comparator]["all"]["cluster_sign_flip_p_two_sided"]
    adjusted = holm(raw_p)
    for comparator, value in adjusted.items():
        comparisons[comparator]["all"]["holm_adjusted_p"] = value

    subset_summary = {}
    subset_rules = {
        "K=1": lambda count: count == 1,
        "K=2": lambda count: count == 2,
        "K=3": lambda count: count == 3,
        "K=4": lambda count: count == 4,
        "K>=2": lambda count: count >= 2,
    }
    for subset, predicate in subset_rules.items():
        subset_summary[subset] = {}
        for method in sorted(expected_methods):
            selected = [
                row for row in all_rows
                if row["method"] == method and int(row["budget"]) == primary_budget
                and predicate(int(row["group_count"]))
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
    nearest = [
        float(row["nearest_train_query_cosine"]) for row in all_rows
        if row["method"] == "qesa" and int(row["budget"]) == primary_budget
    ]
    output_csv = VERSION_ROOT / "aggregate" / "v21_qesa_oof_metrics.csv"
    write_csv_new(output_csv, all_rows)
    result = {
        "schema_version": "v21-qesa-gate-b-result-1",
        "status": "passed" if gate_passed else "not_passed",
        "run_id": run_id,
        "units": len(manifests),
        "metric_rows": len(all_rows),
        "primary_budget": primary_budget,
        "comparisons": comparisons,
        "subset_summary": subset_summary,
        "selected_latent_aspect_counts": {str(key): value for key, value in sorted(latent_counts.items())},
        "nearest_train_query_cosine": {
            "mean": float(np.mean(nearest)),
            "p50": float(np.quantile(nearest, 0.5)),
            "p95": float(np.quantile(nearest, 0.95)),
            "maximum": float(np.max(nearest)),
            "exact_or_near_duplicate_rate_ge_0_999999": float(np.mean(np.asarray(nearest) >= 0.999999)),
        },
        "query_id_permutation_audit": {
            "query_id_used_as_model_input": False,
            "all_unit_interfaces_mark_permutation_invariant": all(
                manifest["generalization_audits"]["query_id_permutation_invariant_by_interface"]
                for manifest in manifests
            ),
        },
        "gate_rule": gate["gate_rule"],
        "multiple_comparison_correction": gate["multiple_comparison_correction"],
        "gate_config_sha256": sha256_file(VERSION_ROOT / "configs" / "v21_qesa_gate_b.json"),
        "training_config_sha256": sha256_file(VERSION_ROOT / "configs" / "v21_qesa.json"),
        "metrics_csv": str(output_csv.relative_to(VERSION_ROOT)),
        "metrics_sha256": sha256_file(output_csv),
        "calibration_read": False,
        "formal_test_read": False,
    }
    output_json = VERSION_ROOT / "aggregate" / "v21_qesa_oof_summary.json"
    write_json_new(output_json, result)

    report = "# Method 1 v21 QESA Generalization Results\n\n"
    report += f"Gate B status: `{result['status']}`. All {len(manifests)} fixed-seed, session-blocked outer OOF units were aggregated without replacing failed seeds or folds.\n\n"
    report += "| Comparator (B=16) | All-query effect | Clustered 95% CI | K>=2 effect | K>=2 clustered 95% CI | Holm p |\n"
    report += "| --- | ---: | --- | ---: | --- | ---: |\n"
    for comparator in gate["required_comparators"]:
        all_result = comparisons[comparator]["all"]
        multi_result = comparisons[comparator]["K>=2"]
        report += (
            f"| {comparator} | {all_result['effect_qesa_minus_comparator']:+.6f} | "
            f"[{all_result['session_clustered_ci95'][0]:+.6f}, {all_result['session_clustered_ci95'][1]:+.6f}] | "
            f"{multi_result['effect_qesa_minus_comparator']:+.6f} | "
            f"[{multi_result['session_clustered_ci95'][0]:+.6f}, {multi_result['session_clustered_ci95'][1]:+.6f}] | "
            f"{all_result['holm_adjusted_p']:.6g} |\n"
        )
    report += "\nEach two-member evidence group is an OR set of acceptable alternatives. K>=2 denotes the multi-group/compositional subset, not a higher-order member relation. Query IDs are metadata only and are absent from the QESA model interface.\n\n"
    report += "Calibration and formal test remained locked and unread. Gate B passing permits QHDS development; it does not establish an independent hypergraph advantage over a matched pairwise factor graph.\n"
    report_path = VERSION_ROOT / "reports" / "METHOD1_V21_QESA_GENERALIZATION_RESULTS.md"
    with report_path.open("x", encoding="utf-8") as handle:
        handle.write(report)
    print(json.dumps({"status": result["status"], "units": len(manifests), "metric_rows": len(all_rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
