#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


VERSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VERSION_ROOT / "src"))

from v21.io import load_config, read_json, sha256_file, write_csv_new, write_json_new  # noqa: E402


METHODS = ("bundle_deepsets", "independent_member", "shuffled_bundle_deepsets", "pairwise_only")
METRICS = ("gold_compatible_bundle_hit_rate", "all_groups_covered_rate", "bundle_recall", "bundle_member_redundancy")


def float_row(row: dict) -> dict:
    for key in METRICS + ("nearest_train_query_cosine",):
        row[key] = float(row[key])
    for key in ("seed", "fold", "query_session_id", "actual_group_count", "predicted_aspect_count", "candidate_bundle_count"):
        row[key] = int(row[key])
    return row


def percentile_interval(values: np.ndarray) -> list[float]:
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def paired_intervals(rows: list[dict], method: str, comparator: str, metric: str, category: str, seed: int) -> dict:
    subset = [row for row in rows if category == "all" or category == "K>=2" or int(row["actual_group_count"]) == int(category)]
    if category == "K>=2":
        subset = [row for row in subset if int(row["actual_group_count"]) >= 2]
    method_rows = {(row["seed"], row["query_id"]): row for row in subset if row["method"] == method}
    comparator_rows = {(row["seed"], row["query_id"]): row for row in subset if row["method"] == comparator}
    keys = sorted(set(method_rows).intersection(comparator_rows))
    deltas = np.asarray([method_rows[key][metric] - comparator_rows[key][metric] for key in keys], dtype=np.float64)
    sessions = [method_rows[key] for key in keys]
    rng = np.random.default_rng(seed)
    query_bootstrap = np.asarray([deltas[rng.integers(len(deltas), size=len(deltas))].mean() for _ in range(2000)])
    cluster_to_indices: dict[tuple[int, str, int], list[int]] = defaultdict(list)
    for index, row in enumerate(sessions):
        cluster_to_indices[(int(row["seed"]), str(row["persona_id"]), int(row["query_session_id"]))].append(index)
    clusters = list(cluster_to_indices.values())
    clustered = []
    for _ in range(2000):
        chosen = rng.integers(len(clusters), size=len(clusters))
        positions = np.concatenate([clusters[index] for index in chosen])
        clustered.append(float(deltas[positions].mean()))
    return {
        "effect": float(deltas.mean()), "paired_rows": len(deltas),
        "paired_query_ci95": percentile_interval(query_bootstrap),
        "session_clustered_ci95": percentile_interval(np.asarray(clustered)),
        "wins": int((deltas > 0).sum()), "ties": int((deltas == 0).sum()), "losses": int((deltas < 0).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate one immutable v21 bundle-locality OOF attempt.")
    parser.add_argument("--run-id", default="v21_bundle_locality_oof_r1_20260714")
    parser.add_argument("--attempt", default="attempt_001")
    args = parser.parse_args()
    config = load_config("v21_bundle_locality.json")
    if args.run_id != config["run_id"]:
        raise ValueError("run ID must match bundle-locality preregistration")
    rows, manifests = [], []
    root = VERSION_ROOT / "artifacts" / "v21_bundle_locality" / args.run_id
    for seed in config["fixed_seeds"]:
        for fold in range(int(config["outer_folds"])):
            unit = root / f"seed_{seed}" / f"fold_{fold}" / "attempts" / args.attempt
            manifest_path = unit / "v21_bundle_locality_unit_manifest.json"
            if not manifest_path.is_file():
                raise FileNotFoundError(f"missing completed OOF unit: {manifest_path}")
            manifest = read_json(manifest_path)
            if manifest["status"] != "completed" or manifest["attempt"] != args.attempt:
                raise ValueError(f"invalid OOF unit manifest: {manifest_path}")
            manifests.append(manifest)
            with (unit / "v21_bundle_locality_query_metrics.csv").open(newline="", encoding="utf-8") as handle:
                rows.extend(float_row(dict(item)) for item in csv.DictReader(handle))
    expected = len(config["fixed_seeds"]) * sum(manifest["outer_heldout_multi_group_queries"] for manifest in manifests[:5])
    if not rows or any(row["method"] not in METHODS for row in rows):
        raise ValueError("bundle locality metrics are empty or have an unknown method")
    raw_output = VERSION_ROOT / "aggregate" / "v21_bundle_locality_oof_metrics_r1.csv"
    write_csv_new(raw_output, rows)
    categories = ("all", "2", "3", "4", "K>=2")
    means = {}
    for method in METHODS:
        means[method] = {}
        for category in categories:
            subset = [row for row in rows if row["method"] == method and (category in ("all", "K>=2") or row["actual_group_count"] == int(category))]
            if category == "K>=2":
                subset = [row for row in subset if row["actual_group_count"] >= 2]
            if subset:
                means[method][category] = {metric: float(np.mean([row[metric] for row in subset])) for metric in METRICS}
                means[method][category]["rows"] = len(subset)
    comparisons = {}
    for comparator in ("independent_member", "shuffled_bundle_deepsets", "pairwise_only"):
        comparisons[comparator] = {}
        for category in categories:
            comparisons[comparator][category] = {
                metric: paired_intervals(rows, "bundle_deepsets", comparator, metric, category, 20260714 + len(metric) + len(category))
                for metric in METRICS
            }
    audit = {
        "query_id_is_not_a_model_input": all(item["generalization_audits"]["query_id_is_not_a_model_input"] for item in manifests),
        "query_id_permutation_audit_recorded": all(item["generalization_audits"]["query_id_permutation_audit_recorded"] for item in manifests),
        "nearest_train_query_similarity_recorded": all(item["generalization_audits"]["nearest_train_query_similarity_recorded"] for item in manifests),
        "outer_train_only_supervision": all(item["generalization_audits"]["outer_train_only_supervision"] for item in manifests),
        "inference_rejects_evidence_groups_and_acceptable_members": all(item["generalization_audits"]["inference_rejects_evidence_groups_and_acceptable_members"] for item in manifests),
        "mean_nearest_train_query_cosine": float(np.mean([row["nearest_train_query_cosine"] for row in rows if row["method"] == "bundle_deepsets"])),
        "predicted_cardinality_exact_rate": float(np.mean([row["actual_group_count"] == row["predicted_aspect_count"] for row in rows if row["method"] == "bundle_deepsets"])),
    }
    result = {
        "schema_version": "v21-evidence-bundle-locality-oof-summary-1", "status": "completed", "revision_id": config["revision_id"],
        "run_id": args.run_id, "attempt": args.attempt, "units": len(manifests), "metric_rows": len(rows),
        "methods": list(METHODS), "means": means, "comparisons_bundle_deepsets_minus": comparisons, "generalization_audits": audit,
        "hashes": {"raw_metrics": sha256_file(raw_output), "config": sha256_file(VERSION_ROOT / "configs" / "v21_bundle_locality.json")},
        "calibration_read": False, "formal_test_read": False,
    }
    summary_output = VERSION_ROOT / "aggregate" / "v21_bundle_locality_oof_summary_r1.json"
    write_json_new(summary_output, result)
    report = "# Method 1 v21 Evidence-Bundle Locality OOF Results\n\n"
    report += "This is train-only supervision-derived compositional evidence locality, not observed runtime co-access. The scorer is evaluated outer-OOF; inference uses no evidence group, acceptable member, query ID, or gold bundle.\n\n"
    report += "| Comparator | Subset | Metric | Effect | Session-clustered 95% CI |\n| --- | --- | --- | ---: | --- |\n"
    for comparator, by_category in comparisons.items():
        for category in ("2", "3", "4", "K>=2"):
            for metric in METRICS:
                item = by_category[category][metric]
                report += f"| {comparator} | {category} | {metric} | {item['effect']:+.6f} | [{item['session_clustered_ci95'][0]:+.6f}, {item['session_clustered_ci95'][1]:+.6f}] |\n"
    report += "\nCalibration and formal test remained unread. Dense QHDS remains a quality reference; this report does not claim a sparse-path or hardware result.\n"
    report_output = VERSION_ROOT / "reports" / "METHOD1_V21_EVIDENCE_BUNDLE_LOCALITY_RESULTS.md"
    with report_output.open("x", encoding="utf-8") as handle:
        handle.write(report)
    print(json.dumps({"status": result["status"], "units": len(manifests), "metric_rows": len(rows), "expected_reference": expected}, sort_keys=True))


if __name__ == "__main__":
    main()
