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


STATS = runpy.run_path(str(VERSION_ROOT / "scripts" / "92_aggregate_v21_qesa_gate.py"), run_name="v21_stats")


def paired(rows, comparator, budget, subset, metric):
    selected = [row for row in rows if int(row["budget"]) == budget and row["method"] in {"qhds", comparator} and (subset == "all" or int(row["group_count"]) >= 2)]
    indexed = {(int(row["seed"]), row["query_id"], row["method"]): row for row in selected}
    output = []
    for seed, query_id in sorted({(int(row["seed"]), row["query_id"]) for row in selected}):
        qhds, baseline = indexed[(seed, query_id, "qhds")], indexed[(seed, query_id, comparator)]
        output.append({"seed": seed, "query_id": query_id, "persona_id": qhds["persona_id"], "query_session_id": qhds["query_session_id"], "delta": float(qhds[metric]) - float(baseline[metric])})
    return output


def summarize_effect(rows, seed):
    values = np.asarray([row["delta"] for row in rows])
    return {"rows": len(rows), "effect": float(values.mean()), "paired_query_ci95": STATS["query_bootstrap"](rows, seed, 5000), "session_clustered_ci95": STATS["cluster_bootstrap"](rows, seed + 10000, 5000), "wins": int((values > 0).sum()), "losses": int((values < 0).sum()), "ties": int((values == 0).sum())}


def main() -> None:
    config = load_config("v21_qhds.json")
    run_dir = VERSION_ROOT / "artifacts" / "v21_qhds" / config["run_id"]
    rows, manifests, errors = [], [], []
    methods = {"qhds", "qhds_without_latent_aspects", "qhds_without_uot_soft_incidence", "flat_cosine", "qesa_association"}
    budgets = set(map(int, config["budgets"]))
    for seed in config["fixed_seeds"]:
        for fold in range(5):
            unit = run_dir / f"seed_{seed}" / f"fold_{fold}"
            path = unit / "v21_qhds_unit_manifest.json"
            if not path.is_file(): errors.append(f"missing seed={seed} fold={fold}"); continue
            manifest = json.loads(path.read_text())
            metrics_path = VERSION_ROOT / manifest["artifacts"]["metrics"]
            selections_path = VERSION_ROOT / manifest["artifacts"]["selections"]
            model_path = VERSION_ROOT / manifest["artifacts"]["model"]
            for key, item in (("metrics", metrics_path), ("selections", selections_path), ("model", model_path)):
                if sha256_file(item) != manifest["hashes"][key]: errors.append(f"{key} hash mismatch seed={seed} fold={fold}")
            if sha256_file(VERSION_ROOT / "configs" / "v21_qhds.json") != manifest["hashes"]["config"]: errors.append(f"config hash mismatch seed={seed} fold={fold}")
            with metrics_path.open(newline="") as handle: unit_rows = list(csv.DictReader(handle))
            keys = [(row["query_id"], row["method"], int(row["budget"])) for row in unit_rows]
            if len(keys) != len(set(keys)) or set(row["method"] for row in unit_rows) != methods or set(int(row["budget"]) for row in unit_rows) != budgets: errors.append(f"coverage mismatch seed={seed} fold={fold}")
            if manifest.get("calibration_read") or manifest.get("formal_test_read"): errors.append(f"locked read seed={seed} fold={fold}")
            rows.extend(unit_rows); manifests.append(manifest)
    if errors: raise RuntimeError("QHDS aggregation failed:\n- " + "\n- ".join(errors))
    expected = 5 * 1355 * len(methods) * len(budgets)
    keys = [(int(row["seed"]), row["query_id"], row["method"], int(row["budget"])) for row in rows]
    if len(rows) != expected or len(keys) != len(set(keys)): raise RuntimeError(f"exact row coverage failed {len(rows)} != {expected}")

    comparisons = {}
    comparators = ["flat_cosine", "qesa_association", "qhds_without_latent_aspects", "qhds_without_uot_soft_incidence"]
    number = 0
    for budget in sorted(budgets):
        comparisons[str(budget)] = {}
        for comparator in comparators:
            comparisons[str(budget)][comparator] = {}
            for subset in ("all", "K>=2"):
                effect_rows = paired(rows, comparator, budget, subset, "group_coverage")
                comparisons[str(budget)][comparator][subset] = summarize_effect(effect_rows, 20260714 + number)
                number += 1
    subset_summary = {}
    for label, predicate in (("K=1", lambda k:k==1), ("K=2", lambda k:k==2), ("K=3", lambda k:k==3), ("K=4", lambda k:k==4), ("K>=2", lambda k:k>=2)):
        subset_summary[label] = {}
        for budget in sorted(budgets):
            subset_summary[label][str(budget)] = {}
            for method in sorted(methods):
                selected = [row for row in rows if row["method"] == method and int(row["budget"]) == budget and predicate(int(row["group_count"]))]
                subset_summary[label][str(budget)][method] = {"rows": len(selected), "group_coverage": float(np.mean([float(row["group_coverage"]) for row in selected])), "all_groups_covered_rate": float(np.mean([float(row["all_groups_covered"]) for row in selected])), "both_alternatives_selected_rate": float(np.mean([float(row["both_alternatives_selected_rate"]) for row in selected])), "unique_groups_per_added_candidate": float(np.mean([float(row["unique_groups_covered_per_added_candidate"]) for row in selected]))}
    output_csv = VERSION_ROOT / "aggregate" / "v21_fixed_budget_query_metrics.csv"
    write_csv_new(output_csv, rows)
    result = {"schema_version": "v21-qhds-fixed-budget-result-1", "status": "completed_not_gate_c", "run_id": config["run_id"], "units": len(manifests), "metric_rows": len(rows), "comparisons": comparisons, "subset_summary": subset_summary, "training": {"loss_decreased_units": sum(m["training"]["epoch_loss_last"] < m["training"]["epoch_loss_first"] for m in manifests), "structural_scale_mean": float(np.mean([m["training"]["structural_scale"] for m in manifests])), "modular_scale_mean": float(np.mean([m["training"]["modular_scale"] for m in manifests]))}, "hashes": {"metrics": sha256_file(output_csv), "config": sha256_file(VERSION_ROOT / "configs" / "v21_qhds.json")}, "calibration_read": False, "formal_test_read": False}
    output_json = VERSION_ROOT / "aggregate" / "v21_qhds_fixed_budget_summary.json"
    write_json_new(output_json, result)
    report = "# Method 1 v21 QHDS Fixed-Budget Results\n\nThis phase is complete but is not Gate C; pairwise factor-graph, byte, Tile, latency, oracle, and Pareto comparisons remain pending.\n\n| B | Comparator | All effect | Clustered 95% CI | K>=2 effect | K>=2 clustered 95% CI |\n| ---: | --- | ---: | --- | ---: | --- |\n"
    for budget in sorted(budgets):
        for comparator in comparators:
            all_result, multi = comparisons[str(budget)][comparator]["all"], comparisons[str(budget)][comparator]["K>=2"]
            report += f"| {budget} | {comparator} | {all_result['effect']:+.6f} | [{all_result['session_clustered_ci95'][0]:+.6f}, {all_result['session_clustered_ci95'][1]:+.6f}] | {multi['effect']:+.6f} | [{multi['session_clustered_ci95'][0]:+.6f}, {multi['session_clustered_ci95'][1]:+.6f}] |\n"
    report += "\nEach evidence group is an OR set. K>=2 is the multi-group/compositional subset. All methods preserve the same protected top-5 and nested B policy. Calibration and formal test remained unread.\n"
    with (VERSION_ROOT / "reports" / "METHOD1_V21_QHDS_FIXED_BUDGET_RESULTS.md").open("x") as handle: handle.write(report)
    print(json.dumps({"status": result["status"], "units": len(manifests), "metric_rows": len(rows)}, sort_keys=True))


if __name__ == "__main__": main()
