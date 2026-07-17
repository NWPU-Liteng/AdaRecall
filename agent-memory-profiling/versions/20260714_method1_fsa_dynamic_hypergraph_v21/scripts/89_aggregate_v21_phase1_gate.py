#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


VERSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VERSION_ROOT / "src"))

from v21.io import load_config, sha256_file, write_csv_new, write_json_new  # noqa: E402


def quantile_ci(values: np.ndarray) -> list[float]:
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def query_bootstrap(delta: np.ndarray, seed: int, samples: int = 5000) -> list[float]:
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 250):
        count = min(250, samples - start)
        positions = rng.integers(0, len(delta), size=(count, len(delta)))
        means[start : start + count] = delta[positions].mean(axis=1)
    return quantile_ci(means)


def clustered_bootstrap(rows: list[dict], seed: int, samples: int = 5000) -> list[float]:
    clusters: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        clusters[(str(row["persona_id"]), str(row["query_session_id"]))].append(float(row["delta"]))
    keys = sorted(clusters)
    sums = np.asarray([sum(clusters[key]) for key in keys], dtype=np.float64)
    counts = np.asarray([len(clusters[key]) for key in keys], dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for sample in range(samples):
        positions = rng.integers(0, len(keys), size=len(keys))
        means[sample] = sums[positions].sum() / counts[positions].sum()
    return quantile_ci(means)


def summarize(rows: list[dict], seed: int) -> dict:
    uot = np.asarray([float(row["uot_fsa_group_colocation"]) for row in rows])
    knn = np.asarray([float(row["cosine_knn_group_colocation"]) for row in rows])
    delta = uot - knn
    augmented = [{**row, "delta": float(value)} for row, value in zip(rows, delta, strict=True)]
    return {
        "queries": len(rows),
        "uot_mean": float(uot.mean()),
        "knn_mean": float(knn.mean()),
        "effect_uot_minus_knn": float(delta.mean()),
        "paired_query_ci95": query_bootstrap(delta, seed),
        "session_clustered_ci95": clustered_bootstrap(augmented, seed),
        "wins": int((delta > 0).sum()),
        "losses": int((delta < 0).sum()),
        "ties": int((delta == 0).sum()),
    }


def main() -> None:
    phase = load_config("v21_phase1_oof.json")
    run_id = str(phase["run_id"])
    run_dir = VERSION_ROOT / "artifacts" / "v21_phase1" / run_id
    all_rows: list[dict] = []
    units = []
    errors = []
    for seed in phase["fixed_seeds"]:
        for fold in range(int(phase["outer_folds"])):
            unit_dir = run_dir / f"seed_{seed}" / f"fold_{fold}"
            manifest_path = unit_dir / "v21_unit_manifest.json"
            if not manifest_path.is_file():
                errors.append(f"missing unit manifest seed={seed} fold={fold}")
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            diagnostics = manifest["outer_transport_diagnostics"]
            metrics_path = VERSION_ROOT / manifest["artifacts"]["query_metrics"]
            if sha256_file(metrics_path) != manifest["hashes"]["query_metrics"]:
                errors.append(f"query metrics hash mismatch seed={seed} fold={fold}")
            if not diagnostics["transport_converged"] or diagnostics["prototype_collapse_flag"]:
                errors.append(f"invalid final transport seed={seed} fold={fold}")
            with metrics_path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            if len(rows) != int(manifest["outer_query_count"]):
                errors.append(f"query count mismatch seed={seed} fold={fold}")
            all_rows.extend(rows)
            units.append(manifest)
    if errors:
        raise RuntimeError("v21 Phase 1 aggregation failed:\n- " + "\n- ".join(errors))

    by_seed: dict[int, list[dict]] = defaultdict(list)
    by_fold: dict[int, list[dict]] = defaultdict(list)
    for row in all_rows:
        by_seed[int(row["seed"])].append(row)
        by_fold[int(row["fold"])].append(row)
    if any(len(by_seed[seed]) != 1355 for seed in phase["fixed_seeds"]):
        raise RuntimeError(f"each fixed seed must have 1355 OOF rows: {Counter({seed: len(rows) for seed, rows in by_seed.items()})}")
    for seed, rows in by_seed.items():
        if len({row["query_id"] for row in rows}) != 1355:
            raise RuntimeError(f"seed {seed} does not contain exactly one OOF row per query")

    seed_summary = {str(seed): summarize(by_seed[seed], seed) for seed in phase["fixed_seeds"]}
    fold_summary = {str(fold): summarize(rows, 20260714 + fold) for fold, rows in sorted(by_fold.items())}
    pooled = summarize(all_rows, 20260714)
    all_transports_valid = all(
        unit["outer_transport_diagnostics"]["transport_converged"]
        and not unit["outer_transport_diagnostics"]["prototype_collapse_flag"]
        for unit in units
    )
    all_seed_clustered_positive = all(seed_summary[str(seed)]["session_clustered_ci95"][0] > 0 for seed in phase["fixed_seeds"])
    gate_passed = all_transports_valid and all_seed_clustered_positive
    recipe_counts = Counter(int(unit["selected_recipe"]["recipe_id"]) for unit in units)
    adapter_counts = Counter(
        (int(unit["selected_adapter"]["bottleneck"]), int(unit["selected_adapter"]["epochs"]), float(unit["selected_adapter"]["learning_rate"]))
        for unit in units
    )
    output_csv = VERSION_ROOT / "aggregate" / "v21_uot_fsa_oof_metrics.csv"
    write_csv_new(output_csv, all_rows)
    result = {
        "schema_version": "v21-phase1-gate-a-1",
        "status": "passed" if gate_passed else "not_passed",
        "run_id": run_id,
        "units": len(units),
        "fixed_seeds": phase["fixed_seeds"],
        "outer_folds": int(phase["outer_folds"]),
        "query_rows": len(all_rows),
        "all_transports_converged_without_collapse": all_transports_valid,
        "all_seed_session_clustered_ci_lower_bounds_positive": all_seed_clustered_positive,
        "seed_summary": seed_summary,
        "fold_summary": fold_summary,
        "pooled_oof": pooled,
        "seed_mean_effect": float(np.mean([seed_summary[str(seed)]["effect_uot_minus_knn"] for seed in phase["fixed_seeds"]])),
        "fold_mean_effect": float(np.mean([summary["effect_uot_minus_knn"] for summary in fold_summary.values()])),
        "selected_recipe_counts": {str(key): value for key, value in sorted(recipe_counts.items())},
        "selected_adapter_counts": {str(key): value for key, value in sorted(adapter_counts.items())},
        "metrics_csv": str(output_csv.relative_to(VERSION_ROOT)),
        "metrics_sha256": sha256_file(output_csv),
        "criterion": phase["gate_a"],
        "calibration_read": False,
        "formal_test_read": False,
    }
    output_json = VERSION_ROOT / "aggregate" / "v21_uot_fsa_oof_summary.json"
    write_json_new(output_json, result)
    report = "# Method 1 v21 UOT-FSA Results\n\n"
    report += f"Gate A status: `{result['status']}`. All 25 train-only outer OOF units completed with converged, non-collapsed final transport: `{all_transports_valid}`.\n\n"
    report += "| Seed | UOT-FSA | cosine-kNN | Difference | Paired-query 95% CI | Session-clustered 95% CI |\n"
    report += "| --- | ---: | ---: | ---: | --- | --- |\n"
    for seed in phase["fixed_seeds"]:
        row = seed_summary[str(seed)]
        report += (
            f"| {seed} | {row['uot_mean']:.6f} | {row['knn_mean']:.6f} | {row['effect_uot_minus_knn']:+.6f} | "
            f"[{row['paired_query_ci95'][0]:+.6f}, {row['paired_query_ci95'][1]:+.6f}] | "
            f"[{row['session_clustered_ci95'][0]:+.6f}, {row['session_clustered_ci95'][1]:+.6f}] |\n"
        )
    report += "\nThe numerical screen only excluded unstable recipes. Adapter and UOT choices were made independently inside each outer unit using train-only inner validation. The metric is evidence-group co-location, not final retrieval quality or a temporal/co-access claim. Calibration and formal test remain unread.\n\n"
    report += f"Selected recipe counts: `{result['selected_recipe_counts']}`.\n\n"
    report += "Gate A passing permits QESA development; it does not establish the later hypergraph quality-cost attribution claim.\n"
    report_path = VERSION_ROOT / "reports" / "METHOD1_V21_UOT_FSA_RESULTS.md"
    with report_path.open("x", encoding="utf-8") as handle:
        handle.write(report)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
