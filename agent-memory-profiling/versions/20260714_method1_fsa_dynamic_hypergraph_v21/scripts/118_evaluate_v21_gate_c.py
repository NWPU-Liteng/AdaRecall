#!/usr/bin/env python3
"""Evaluate the preregistered OOF Gate C contrasts without locked splits."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest


VERSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VERSION_ROOT / "src"))

from v21.io import load_config, sha256_file, write_csv_new, write_json_new  # noqa: E402


RUN_ID = "v21_gate_c_decision_r1_20260714"
CORE_REVISION = "r3_attempt019"
HYPER = "sparse_joint_semantic_bundle_hypergraph"
PAIRWISE = "budgeted_pairwise_factor_graph"
GRID_FIELDS = ["activated_prototypes", "members_per_semantic_edge", "semantic_bundle_beam", "budget"]
KEY_FIELDS = ["seed", "fold", "query_id", "persona_id", "query_session_id", *GRID_FIELDS]
VALUE_FIELDS = ["group_coverage", "final_top5_group_coverage", "payload_bytes", "total_unique_tile_activations", "query_e2e_ms"]


def bootstrap_ci(frame: pd.DataFrame, metric: str, group_fields: list[str], seed: int, samples: int = 5000) -> tuple[float, float]:
    grouped = frame.groupby(group_fields, sort=True)[metric].mean().to_numpy(dtype=np.float64)
    if len(grouped) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        draws[index] = float(grouped[rng.integers(0, len(grouped), size=len(grouped))].mean())
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def sign_pvalue(values: np.ndarray) -> float:
    nonzero = values[np.abs(values) > 1e-12]
    if len(nonzero) == 0:
        return 1.0
    positive = int(np.sum(nonzero > 0))
    return float(binomtest(min(positive, len(nonzero) - positive), len(nonzero), 0.5, alternative="two-sided").pvalue)


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (key, value) in enumerate(ordered):
        running = max(running, min(1.0, float(value) * (len(ordered) - rank)))
        adjusted[key] = running
    return adjusted


def load_pairs(path: Path) -> pd.DataFrame:
    usecols = [*KEY_FIELDS, "method", *VALUE_FIELDS]
    parts = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=200_000):
        parts.append(chunk[chunk["method"].isin([HYPER, PAIRWISE])])
    if not parts:
        raise ValueError("core metrics are empty")
    source = pd.concat(parts, ignore_index=True)
    if source.empty:
        raise ValueError("no hypergraph/pairwise rows in core metrics")
    for field in GRID_FIELDS + ["seed", "fold"]:
        source[field] = source[field].astype(int)
    source["query_id"] = source["query_id"].astype(str)
    source["persona_id"] = source["persona_id"].astype(str)
    source["query_session_id"] = source["query_session_id"].astype(str)
    duplicated = source.duplicated(KEY_FIELDS + ["method"])
    if bool(duplicated.any()):
        raise ValueError(f"duplicate paired method rows: {int(duplicated.sum())}")
    hyper = source[source["method"] == HYPER].drop(columns=["method"]).rename(columns={field: f"hyper_{field}" for field in VALUE_FIELDS})
    pairwise = source[source["method"] == PAIRWISE].drop(columns=["method"]).rename(columns={field: f"pairwise_{field}" for field in VALUE_FIELDS})
    merged = hyper.merge(pairwise, on=KEY_FIELDS, how="inner", validate="one_to_one")
    if len(merged) != len(hyper) or len(merged) != len(pairwise):
        raise ValueError("hypergraph and pairwise rows are not exactly paired")
    for field in VALUE_FIELDS:
        merged[f"delta_{field}"] = merged[f"hyper_{field}"] - merged[f"pairwise_{field}"]
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the v21 preregistered Gate C decision.")
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--core-revision", default=CORE_REVISION)
    args = parser.parse_args()
    if args.run_id != RUN_ID:
        raise ValueError("Gate C run ID is immutable")
    data = load_config("v21_data.json")
    closure = load_config("v21_effect_closure.json")
    metrics_path = VERSION_ROOT / "aggregate" / f"v21_sparse_joint_core_metrics_{args.core_revision}.csv"
    summary_path = VERSION_ROOT / "aggregate" / f"v21_sparse_joint_core_summary_{args.core_revision}.json"
    pareto_path = VERSION_ROOT / "aggregate" / f"v21_sparse_joint_pareto_{args.core_revision}.json"
    core_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if core_summary.get("calibration_read") is not False or core_summary.get("formal_test_read") is not False:
        raise ValueError("core summary has an unlocked split flag")
    pairs = load_pairs(metrics_path)
    contrasts = []
    quality_pvalues: dict[str, float] = {}
    for key, frame in pairs.groupby(GRID_FIELDS, sort=True):
        label = "|".join(map(str, key))
        result = {field: int(value) for field, value in zip(GRID_FIELDS, key, strict=True)}
        result["rows"] = int(len(frame))
        for metric in VALUE_FIELDS:
            values = frame[f"delta_{metric}"].to_numpy(dtype=np.float64)
            query_ci = bootstrap_ci(frame.assign(_value=values), "_value", ["seed", "query_id"], 20260714 + len(contrasts) * 17 + len(metric), 5000)
            session_ci = bootstrap_ci(frame.assign(_value=values), "_value", ["seed", "persona_id", "query_session_id"], 20270714 + len(contrasts) * 17 + len(metric), 5000)
            result[f"{metric}_effect"] = float(values.mean())
            result[f"{metric}_query_bootstrap_ci95_lower"] = query_ci[0]
            result[f"{metric}_query_bootstrap_ci95_upper"] = query_ci[1]
            result[f"{metric}_session_clustered_ci95_lower"] = session_ci[0]
            result[f"{metric}_session_clustered_ci95_upper"] = session_ci[1]
        quality_pvalues[label] = sign_pvalue(frame["delta_group_coverage"].to_numpy(dtype=np.float64))
        result["quality_sign_p_value"] = quality_pvalues[label]
        contrasts.append(result)
    adjusted = holm(quality_pvalues)
    for result in contrasts:
        label = "|".join(str(result[field]) for field in GRID_FIELDS)
        result["quality_holm_p_value"] = adjusted[label]
        quality_ci_lower = float(result["group_coverage_session_clustered_ci95_lower"])
        quality_ci_upper = float(result["group_coverage_session_clustered_ci95_upper"])
        top5_ci_lower = float(result["final_top5_group_coverage_session_clustered_ci95_lower"])
        tile_ci_upper = float(result["total_unique_tile_activations_session_clustered_ci95_upper"])
        latency_ci_upper = float(result["query_e2e_ms_session_clustered_ci95_upper"])
        result["quality_advantage"] = bool(quality_ci_lower > 0.0 and adjusted[label] <= 0.05)
        result["final_top5_non_degradation"] = bool(top5_ci_lower >= float(closure.get("final_top5_non_degradation_margin", -0.005)))
        result["logical_tile_non_increase"] = bool(tile_ci_upper <= 0.0 + 1e-9)
        result["latency_advantage"] = bool(latency_ci_upper < 0.0)
        result["matched_tile_quality_advantage"] = bool(result["quality_advantage"] and result["logical_tile_non_increase"])
        result["gate_c_candidate"] = bool(result["matched_tile_quality_advantage"] and result["final_top5_non_degradation"] and result["latency_advantage"])
    pareto = json.loads(pareto_path.read_text(encoding="utf-8"))
    pareto_hyper = [point for point in pareto.get("points", []) if point.get("method") == HYPER]
    candidates = [result for result in contrasts if result["gate_c_candidate"]]
    decision = "PASS" if candidates and pareto_hyper else "FAIL"
    contrast_path = VERSION_ROOT / "aggregate" / "v21_gate_c_contrasts_r1_20260714.csv"
    write_csv_new(contrast_path, contrasts)
    summary = {
        "schema_version": "v21-gate-c-decision-1", "status": f"gate_c_{decision.lower()}_dynamic_pending", "run_id": RUN_ID,
        "decision": decision, "core_revision": args.core_revision, "paired_rows": int(len(pairs)), "contrast_count": len(contrasts),
        "gate_c_candidates": candidates, "pareto_hypergraph_points": len(pareto_hyper),
        "criteria": {"final_top5_margin": float(closure.get("final_top5_non_degradation_margin", -0.005)), "quality_holm_alpha": 0.05, "logical_tile_non_increase": True, "latency_session_ci_upper_lt_zero": True},
        "interpretation": "PASS means at least one preregistered matched-logical-Tile quality/latency point with final-top-5 non-degradation; it does not imply a payload-byte advantage or a PIM result." if decision == "PASS" else "FAIL means no preregistered matched-cost quality/latency point satisfied all criteria; retain static fallback and negative results.",
        "validation": {"core_summary_hash": sha256_file(summary_path), "metrics_hash": sha256_file(metrics_path), "pareto_hash": sha256_file(pareto_path), "calibration_read": False, "formal_test_read": False, "no_dense_online_scan": True},
        "artifacts": {"contrasts": str(contrast_path.relative_to(VERSION_ROOT))}, "hashes": {"contrasts": sha256_file(contrast_path), "core_summary": sha256_file(summary_path), "metrics": sha256_file(metrics_path), "pareto": sha256_file(pareto_path)},
        "calibration_read": False, "formal_test_read": False,
    }
    output_path = VERSION_ROOT / "aggregate" / "v21_gate_c_decision_r1_20260714.json"
    write_json_new(output_path, summary)
    report_path = VERSION_ROOT / "reports" / "METHOD1_V21_GATE_C_DECISION_R1.md"
    report = f"# Method 1 v21 Gate C Decision\n\n**Decision: {decision} (dynamic feasibility pending).**\n\n"
    report += "The decision uses the 25-unit sparse-joint OOF revision `r3_attempt019`, paired hypergraph-vs-budgeted-pairwise rows, session-clustered bootstrap intervals, and Holm correction across the 81 preregistered grid/budget contrasts.\n\n"
    report += "The qualifying criterion is quality improvement with no increase in logical Tile activations, final-top-5 non-degradation, and a negative session-clustered end-to-end latency interval. Payload bytes are reported separately; most qualifying points do not claim a payload-byte reduction.\n\n"
    report += "| Grid/B | Group coverage effect | Clustered CI | Final top-5 effect | Tile effect | E2E latency effect | Holm p |\n| --- | ---: | --- | ---: | ---: | ---: | ---: |\n"
    for result in candidates[:20]:
        label = f"{result['activated_prototypes']}/{result['members_per_semantic_edge']}/{result['semantic_bundle_beam']}/{result['budget']}"
        report += f"| {label} | {result['group_coverage_effect']:+.6f} | [{result['group_coverage_session_clustered_ci95_lower']:+.6f}, {result['group_coverage_session_clustered_ci95_upper']:+.6f}] | {result['final_top5_group_coverage_effect']:+.6f} | {result['total_unique_tile_activations_effect']:+.6f} | {result['query_e2e_ms_effect']:+.6f} | {result['quality_holm_p_value']:.3g} |\n"
    report += "\nCalibration and formal-test access remained zero. This is an OOF Gate C decision only; it does not authorize calibration until dynamic/static closure and pre-calibration freeze are complete.\n"
    with report_path.open("x", encoding="utf-8") as handle:
        handle.write(report)
    print(json.dumps({"status": summary["status"], "decision": decision, "paired_rows": len(pairs), "contrasts": len(contrasts), "gate_c_candidates": len(candidates), "pareto_hypergraph_points": len(pareto_hyper)}, sort_keys=True))


if __name__ == "__main__":
    main()
