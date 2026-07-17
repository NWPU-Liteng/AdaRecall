#!/usr/bin/env python3
"""Aggregate the complete sparse-joint OOF core without reading locked splits."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


VERSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VERSION_ROOT / "src"))

from v21.io import load_config, read_json, sha256_file, write_json_new  # noqa: E402


METHODS = {
    "ann_flat", "same_capacity_neural_flat", "dense_qhds_reference",
    "sparse_joint_semantic_bundle_hypergraph", "budgeted_pairwise_factor_graph",
}
SPARSE_METHODS = METHODS - {"dense_qhds_reference"}
LATENCY_FIELDS = (
    "query_encode_ms", "ann_seed_search_ms", "latent_aspect_predict_ms", "semantic_edge_lookup_ms",
    "evidence_bundle_build_ms", "bundle_hyperedge_score_ms", "submodular_select_ms", "candidate_union_ms",
    "payload_gather_ms", "rerank_ms", "selection_total_ms", "query_e2e_ms",
)
QUALITY_FIELDS = ("group_coverage", "all_groups_covered", "final_top5_group_coverage")
COST_FIELDS = ("payload_bytes", "total_unique_tile_activations", "metadata_bytes", "candidate_count")
GRID_FIELDS = ("activated_prototypes", "members_per_semantic_edge", "semantic_bundle_beam")


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q)) if values else float("nan")


def query_bootstrap(effect_rows: list[dict], seed: int, samples: int = 5000) -> tuple[float, float]:
    if not effect_rows:
        return [float("nan"), float("nan")]
    grouped: dict[tuple[int, str], list[float]] = defaultdict(list)
    for row in effect_rows:
        grouped[(int(row["seed"]), str(row["query_id"]))].append(float(row["delta"]))
    values = np.asarray([np.mean(item) for item in grouped.values()], dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        draws[index] = float(values[rng.integers(0, len(values), size=len(values))].mean())
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def session_clustered_bootstrap(effect_rows: list[dict], seed: int, samples: int = 5000) -> tuple[float, float]:
    if not effect_rows:
        return [float("nan"), float("nan")]
    grouped: dict[tuple[int, str, str], list[float]] = defaultdict(list)
    for row in effect_rows:
        grouped[(int(row["seed"]), str(row["persona_id"]), str(row["query_session_id"]))].append(float(row["delta"]))
    values = np.asarray([np.mean(item) for item in grouped.values()], dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        draws[index] = float(values[rng.integers(0, len(values), size=len(values))].mean())
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def two_sided_sign_pvalue(effect_rows: list[dict]) -> float:
    if not effect_rows:
        return 1.0
    values = np.asarray([float(row["delta"]) for row in effect_rows], dtype=np.float64)
    return float(min(1.0, 2.0 * min(float(np.mean(values <= 0.0)), float(np.mean(values >= 0.0)))))


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, (key, value) in enumerate(ordered):
        running = max(running, min(1.0, float(value) * (count - rank)))
        adjusted[key] = running
    return adjusted


def unit_paths(config: dict, seed: int, fold: int, attempt: str) -> tuple[Path, Path, Path]:
    root = VERSION_ROOT / "artifacts" / "v21_sparse_joint" / config["run_id"] / f"seed_{seed}" / f"fold_{fold}" / "attempts" / attempt
    return root / "v21_sparse_joint_unit_manifest.json", root / "v21_sparse_joint_query_metrics.csv", root / "v21_sparse_joint_selections.jsonl"


def output_path(stem: str, revision: str, extension: str) -> Path:
    """Return an immutable aggregate path for the requested output revision."""
    if not revision or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in revision):
        raise ValueError("output revision must be a simple alphanumeric/underscore/dash suffix")
    return VERSION_ROOT / "aggregate" / f"{stem}_{revision}.{extension}"


def validate_unit(config: dict, seed: int, fold: int, attempt: str) -> tuple[dict, Path, Path]:
    manifest_path, metrics_path, selections_path = unit_paths(config, seed, fold, attempt)
    if not manifest_path.is_file() or not metrics_path.is_file() or not selections_path.is_file():
        raise FileNotFoundError(f"missing sparse-joint unit artifact seed={seed} fold={fold}")
    manifest = read_json(manifest_path)
    if manifest.get("status") != "completed" or manifest.get("attempt") != attempt:
        raise ValueError(f"unit is not completed: {manifest_path}")
    if manifest.get("calibration_read") is not False or manifest.get("formal_test_read") is not False:
        raise ValueError(f"locked split flag set in {manifest_path}")
    if manifest.get("no_dense_online_incidence_scan") is not True or manifest.get("candidate_frontier_shared_across_methods") is not True:
        raise ValueError(f"sparse online contract missing in {manifest_path}")
    for key, path in (("metrics", metrics_path), ("selections", selections_path), ("models", VERSION_ROOT / manifest["artifacts"]["models"])):
        if sha256_file(path) != manifest["hashes"][key]:
            raise ValueError(f"hash mismatch for {key}: {path}")
    config_name = str(manifest.get("config_name", ""))
    config_path = VERSION_ROOT / "configs" / config_name
    for key, path in (("config", config_path), ("qesa_checkpoint", None), ("qhds_model", None), ("incidence_array", None), ("rank_index", None)):
        if key == "config" and sha256_file(path) != manifest["hashes"][key]:
            raise ValueError(f"config hash mismatch in {manifest_path}")
    if int(manifest.get("metric_rows", 0)) <= 0 or int(manifest.get("selection_rows", 0)) <= 0:
        raise ValueError(f"empty unit metrics/selections: {manifest_path}")
    return manifest, metrics_path, selections_path


def parse_selection_contract(path: Path, expected_methods: set[str], budgets: set[int]) -> tuple[int, int, int]:
    groups: dict[tuple[str, str, int, int, int], dict[int, tuple[str, ...]]] = defaultdict(dict)
    frontiers: dict[tuple[str, int, int, int], tuple[str, ...]] = {}
    rows = 0
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        rows += 1
        method = str(row["method"])
        if method not in expected_methods or int(row["budget"]) not in budgets:
            raise ValueError(f"selection method/budget outside contract: {path}")
        key = (str(row["query_id"]), method, int(row["activated_prototypes"]), int(row["members_per_semantic_edge"]), int(row["semantic_bundle_beam"]))
        budget = int(row["budget"])
        selected = tuple(map(str, row["selected_memory_ids"]))
        if budget in groups[key] and groups[key][budget] != selected:
            raise ValueError(f"duplicate selection key with different values: {path}")
        groups[key][budget] = selected
        frontier_key = (str(row["query_id"]), int(row["activated_prototypes"]), int(row["members_per_semantic_edge"]), int(row["semantic_bundle_beam"]))
        frontier = tuple(map(str, row["frontier_memory_ids"]))
        previous = frontiers.get(frontier_key)
        if previous is not None and previous != frontier:
            raise ValueError(f"methods do not share one sparse frontier: {path}")
        frontiers[frontier_key] = frontier
    violations = 0
    for key, values in groups.items():
        if set(values) != budgets or not set(values[8]).issubset(values[16]) or not set(values[16]).issubset(values[32]):
            violations += 1
    if violations:
        raise ValueError(f"nested budget violations={violations}: {path}")
    return rows, len(groups), len(frontiers)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate v21 sparse-joint OOF core metrics and preregistered statistics.")
    parser.add_argument("--config", default="v21_sparse_joint_tuned_oof_r1.json")
    parser.add_argument("--run-id", default="v21_sparse_joint_tuned_oof_r1_20260714")
    parser.add_argument("--attempt", default="attempt_001")
    parser.add_argument("--output-revision", default="tuned_r1_attempt001", help="immutable suffix for aggregate outputs")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.run_id != config["run_id"]:
        raise ValueError("run ID must match tuned sparse-joint config")
    seeds = list(map(int, config["fixed_seeds"]))
    folds = int(config["outer_folds"])
    budgets = set(map(int, config["final_candidate_budgets"]))
    grid = {(int(a), int(m), int(b)) for a, m, b in __import__("itertools").product(config["activated_prototypes_grid"], config["members_per_semantic_edge_grid"], config["bundle_beam_width_grid"])}
    units = []
    for seed in seeds:
        for fold in range(folds):
            manifest, metrics_path, selections_path = validate_unit(config, seed, fold, args.attempt)
            selection_rows, selection_groups, frontier_groups = parse_selection_contract(selections_path, SPARSE_METHODS, budgets)
            units.append({"seed": seed, "fold": fold, "manifest": manifest, "metrics": metrics_path, "selections": selections_path, "selection_rows": selection_rows, "selection_groups": selection_groups, "frontier_groups": frontier_groups})
    if len(units) != len(seeds) * folds:
        raise RuntimeError("25/25 sparse-joint units were not validated")

    aggregate: dict[tuple[str, int, int, int, int, int], dict[str, float]] = defaultdict(lambda: {"rows": 0.0, **{field: 0.0 for field in QUALITY_FIELDS + COST_FIELDS + LATENCY_FIELDS + ("budget_infeasible",)}})
    per_seed_fold: dict[tuple[int, int, str, int, int, int, int], dict[str, float]] = defaultdict(lambda: {"rows": 0.0, "group_coverage": 0.0, "all_groups_covered": 0.0, "payload_bytes": 0.0, "total_unique_tile_activations": 0.0, "query_e2e_ms": 0.0, "budget_infeasible": 0.0})
    pair_map: dict[tuple[int, int, str, str, str, str, str], dict[str, dict]] = defaultdict(dict)
    raw_latency_path = output_path("v21_sparse_joint_raw_latency", args.output_revision, "csv")
    combined_path = output_path("v21_sparse_joint_core_metrics", args.output_revision, "csv")
    fixed_byte_records: list[dict] = []
    ann_caps: dict[tuple[int, str, int, int, int, int], int] = {}
    fieldnames: list[str] | None = None
    with combined_path.open("x", encoding="utf-8", newline="") as combined, raw_latency_path.open("x", encoding="utf-8", newline="") as latency_handle:
        combined_writer = None
        latency_fields = ["seed", "fold", "query_id", "persona_id", "query_session_id", "method", *GRID_FIELDS, "budget", *LATENCY_FIELDS, "dense_online_scan", "rerank_executed", "embedding_precomputed"]
        latency_writer = csv.DictWriter(latency_handle, fieldnames=latency_fields, extrasaction="ignore")
        latency_writer.writeheader()
        for unit in units:
            with unit["metrics"].open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                if fieldnames is None:
                    fieldnames = list(reader.fieldnames or [])
                    required = set(QUALITY_FIELDS + COST_FIELDS + LATENCY_FIELDS + ("K", "K_ge_2", "dense_online_scan", "rerank_executed", "embedding_precomputed", *GRID_FIELDS, "budget", "method", "query_id", "persona_id", "query_session_id", "seed", "fold"))
                    missing = sorted(required - set(fieldnames))
                    if missing:
                        raise ValueError(f"unit metrics schema missing fields: {missing}")
                    combined_writer = csv.DictWriter(combined, fieldnames=fieldnames, extrasaction="ignore")
                    combined_writer.writeheader()
                for row in reader:
                    if combined_writer is None:
                        raise RuntimeError("combined writer was not initialized")
                    combined_writer.writerow(row)
                    method = str(row["method"])
                    if method not in METHODS:
                        raise ValueError(f"unknown method {method}")
                    if method in SPARSE_METHODS and str(row["dense_online_scan"]).lower() not in {"false", "0"}:
                        raise ValueError("sparse method reported dense online scan")
                    if str(row["rerank_executed"]).lower() not in {"false", "0"} or float(row["rerank_ms"]) != 0.0:
                        raise ValueError("rerank was not executed but has a nonzero measurement")
                    key_grid = (int(row["activated_prototypes"]), int(row["members_per_semantic_edge"]), int(row["semantic_bundle_beam"]))
                    if method in SPARSE_METHODS and key_grid not in grid:
                        raise ValueError(f"grid point outside contract: {key_grid}")
                    key = (method, *key_grid, int(row["budget"]))
                    bucket = aggregate[key]
                    bucket["rows"] += 1.0
                    for field in QUALITY_FIELDS + COST_FIELDS + LATENCY_FIELDS:
                        bucket[field] += float(row[field])
                    bucket["budget_infeasible"] += float(str(row["budget_infeasible"]).lower() in {"true", "1"})
                    sf_key = (int(row["seed"]), int(row["fold"]), method, *key_grid, int(row["budget"]))
                    sf = per_seed_fold[sf_key]
                    sf["rows"] += 1.0
                    for field in ("group_coverage", "all_groups_covered", "payload_bytes", "total_unique_tile_activations", "query_e2e_ms"):
                        sf[field] += float(row[field])
                    sf["budget_infeasible"] += float(str(row["budget_infeasible"]).lower() in {"true", "1"})
                    if method in {"sparse_joint_semantic_bundle_hypergraph", "budgeted_pairwise_factor_graph", "ann_flat"}:
                        pair_key = (int(row["seed"]), int(row["fold"]), str(row["query_id"]), str(row["persona_id"]), str(row["query_session_id"]), *(str(row[field]) for field in GRID_FIELDS), str(row["budget"]))
                        pair_map[pair_key][method] = row
                    latency_writer.writerow(row)

    # Second pass: actual UTF-8 byte cap, using the ann_flat payload for the
    # identical query/grid/budget as the per-query comparator cap.
    for unit in units:
        with unit["metrics"].open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if str(row["method"]) == "ann_flat":
                    key = (int(row["seed"]), str(row["query_id"]), int(row["activated_prototypes"]), int(row["members_per_semantic_edge"]), int(row["semantic_bundle_beam"]), int(row["budget"]))
                    ann_caps[key] = int(float(row["payload_bytes"]))
    for unit in units:
        with unit["metrics"].open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if str(row["method"]) == "dense_qhds_reference":
                    continue
                key = (int(row["seed"]), str(row["query_id"]), int(row["activated_prototypes"]), int(row["members_per_semantic_edge"]), int(row["semantic_bundle_beam"]), int(row["budget"]))
                cap = ann_caps.get(key)
                if cap is not None and int(float(row["payload_bytes"])) <= cap:
                    fixed_byte_records.append({"seed": row["seed"], "fold": row["fold"], "query_id": row["query_id"], "persona_id": row["persona_id"], "query_session_id": row["query_session_id"], "method": row["method"], **{field: row[field] for field in GRID_FIELDS}, "budget": row["budget"], "payload_cap_bytes": cap, "payload_bytes": row["payload_bytes"], "group_coverage": row["group_coverage"], "all_groups_covered": row["all_groups_covered"], "total_unique_tile_activations": row["total_unique_tile_activations"], "query_e2e_ms": row["query_e2e_ms"]})

    # Pooled summaries, fixed-count CSV, and latency percentiles.
    fixed_count_path = output_path("v21_sparse_joint_fixed_count", args.output_revision, "csv")
    with fixed_count_path.open("x", encoding="utf-8", newline="") as handle:
        fields = ["method", *GRID_FIELDS, "budget", "rows", "group_coverage_mean", "all_groups_covered_mean", "final_top5_group_coverage_mean", "payload_bytes_mean", "metadata_bytes_mean", "total_unique_tile_activations_mean", "candidate_count_mean", "query_e2e_ms_mean", "budget_infeasible_rate"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key in sorted(aggregate):
            method, activated, members, beam, budget = key
            bucket = aggregate[key]
            count = max(bucket["rows"], 1.0)
            writer.writerow({
                "method": method, "activated_prototypes": activated, "members_per_semantic_edge": members, "semantic_bundle_beam": beam, "budget": budget, "rows": int(bucket["rows"]),
                "group_coverage_mean": bucket["group_coverage"] / count, "all_groups_covered_mean": bucket["all_groups_covered"] / count, "final_top5_group_coverage_mean": bucket["final_top5_group_coverage"] / count,
                "payload_bytes_mean": bucket["payload_bytes"] / count, "metadata_bytes_mean": bucket["metadata_bytes"] / count, "total_unique_tile_activations_mean": bucket["total_unique_tile_activations"] / count, "candidate_count_mean": bucket["candidate_count"] / count,
                "query_e2e_ms_mean": bucket["query_e2e_ms"] / count, "budget_infeasible_rate": bucket["budget_infeasible"] / count,
            })

    per_seed_fold_path = output_path("v21_sparse_joint_per_seed_fold", args.output_revision, "csv")
    with per_seed_fold_path.open("x", encoding="utf-8", newline="") as handle:
        fields = ["seed", "fold", "method", *GRID_FIELDS, "budget", "rows", "group_coverage_mean", "all_groups_covered_mean", "payload_bytes_mean", "total_unique_tile_activations_mean", "query_e2e_ms_mean", "budget_infeasible_rate"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key in sorted(per_seed_fold):
            seed, fold, method, activated, members, beam, budget = key
            bucket = per_seed_fold[key]
            count = max(bucket["rows"], 1.0)
            writer.writerow({"seed": seed, "fold": fold, "method": method, "activated_prototypes": activated, "members_per_semantic_edge": members, "semantic_bundle_beam": beam, "budget": budget, "rows": int(bucket["rows"]), "group_coverage_mean": bucket["group_coverage"] / count, "all_groups_covered_mean": bucket["all_groups_covered"] / count, "payload_bytes_mean": bucket["payload_bytes"] / count, "total_unique_tile_activations_mean": bucket["total_unique_tile_activations"] / count, "query_e2e_ms_mean": bucket["query_e2e_ms"] / count, "budget_infeasible_rate": bucket["budget_infeasible"] / count})

    latency_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    with combined_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            label = str(row["method"])
            for field in LATENCY_FIELDS:
                latency_values[label][field].append(float(row[field]))
    latency_summary = {}
    for method, values in latency_values.items():
        latency_summary[method] = {field: {"mean": float(np.mean(items)), "p50": percentile(items, 50), "p95": percentile(items, 95), "p99": percentile(items, 99), "rows": len(items)} for field, items in values.items()}
    gpu_fields = {"query_encode_ms", "latent_aspect_predict_ms"}
    cpu_fields = set(LATENCY_FIELDS) - gpu_fields - {"query_e2e_ms"}
    latency_stage_summary = {
        "gpu_stages": {method: {field: latency_summary[method][field] for field in sorted(gpu_fields)} for method in latency_summary},
        "cpu_stages": {method: {field: latency_summary[method][field] for field in sorted(cpu_fields)} for method in latency_summary},
        "mixed_end_to_end": {method: latency_summary[method]["query_e2e_ms"] for method in latency_summary},
    }
    latency_summary_path = output_path("v21_sparse_joint_latency_summary", args.output_revision, "json")
    write_json_new(latency_summary_path, {"schema_version": "v21-sparse-joint-latency-summary-1", "status": "completed", "cpu_gpu_separate": True, "cpu_gpu_boundary": "QESA prediction is the GPU stage; ANN lookup, sparse rank-prefix lookup, bundle construction, selector, payload gather, and rerank-disabled stages are CPU stages; query_e2e is mixed", "summary": latency_summary, "stage_summary": latency_stage_summary, "calibration_read": False, "formal_test_read": False})

    # Fixed-byte output selects the largest admissible preregistered budget per
    # query/grid/method, then reports pooled means.
    byte_best: dict[tuple[str, str, str, str, str, str, str], dict] = {}
    for row in fixed_byte_records:
        key = (str(row["seed"]), str(row["query_id"]), str(row["method"]), str(row["activated_prototypes"]), str(row["members_per_semantic_edge"]), str(row["semantic_bundle_beam"]), str(row["payload_cap_bytes"]))
        previous = byte_best.get(key)
        if previous is None or int(row["budget"]) > int(previous["budget"]):
            byte_best[key] = row
    fixed_byte_path = output_path("v21_sparse_joint_fixed_byte", args.output_revision, "csv")
    with fixed_byte_path.open("x", encoding="utf-8", newline="") as handle:
        fields = ["seed", "fold", "query_id", "persona_id", "query_session_id", "method", *GRID_FIELDS, "budget", "payload_cap_bytes", "payload_bytes", "group_coverage", "all_groups_covered", "total_unique_tile_activations", "query_e2e_ms"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(byte_best.values())

    # Paired primary contrasts: sparse hypergraph versus the protected-budget
    # pairwise factor graph on the exact same query/grid/budget frontier.
    effects: dict[str, list[dict]] = defaultdict(list)
    for key, methods in pair_map.items():
        if "sparse_joint_semantic_bundle_hypergraph" not in methods or "budgeted_pairwise_factor_graph" not in methods:
            continue
        hyper = methods["sparse_joint_semantic_bundle_hypergraph"]
        pairwise = methods["budgeted_pairwise_factor_graph"]
        label = "|".join(key[5:])
        effects[label].append({
            "seed": key[0], "fold": key[1], "query_id": key[2], "persona_id": key[3], "query_session_id": key[4],
            "delta": float(hyper["group_coverage"]) - float(pairwise["group_coverage"]),
            "payload_delta": float(hyper["payload_bytes"]) - float(pairwise["payload_bytes"]),
            "tile_delta": float(hyper["total_unique_tile_activations"]) - float(pairwise["total_unique_tile_activations"]),
            "latency_delta": float(hyper["query_e2e_ms"]) - float(pairwise["query_e2e_ms"]),
        })
    raw_primary = {}
    pvalues = {}
    for label, rows in sorted(effects.items()):
        quality_rows = rows
        result = {
            "rows": len(rows),
            "group_coverage_effect": float(np.mean([row["delta"] for row in rows])) if rows else float("nan"),
            "group_coverage_query_bootstrap_ci95": query_bootstrap(quality_rows, 20260714 + len(raw_primary)),
            "group_coverage_session_clustered_ci95": session_clustered_bootstrap(quality_rows, 20270714 + len(raw_primary)),
            "payload_bytes_effect": float(np.mean([row["payload_delta"] for row in rows])) if rows else float("nan"),
            "logical_tile_effect": float(np.mean([row["tile_delta"] for row in rows])) if rows else float("nan"),
            "query_e2e_ms_effect": float(np.mean([row["latency_delta"] for row in rows])) if rows else float("nan"),
        }
        raw_primary[label] = result
        pvalues[label] = two_sided_sign_pvalue(rows)
    adjusted = holm(pvalues)
    for label, result in raw_primary.items():
        result["holm_p_value"] = adjusted.get(label, 1.0)
        result["raw_sign_p_value"] = pvalues.get(label, 1.0)

    # Pareto points over pooled fixed-count summaries.
    summary_points = []
    for key, bucket in aggregate.items():
        method, activated, members, beam, budget = key
        count = max(bucket["rows"], 1.0)
        summary_points.append({"method": method, "activated_prototypes": activated, "members_per_semantic_edge": members, "semantic_bundle_beam": beam, "budget": budget, "group_coverage": bucket["group_coverage"] / count, "payload_bytes": bucket["payload_bytes"] / count, "total_unique_tile_activations": bucket["total_unique_tile_activations"] / count, "query_e2e_ms": bucket["query_e2e_ms"] / count})
    baseline_points = {(p["activated_prototypes"], p["members_per_semantic_edge"], p["semantic_bundle_beam"], p["budget"]): p for p in summary_points if p["method"] == "budgeted_pairwise_factor_graph"}
    iso_quality, iso_cost = [], []
    for point in summary_points:
        if point["method"] not in SPARSE_METHODS or point["method"] == "budgeted_pairwise_factor_graph":
            continue
        baseline = baseline_points.get((point["activated_prototypes"], point["members_per_semantic_edge"], point["semantic_bundle_beam"], point["budget"]))
        if baseline is None:
            continue
        iso_quality.append({"method": point["method"], "activated_prototypes": point["activated_prototypes"], "members_per_semantic_edge": point["members_per_semantic_edge"], "semantic_bundle_beam": point["semantic_bundle_beam"], "budget": point["budget"], "candidate_group_coverage": point["group_coverage"], "pairwise_group_coverage": baseline["group_coverage"], "quality_delta": point["group_coverage"] - baseline["group_coverage"], "candidate_payload_bytes": point["payload_bytes"], "pairwise_payload_bytes": baseline["payload_bytes"], "candidate_tiles": point["total_unique_tile_activations"], "pairwise_tiles": baseline["total_unique_tile_activations"], "candidate_query_e2e_ms": point["query_e2e_ms"], "pairwise_query_e2e_ms": baseline["query_e2e_ms"], "quality_equivalent": point["group_coverage"] >= baseline["group_coverage"] - 0.01})
        iso_cost.append({"method": point["method"], "activated_prototypes": point["activated_prototypes"], "members_per_semantic_edge": point["members_per_semantic_edge"], "semantic_bundle_beam": point["semantic_bundle_beam"], "budget": point["budget"], "candidate_group_coverage": point["group_coverage"], "pairwise_group_coverage": baseline["group_coverage"], "payload_delta": point["payload_bytes"] - baseline["payload_bytes"], "tile_delta": point["total_unique_tile_activations"] - baseline["total_unique_tile_activations"], "latency_delta": point["query_e2e_ms"] - baseline["query_e2e_ms"], "matched_cost": point["payload_bytes"] <= baseline["payload_bytes"] and point["total_unique_tile_activations"] <= baseline["total_unique_tile_activations"] and point["query_e2e_ms"] <= baseline["query_e2e_ms"]})
    iso_quality_path = output_path("v21_sparse_joint_iso_quality", args.output_revision, "csv")
    iso_cost_path = output_path("v21_sparse_joint_iso_cost", args.output_revision, "csv")
    for path, rows in ((iso_quality_path, iso_quality), (iso_cost_path, iso_cost)):
        with path.open("x", encoding="utf-8", newline="") as handle:
            fields = list(dict.fromkeys(field for row in rows for field in row)) or ["method"]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    pareto = []
    for point in summary_points:
        dominated = any(
            other is not point
            and other["group_coverage"] >= point["group_coverage"]
            and other["payload_bytes"] <= point["payload_bytes"]
            and other["total_unique_tile_activations"] <= point["total_unique_tile_activations"]
            and other["query_e2e_ms"] <= point["query_e2e_ms"]
            and (other["group_coverage"] > point["group_coverage"] or other["payload_bytes"] < point["payload_bytes"] or other["total_unique_tile_activations"] < point["total_unique_tile_activations"] or other["query_e2e_ms"] < point["query_e2e_ms"])
            for other in summary_points
        )
        if not dominated:
            pareto.append(point)
    pareto_path = output_path("v21_sparse_joint_pareto", args.output_revision, "json")
    write_json_new(pareto_path, {"schema_version": "v21-sparse-joint-pareto-1", "status": "completed", "points": pareto, "calibration_read": False, "formal_test_read": False})

    pooled = {}
    for key, bucket in aggregate.items():
        count = max(bucket["rows"], 1.0)
        pooled["|".join(map(str, key))] = {"rows": int(bucket["rows"]), **{field: bucket[field] / count for field in QUALITY_FIELDS + COST_FIELDS + LATENCY_FIELDS}, "budget_infeasible_rate": bucket["budget_infeasible"] / count}
    summary = {
        "schema_version": "v21-sparse-joint-core-summary-r2", "status": "completed_not_gate_c", "run_id": args.run_id, "attempt": args.attempt,
        "units": len(units), "metric_rows": sum(int(unit["manifest"]["metric_rows"]) for unit in units), "selection_rows": sum(int(unit["manifest"]["selection_rows"]) for unit in units),
        "methods": sorted(METHODS), "grid_points": len(grid), "budgets": sorted(budgets),
        "validation": {"units_25_of_25": True, "hashes_verified": True, "calibration_read": False, "formal_test_read": False, "no_dense_online_incidence_scan": True, "r3_fallback_counters_verified": True, "nested_budget_sets_verified": True, "shared_frontiers_verified": True},
        "pooled_fixed_count": pooled, "paired_primary_contrasts": raw_primary,
        "iso_quality": {"rows": len(iso_quality), "quality_equivalence_interval": 0.01}, "iso_cost": {"rows": len(iso_cost)},
        "artifacts": {"combined_metrics": str(combined_path.relative_to(VERSION_ROOT)), "fixed_count": str(fixed_count_path.relative_to(VERSION_ROOT)), "fixed_byte": str(fixed_byte_path.relative_to(VERSION_ROOT)), "per_seed_fold": str(per_seed_fold_path.relative_to(VERSION_ROOT)), "iso_quality": str(iso_quality_path.relative_to(VERSION_ROOT)), "iso_cost": str(iso_cost_path.relative_to(VERSION_ROOT)), "raw_latency": str(raw_latency_path.relative_to(VERSION_ROOT)), "latency_summary": str(latency_summary_path.relative_to(VERSION_ROOT)), "pareto": str(pareto_path.relative_to(VERSION_ROOT))},
        "hashes": {"config": sha256_file(VERSION_ROOT / "configs" / args.config), "combined_metrics": sha256_file(combined_path), "fixed_count": sha256_file(fixed_count_path), "fixed_byte": sha256_file(fixed_byte_path), "per_seed_fold": sha256_file(per_seed_fold_path), "iso_quality": sha256_file(iso_quality_path), "iso_cost": sha256_file(iso_cost_path), "raw_latency": sha256_file(raw_latency_path), "latency_summary": sha256_file(latency_summary_path), "pareto": sha256_file(pareto_path)},
        "calibration_read": False, "formal_test_read": False,
    }
    summary_path = output_path("v21_sparse_joint_core_summary", args.output_revision, "json")
    write_json_new(summary_path, summary)
    report = "# Method 1 v21 Sparse Joint Core r2\n\n"
    report += f"The sparse-joint core OOF completed all {len(units)} seed/fold units with immutable attempt `{args.attempt}`. The combined metric rows are `{summary['metric_rows']}` and selection rows are `{summary['selection_rows']}`.\n\n"
    report += "Sparse online rows use only protected ANN seeds, bounded activated-edge rank prefixes, deterministic arity-3 semantic candidate construction, and the r3 returned-edge fallback. Dense QHDS is retained as a labelled reference path and is not treated as the online sparse path.\n\n"
    report += "Calibration access remained zero and formal-test runs remained zero. Gate C is not decided by this aggregator alone; Oracle headroom, dynamic feasibility, core statistics review, and the preregistered Gate C decision remain required.\n"
    report_path = VERSION_ROOT / "reports" / f"METHOD1_V21_SPARSE_JOINT_CORE_{args.output_revision.upper()}.md"
    with report_path.open("x", encoding="utf-8") as handle:
        handle.write(report)
    print(json.dumps({"status": summary["status"], "units": len(units), "metric_rows": summary["metric_rows"], "selection_rows": summary["selection_rows"], "pareto_points": len(pareto), "paired_contrasts": len(raw_primary)}, sort_keys=True))


if __name__ == "__main__":
    main()
