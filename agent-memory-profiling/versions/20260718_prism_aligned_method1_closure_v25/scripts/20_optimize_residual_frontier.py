from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

VERSION_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = VERSION_ROOT.parents[1]
V24_ROOT = PROJECT_ROOT / "versions" / "20260718_prism_aligned_method1_closure_v24"
sys.path.insert(0, str(VERSION_ROOT / "src"))
sys.path.insert(0, str(V24_ROOT / "src"))

from v24.dataset import load_canonical  # noqa: E402
from v24.diffusion import diffuse  # noqa: E402
from v24.frontier_selector import fuse_relevance, materialize_pool, select_frontier  # noqa: E402
from v24.metrics import quality_metrics  # noqa: E402
from v24.pipeline import Method1Runner  # noqa: E402
from v25.residual_frontier import (  # noqa: E402
    ResidualConfig,
    calibrated_edge_weight,
    select_residual_frontier,
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _partition(query_id: str) -> str:
    digest = hashlib.sha256(("v25-development-partition:" + query_id).encode("utf-8")).digest()
    return "tuning" if digest[0] % 2 == 0 else "validation"


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _bootstrap_ci(values: list[float], seed: int, samples: int = 5000) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        means[index] = float(rng.choice(array, size=len(array), replace=True).mean())
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _feature_row(runner: Method1Runner, query: Any, config: dict[str, Any]) -> dict[str, Any]:
    started = perf_counter()
    query_vector = runner.dataset.query_vector("development", query.query_id)
    eligible = runner.dataset.eligible_positions(query)
    all_dense = runner._dense_scores(query_vector, eligible)
    eligible_order = np.asarray(
        sorted(range(len(eligible)), key=lambda i: (-float(all_dense[i]), runner.memory_ids[int(eligible[i])])),
        dtype=np.int64,
    )
    frontier_c = int(config["budget"]["frontier_c"])
    seed_s = int(config["budget"]["seed_s"])
    dense_positions = eligible[eligible_order[:frontier_c]]
    seed_positions = dense_positions[:seed_s]
    seed_scores = all_dense[eligible_order[:seed_s]]
    seed_ids = [runner.memory_ids[int(position)] for position in seed_positions]

    semantic_cfg = config["semantic_hypergraph"]
    history_cfg = config["history_hypergraph"]
    active_semantic, _ = runner.semantic.active_edges(
        query_vector,
        seed_positions,
        seed_scores,
        top_edges=int(semantic_cfg["top_active_edges"]),
        tau=float(semantic_cfg["activation_temperature"]),
    )
    active_history, _ = runner.history.active_edges(
        query_vector,
        seed_ids,
        seed_scores,
        top_edges=int(history_cfg["top_active_edges"]),
        tau=float(history_cfg["activation_temperature"]),
    )
    visible_count = len(eligible)
    seed_set = set(int(value) for value in seed_positions)
    pool_edges: list[dict[str, Any]] = []
    semantic_edges: list[tuple[np.ndarray, float]] = []
    for edge in active_semantic:
        members = np.asarray(edge["members"], dtype=np.int64)
        overlap = len(seed_set.intersection(int(value) for value in members))
        weight = calibrated_edge_weight(
            float(edge["activation"]), edge_size=len(members), visible_count=visible_count, seed_overlap=overlap
        )
        pool_edges.append({**edge, "members": members})
        if weight > 0.0:
            semantic_edges.append((members, weight))

    history_edges: list[tuple[np.ndarray, float]] = []
    for edge in active_history:
        members = np.asarray(
            [runner.position_by_id[memory_id] for memory_id in edge["members"] if memory_id in runner.position_by_id],
            dtype=np.int64,
        )
        overlap = len(seed_set.intersection(int(value) for value in members))
        weight = calibrated_edge_weight(
            float(edge["activation"]), edge_size=len(members), visible_count=visible_count, seed_overlap=overlap
        )
        pool_edges.append({**edge, "members": members})
        if weight > 0.0:
            history_edges.append((members, weight))

    dense_by_position = {int(position): float(score) for position, score in zip(eligible, all_dense)}
    pool_positions = materialize_pool(
        eligible,
        dense_positions,
        pool_edges,
        cap=int(config["budget"]["expanded_pool_cap"]),
        dense_scores_by_position=dense_by_position,
    )
    if not len(pool_positions):
        pool_positions = dense_positions
    local_index = {int(position): index for index, position in enumerate(pool_positions)}
    local_ids = [runner.memory_ids[int(position)] for position in pool_positions]
    local_dense = runner._dense_scores(query_vector, pool_positions)
    x0 = np.zeros(len(pool_positions), dtype=np.float32)
    positive_seed = np.maximum(seed_scores - float(seed_scores.min(initial=0.0)), 0.0)
    if not np.any(positive_seed):
        positive_seed = np.ones_like(seed_scores)
    positive_seed /= max(float(positive_seed.sum()), 1e-12)
    for position, score in zip(seed_positions, positive_seed):
        if int(position) in local_index:
            x0[local_index[int(position)]] = float(score)

    def localize(edges: list[tuple[np.ndarray, float]]) -> tuple[list[np.ndarray], list[float]]:
        members_out: list[np.ndarray] = []
        weights_out: list[float] = []
        for members, weight in edges:
            local = np.asarray([local_index[int(value)] for value in members if int(value) in local_index], dtype=np.int64)
            if len(local) > 1:
                members_out.append(local)
                weights_out.append(weight)
        return members_out, weights_out

    semantic_members, semantic_weights = localize(semantic_edges)
    history_members, history_weights = localize(history_edges)
    diffusion_cfg = config["diffusion"]
    kwargs = {
        "alpha": float(diffusion_cfg["alpha_restart"]),
        "steps": int(diffusion_cfg["steps"]),
    }
    semantic_support = (
        diffuse(len(pool_positions), semantic_members, semantic_weights, x0, **kwargs)
        if semantic_members
        else np.zeros(len(pool_positions), dtype=np.float32)
    )
    history_support = (
        diffuse(len(pool_positions), history_members, history_weights, x0, **kwargs)
        if history_members
        else np.zeros(len(pool_positions), dtype=np.float32)
    )
    evaluation = runner.dataset.evaluations["development"][query.query_id]
    dense_ids = [runner.memory_ids[int(position)] for position in dense_positions]
    expanded_ids = local_ids

    # Reproduce the v24 all-position fusion on the calibrated components so the
    # loss decomposition separates expansion from frontier selection.
    fused, _ = fuse_relevance(local_ids, local_dense, semantic_support, history_support, beta_dense=0.7, history_weight=0.25)
    fused_local, _ = select_frontier(local_ids, fused, cap=frontier_c)
    fused_ids = [local_ids[index] for index in fused_local]
    return {
        "query_id": query.query_id,
        "partition": _partition(query.query_id),
        "memory_ids": local_ids,
        "dense_scores": local_dense,
        "semantic_support": semantic_support,
        "history_support": history_support,
        "evaluation": evaluation,
        "dense_ids": dense_ids,
        "expanded_ids": expanded_ids,
        "fused_ids": fused_ids,
        "active_semantic_edges": len(active_semantic),
        "calibrated_semantic_edges": len(semantic_edges),
        "active_history_edges": len(active_history),
        "calibrated_history_edges": len(history_edges),
        "elapsed_ms": float((perf_counter() - started) * 1000.0),
    }


def _evaluate(features: list[dict[str, Any]], candidate: ResidualConfig, cap: int) -> dict[str, Any]:
    rows = []
    for feature in features:
        selected, trace = select_residual_frontier(
            feature["memory_ids"],
            feature["dense_scores"],
            feature["semantic_support"],
            feature["history_support"],
            cap=cap,
            config=candidate,
        )
        ids = [feature["memory_ids"][index] for index in selected]
        ours = quality_metrics(ids, feature["evaluation"])
        dense = quality_metrics(feature["dense_ids"], feature["evaluation"])
        rows.append(
            {
                "query_id": feature["query_id"],
                "candidate_recall_delta": float(ours["candidate_recall"] - dense["candidate_recall"]),
                "group_coverage_delta": float(ours["evidence_group_coverage"] - dense["evidence_group_coverage"]),
                "replacements": int(sum(bool(row["residual_replacement"]) for row in trace)),
            }
        )
    recall_delta = _mean([row["candidate_recall_delta"] for row in rows])
    coverage_delta = _mean([row["group_coverage_delta"] for row in rows])
    return {
        "config": asdict(candidate),
        "query_count": len(rows),
        "candidate_recall_delta": recall_delta,
        "group_coverage_delta": coverage_delta,
        "strict_win_rate": _mean(
            [float(row["candidate_recall_delta"] > 0.0 or row["group_coverage_delta"] > 0.0) for row in rows]
        ),
        "loss_rate": _mean(
            [float(row["candidate_recall_delta"] < 0.0 or row["group_coverage_delta"] < 0.0) for row in rows]
        ),
        "mean_replacements": _mean([float(row["replacements"]) for row in rows]),
        "rows": rows,
    }


def _evaluate_fusion(
    features: list[dict[str, Any]], *, beta_dense: float, history_weight: float, cap: int
) -> dict[str, Any]:
    rows = []
    for feature in features:
        fused, _ = fuse_relevance(
            feature["memory_ids"],
            feature["dense_scores"],
            feature["semantic_support"],
            feature["history_support"],
            beta_dense=beta_dense,
            history_weight=history_weight,
        )
        selected, _ = select_frontier(feature["memory_ids"], fused, cap=cap)
        ids = [feature["memory_ids"][index] for index in selected]
        ours = quality_metrics(ids, feature["evaluation"])
        dense = quality_metrics(feature["dense_ids"], feature["evaluation"])
        dense_set = set(feature["dense_ids"])
        rows.append(
            {
                "query_id": feature["query_id"],
                "candidate_recall_delta": float(ours["candidate_recall"] - dense["candidate_recall"]),
                "group_coverage_delta": float(ours["evidence_group_coverage"] - dense["evidence_group_coverage"]),
                "replacements": int(len(set(ids) - dense_set)),
            }
        )
    return {
        "strategy": "degree_calibrated_fusion",
        "config": {"beta_dense": float(beta_dense), "history_weight": float(history_weight)},
        "query_count": len(rows),
        "candidate_recall_delta": _mean([row["candidate_recall_delta"] for row in rows]),
        "group_coverage_delta": _mean([row["group_coverage_delta"] for row in rows]),
        "strict_win_rate": _mean(
            [float(row["candidate_recall_delta"] > 0.0 or row["group_coverage_delta"] > 0.0) for row in rows]
        ),
        "loss_rate": _mean(
            [float(row["candidate_recall_delta"] < 0.0 or row["group_coverage_delta"] < 0.0) for row in rows]
        ),
        "mean_replacements": _mean([float(row["replacements"]) for row in rows]),
        "rows": rows,
    }


def _candidate_configs(search: dict[str, list[Any]]) -> list[ResidualConfig]:
    keys = [
        "max_replacements",
        "protected_prefix",
        "graph_margin",
        "max_dense_drop",
        "history_weight",
        "minimum_graph_rank",
    ]
    seen: set[tuple[Any, ...]] = set()
    output = []
    for values in itertools.product(*(search[key] for key in keys)):
        if int(values[0]) == 0:
            values = (0, 32, 1.0, 0.0, 0.0, 1.0)
        if values in seen:
            continue
        seen.add(values)
        output.append(ResidualConfig(**dict(zip(keys, values))))
    return output


def _decomposition(features: list[dict[str, Any]]) -> dict[str, Any]:
    methods = {"dense_c": [], "expanded_pool": [], "calibrated_fused_c": []}
    for feature in features:
        evaluation = feature["evaluation"]
        for name, key in (("dense_c", "dense_ids"), ("expanded_pool", "expanded_ids"), ("calibrated_fused_c", "fused_ids")):
            metrics = quality_metrics(feature[key], evaluation)
            methods[name].append(metrics)
    return {
        name: {
            "candidate_recall": _mean([float(row["candidate_recall"]) for row in rows]),
            "evidence_group_coverage": _mean([float(row["evidence_group_coverage"]) for row in rows]),
        }
        for name, rows in methods.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(VERSION_ROOT / "configs" / "optimization.json"))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    config = _read_json(Path(args.config))
    if config["scientific_boundary"].get("test_must_not_be_read") is not True:
        raise RuntimeError("v25 requires an explicit test_must_not_be_read boundary")

    dataset = load_canonical(V24_ROOT / "inputs", split="development")
    source_index = V24_ROOT / "runs" / config["source_index_run_id"] / "index"
    runner_config = _read_json(V24_ROOT / "configs" / "closure_main.json")
    runner_config["reranker"] = {"backend": "dense_reference"}
    runner = Method1Runner(dataset, source_index, runner_config, reranker_backend="dense_reference")
    queries = dataset.queries["development"][: args.limit or None]
    features = [_feature_row(runner, query, config) for query in queries]
    tuning = [row for row in features if row["partition"] == "tuning"]
    validation = [row for row in features if row["partition"] == "validation"]

    candidates = _candidate_configs(config["search"])
    frontier_c = int(config["budget"]["frontier_c"])
    tuning_results = [_evaluate(tuning, candidate, frontier_c) for candidate in candidates]
    for beta_dense, history_weight in itertools.product(
        config["fusion_search"]["beta_dense"], config["fusion_search"]["history_weight"]
    ):
        tuning_results.append(
            _evaluate_fusion(
                tuning,
                beta_dense=float(beta_dense),
                history_weight=float(history_weight),
                cap=frontier_c,
            )
        )
    tuning_results.sort(
        key=lambda row: (
            -float(row["candidate_recall_delta"] + row["group_coverage_delta"]),
            -float(row["candidate_recall_delta"]),
            float(row["loss_rate"]),
            float(row["mean_replacements"]),
            json.dumps(row["config"], sort_keys=True),
        )
    )
    best_tuning = tuning_results[0]
    selected_strategy = str(best_tuning.get("strategy", "degree_calibrated_residual"))
    if selected_strategy == "degree_calibrated_fusion":
        validation_result = _evaluate_fusion(validation, cap=frontier_c, **best_tuning["config"])
    else:
        validation_result = _evaluate(validation, ResidualConfig(**best_tuning["config"]), frontier_c)
        validation_result["strategy"] = selected_strategy
    validation_rows = validation_result.pop("rows")
    recall_values = [float(row["candidate_recall_delta"]) for row in validation_rows]
    coverage_values = [float(row["group_coverage_delta"]) for row in validation_rows]
    promotion_cfg = config["promotion"]
    strict_positive = bool(
        validation_result["candidate_recall_delta"] > 0.0 or validation_result["group_coverage_delta"] > 0.0
    )
    promoted = bool(
        validation_result["candidate_recall_delta"] >= float(promotion_cfg["validation_candidate_recall_delta_min"])
        and validation_result["group_coverage_delta"] >= float(promotion_cfg["validation_group_coverage_delta_min"])
        and validation_result["mean_replacements"] <= float(promotion_cfg["max_mean_replacements"])
        and (strict_positive or not promotion_cfg["require_strict_positive_signal"])
    )

    report = {
        "status": "PROMOTED_TO_DEVELOPMENT_RERANKER" if promoted else "NOT_PROMOTED",
        "test_read": False,
        "source_test_reused": False,
        "query_count": len(features),
        "partition_counts": {"tuning": len(tuning), "validation": len(validation)},
        "candidate_config_count": len(candidates),
        "fusion_config_count": len(config["fusion_search"]["beta_dense"]) * len(config["fusion_search"]["history_weight"]),
        "loss_decomposition": {
            "tuning": _decomposition(tuning),
            "validation": _decomposition(validation),
        },
        "selected_strategy": selected_strategy,
        "selected_config": best_tuning["config"],
        "tuning_result": {key: value for key, value in best_tuning.items() if key != "rows"},
        "validation_result": validation_result,
        "validation_ci": {
            "candidate_recall_delta_95": _bootstrap_ci(recall_values, int(config["random_seed"])),
            "group_coverage_delta_95": _bootstrap_ci(coverage_values, int(config["random_seed"]) + 1),
        },
        "activation_diagnostics": {
            "active_semantic_edges_mean": _mean([float(row["active_semantic_edges"]) for row in features]),
            "calibrated_semantic_edges_mean": _mean([float(row["calibrated_semantic_edges"]) for row in features]),
            "active_history_edges_mean": _mean([float(row["active_history_edges"]) for row in features]),
            "calibrated_history_edges_mean": _mean([float(row["calibrated_history_edges"]) for row in features]),
        },
        "feature_elapsed_ms_mean": _mean([float(row["elapsed_ms"]) for row in features]),
        "promotion_passed": promoted,
    }
    output = VERSION_ROOT / "aggregate" / "residual_frontier_optimization.json"
    _write_json(output, report)
    _write_json(
        VERSION_ROOT / "artifacts" / "selected_development_config.json",
        {"strategy": selected_strategy, "config": best_tuning["config"], "promotion_passed": promoted},
    )
    print(json.dumps({"status": report["status"], "selected_config": report["selected_config"], "validation": validation_result}, sort_keys=True))


if __name__ == "__main__":
    main()
