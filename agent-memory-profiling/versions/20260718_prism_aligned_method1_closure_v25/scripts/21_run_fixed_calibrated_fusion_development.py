from __future__ import annotations

import argparse
import importlib.util
import json
import sys
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
from v24.frontier_selector import fuse_relevance, select_frontier  # noqa: E402
from v24.io import iter_jsonl  # noqa: E402
from v24.metrics import quality_metrics  # noqa: E402
from v24.pipeline import Method1Runner  # noqa: E402


def _load_optimizer_module() -> Any:
    path = VERSION_ROOT / "scripts" / "20_optimize_residual_frontier.py"
    spec = importlib.util.spec_from_file_location("v25_optimizer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load optimizer module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _bootstrap_ci(values: list[float], seed: int, samples: int = 5000) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    means = [float(rng.choice(array, size=len(array), replace=True).mean()) for _ in range(samples)]
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--beta-dense", type=float, default=0.7)
    parser.add_argument("--history-weight", type=float, default=0.25)
    args = parser.parse_args()
    optimizer = _load_optimizer_module()
    optimization = optimizer._read_json(VERSION_ROOT / "configs" / "optimization.json")
    if optimization["scientific_boundary"].get("test_must_not_be_read") is not True:
        raise RuntimeError("v25 development reranking cannot run without the test boundary")

    dataset = load_canonical(V24_ROOT / "inputs", split="development")
    v24_config = optimizer._read_json(V24_ROOT / "configs" / "closure_main.json")
    v24_config["reranker"] = dict(v24_config["reranker"])
    v24_config["reranker"]["model_dir"] = str(PROJECT_ROOT / v24_config["model"]["reranker_model_dir"])
    v24_config["reranker"]["device"] = args.device
    source_index = V24_ROOT / "runs" / optimization["source_index_run_id"] / "index"
    runner = Method1Runner(dataset, source_index, v24_config, reranker_backend="qwen3")
    queries = dataset.queries["development"][: args.limit or None]
    output_dir = VERSION_ROOT / "runs" / "v25_fixed_calibrated_fusion_development"
    output_dir.mkdir(parents=True, exist_ok=True)
    row_path = output_dir / "query_results.jsonl"
    existing = {row["query_id"]: row for row in iter_jsonl(row_path)} if row_path.exists() else {}

    for query in queries:
        if query.query_id in existing:
            continue
        started = perf_counter()
        feature = optimizer._feature_row(runner, query, optimization)
        fused, _ = fuse_relevance(
            feature["memory_ids"],
            feature["dense_scores"],
            feature["semantic_support"],
            feature["history_support"],
            beta_dense=float(args.beta_dense),
            history_weight=float(args.history_weight),
        )
        selected, frontier_trace = select_frontier(
            feature["memory_ids"], fused, cap=int(optimization["budget"]["frontier_c"])
        )
        positions = np.asarray([runner.position_by_id[feature["memory_ids"][index]] for index in selected], dtype=np.int64)
        dense_scores = np.asarray([feature["dense_scores"][index] for index in selected], dtype=np.float32)
        final_ids, rerank_ms, input_tokens, reranked_rows = runner._rerank(query, positions, dense_scores, final_k=5)
        frontier_ids = [feature["memory_ids"][index] for index in selected]
        evaluation = feature["evaluation"]
        frontier_metrics = quality_metrics(frontier_ids, evaluation)
        final_metrics = quality_metrics(final_ids, evaluation, k=5)
        eligible = set(int(value) for value in dataset.eligible_positions(query))
        row = {
            "query_id": query.query_id,
            "namespace_id": query.namespace_id,
            "method": "v25_fixed_calibrated_fusion_c",
            "config": {"beta_dense": float(args.beta_dense), "history_weight": float(args.history_weight)},
            "frontier_ids": frontier_ids,
            "reranked_ids": final_ids,
            "frontier_trace": frontier_trace,
            "reranked_candidates": reranked_rows,
            "metrics": {
                "candidate_recall_frontier": frontier_metrics["candidate_recall"],
                "evidence_group_coverage_frontier": frontier_metrics["evidence_group_coverage"],
                "candidate_recall_final": final_metrics["candidate_recall"],
                "evidence_group_coverage_final": final_metrics["evidence_group_coverage"],
                "precision_at_k": final_metrics["precision_at_k"],
                "recall_at_k": final_metrics["recall_at_k"],
                "ndcg_at_10": final_metrics["ndcg_at_10"],
                "mrr_at_10": final_metrics["mrr_at_10"],
                "frontier_count": len(frontier_ids),
                "cross_namespace_violation": any(
                    dataset.memories[int(position)].namespace_id != query.namespace_id for position in positions
                ),
                "future_memory_violation": any(int(position) not in eligible for position in positions),
                "duplicate_frontier_ids": len(frontier_ids) != len(set(frontier_ids)),
            },
            "feature_ms": float(feature["elapsed_ms"]),
            "rerank_ms": float(rerank_ms),
            "total_wall_ms": float((perf_counter() - started) * 1000.0),
            "reranker_input_tokens": int(input_tokens),
            "test_read": False,
        }
        with row_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
        existing[query.query_id] = row
        if len(existing) % 25 == 0:
            print(json.dumps({"completed": len(existing), "total": len(queries)}, sort_keys=True), flush=True)

    ours = [existing[query.query_id] for query in queries if query.query_id in existing]
    baseline_path = V24_ROOT / "runs" / "v24_development_20260718" / "query_results.jsonl"
    baseline_rows = list(iter_jsonl(baseline_path))
    baseline = {(row["query_id"], row["method"]): row for row in baseline_rows}
    metric_keys = [
        "candidate_recall_frontier",
        "evidence_group_coverage_frontier",
        "candidate_recall_final",
        "evidence_group_coverage_final",
        "precision_at_k",
        "ndcg_at_10",
    ]
    deltas: dict[str, Any] = {}
    for key in metric_keys:
        values = [
            float(row["metrics"][key]) - float(baseline[(row["query_id"], "dense_c")]["metrics"][key])
            for row in ours
        ]
        deltas[key] = {
            "mean": _mean(values),
            "ci_95": _bootstrap_ci(values, 20260718 + metric_keys.index(key)),
            "win_rate": _mean([float(value > 0.0) for value in values]),
            "loss_rate": _mean([float(value < 0.0) for value in values]),
        }
    ours_wall = _mean([float(row["total_wall_ms"]) for row in ours])
    full_wall = _mean(
        [float(baseline[(row["query_id"], "full_n")]["semantic_selection_ms"]) for row in ours]
    )
    safety_passed = not any(
        row["metrics"][key]
        for row in ours
        for key in ("cross_namespace_violation", "future_memory_violation", "duplicate_frontier_ids")
    )
    complete = len(ours) == len(queries)
    development_signal = bool(
        complete
        and safety_passed
        and deltas["candidate_recall_frontier"]["mean"] >= 0.0
        and deltas["ndcg_at_10"]["mean"] >= -0.005
        and any(deltas[key]["mean"] > 0.0 for key in metric_keys)
        and ours_wall < full_wall
    )
    summary = {
        "status": "DEVELOPMENT_SIGNAL" if development_signal else "DEVELOPMENT_SIGNAL_FAILED",
        "scientific_status": "development_only_already_seen_not_independent",
        "test_read": False,
        "query_count": len(ours),
        "expected_query_count": len(queries),
        "config": {"beta_dense": float(args.beta_dense), "history_weight": float(args.history_weight)},
        "deltas_vs_v24_dense_c": deltas,
        "cost": {
            "ours_total_wall_ms_mean": ours_wall,
            "v24_full_semantic_selection_ms_mean": full_wall,
            "net_gain_ms_mean": full_wall - ours_wall,
            "candidate_reduction_vs_full": 0.75,
        },
        "safety_passed": safety_passed,
        "development_signal": development_signal,
    }
    _write_json(VERSION_ROOT / "aggregate" / "fixed_calibrated_fusion_development.json", summary)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
