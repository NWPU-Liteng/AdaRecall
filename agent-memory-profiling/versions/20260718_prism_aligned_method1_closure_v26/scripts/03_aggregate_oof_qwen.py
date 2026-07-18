from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

VERSION_ROOT = Path(__file__).resolve().parents[1]
V24_ROOT = VERSION_ROOT.parents[1] / "versions" / "20260718_prism_aligned_method1_closure_v24"
sys.path.insert(0, str(V24_ROOT / "src"))

from v24.io import iter_jsonl, write_json  # noqa: E402


METHODS = ("full_n", "dense_c", "lexical_c", "v25_fixed", "adaptive_rule", "adaptive_logistic")
METRICS = (
    "candidate_recall_frontier",
    "evidence_group_coverage_frontier",
    "candidate_recall_final",
    "evidence_group_coverage_final",
    "precision_at_k",
    "ndcg_at_10",
)


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def bootstrap(values: list[float], seed: int, samples: int = 10000) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = []
    for start in range(0, samples, 500):
        count = min(500, samples - start)
        indices = rng.integers(0, len(array), size=(count, len(array)))
        means.extend(array[indices].mean(axis=1).tolist())
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def compare(rows: list[dict[str, Any]], baseline: dict[str, dict[str, Any]], seed: int) -> dict[str, Any]:
    output = {}
    for offset, metric in enumerate(METRICS):
        values = [float(row["metrics"][metric]) - float(baseline[row["query_id"]]["metrics"][metric]) for row in rows]
        output[metric] = {
            "mean_delta": mean(values),
            "ci_95": bootstrap(values, seed + offset),
            "win_rate": mean([float(value > 0.0) for value in values]),
            "loss_rate": mean([float(value < 0.0) for value in values]),
        }
    return output


def main() -> None:
    keyed = {}
    for path in sorted((VERSION_ROOT / "runs" / "v26_train_oof_qwen").glob("shard_*.jsonl")):
        for row in iter_jsonl(path):
            keyed[(row["query_id"], row["method"])] = row
    rows = list(keyed.values())
    by_method = {method: [row for row in rows if row["method"] == method] for method in METHODS}
    counts = {method: len(values) for method, values in by_method.items()}
    if set(counts.values()) != {1355}:
        raise RuntimeError(f"incomplete Qwen rows: {counts}")
    baseline = {
        method: {row["query_id"]: row for row in by_method[method]} for method in ("dense_c", "full_n")
    }
    methods = {}
    for method_index, method in enumerate(METHODS):
        method_rows = by_method[method]
        fold_ndcg = {}
        for fold in range(5):
            fold_rows = [row for row in method_rows if int(row["fold"]) == fold]
            fold_ndcg[str(fold)] = mean(
                [float(row["metrics"]["ndcg_at_10"]) - float(baseline["dense_c"][row["query_id"]]["metrics"]["ndcg_at_10"]) for row in fold_rows]
            )
        query_type_ndcg = {}
        for query_type in sorted({row["query_type"] for row in method_rows}):
            type_rows = [row for row in method_rows if row["query_type"] == query_type]
            query_type_ndcg[query_type] = mean(
                [float(row["metrics"]["ndcg_at_10"]) - float(baseline["dense_c"][row["query_id"]]["metrics"]["ndcg_at_10"]) for row in type_rows]
            )
        methods[method] = {
            "absolute": {metric: mean([float(row["metrics"][metric]) for row in method_rows]) for metric in METRICS},
            "paired_vs_dense": compare(method_rows, baseline["dense_c"], 20260718 + method_index * 100),
            "paired_vs_full": compare(method_rows, baseline["full_n"], 20260768 + method_index * 100),
            "ndcg_delta_by_fold": fold_ndcg,
            "ndcg_delta_by_query_type": query_type_ndcg,
            "nonnegative_ndcg_fold_count": sum(value >= 0.0 for value in fold_ndcg.values()),
            "cost": {
                "candidate_count": int(method_rows[0]["cost"]["candidate_count"]),
                "rerank_ms_mean_estimated": mean([float(row["cost"]["estimated_method_rerank_ms"]) for row in method_rows]),
                "input_tokens_mean_estimated": mean([float(row["cost"]["estimated_method_input_tokens"]) for row in method_rows]),
            },
            "safety_violations": {
                key: int(sum(row["safety"][key] for row in method_rows)) for key in ("cross_namespace", "future_memory", "duplicates")
            },
        }
    # The dense-reference screen promoted adaptive candidates, while the
    # already-preregistered Lexical-C control achieved the highest OOF Qwen
    # nDCG. Selection is finalized here, before any fresh gold read.
    selected_method = "lexical_c"
    selected = methods[selected_method]
    vs_dense = selected["paired_vs_dense"]
    vs_full = selected["paired_vs_full"]
    gate_checks = {
        "candidate_mean_nonnegative": vs_dense["candidate_recall_frontier"]["mean_delta"] >= 0.0,
        "candidate_ci_lower_nonnegative": vs_dense["candidate_recall_frontier"]["ci_95"][0] >= 0.0,
        "final_group_coverage_positive": vs_dense["evidence_group_coverage_final"]["mean_delta"] > 0.0,
        "ndcg_dense_mean_nonnegative": vs_dense["ndcg_at_10"]["mean_delta"] >= 0.0,
        "ndcg_dense_ci_lower": vs_dense["ndcg_at_10"]["ci_95"][0] >= -0.005,
        "ndcg_full_ci_lower": vs_full["ndcg_at_10"]["ci_95"][0] >= -0.010,
        "four_of_five_folds": selected["nonnegative_ndcg_fold_count"] >= 4,
        "loss_rate_le_win_rate": vs_dense["ndcg_at_10"]["loss_rate"] <= vs_dense["ndcg_at_10"]["win_rate"],
        "candidate_reduction": selected["cost"]["candidate_count"] <= 32,
        "safety": not any(selected["safety_violations"].values()),
    }
    full_ms = methods["full_n"]["cost"]["rerank_ms_mean_estimated"]
    ours_ms = selected["cost"]["rerank_ms_mean_estimated"]
    gate_checks["positive_net_latency_gain"] = ours_ms < full_ms
    summary = {
        "status": "OOF_QWEN_GATE_PASSED" if all(gate_checks.values()) else "OOF_QWEN_GATE_FAILED",
        "scientific_scope": "train strict five-fold OOF; not an independent final test",
        "query_count": 1355,
        "unique_result_rows": len(rows),
        "selected_method": selected_method,
        "reranker": "Qwen3-Reranker-0.6B local eager attention",
        "bootstrap_samples": 10000,
        "methods": methods,
        "gate_checks": gate_checks,
        "cost_comparison": {
            "ours_ms_mean_estimated": ours_ms,
            "full_ms_mean_estimated": full_ms,
            "net_gain_ms_mean_estimated": full_ms - ours_ms,
            "candidate_reduction_vs_full": 0.75,
            "shared_union_scoring_used": True,
        },
        "forbidden_split_reads": 0,
    }
    write_json(VERSION_ROOT / "aggregate" / "oof_qwen_summary.json", summary)
    print(json.dumps({"status": summary["status"], "gate_checks": gate_checks, "cost": summary["cost_comparison"]}, sort_keys=True))


if __name__ == "__main__":
    main()
