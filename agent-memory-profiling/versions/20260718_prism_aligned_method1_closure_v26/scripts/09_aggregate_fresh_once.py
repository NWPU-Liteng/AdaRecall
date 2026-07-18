from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

VERSION_ROOT = Path(__file__).resolve().parents[1]
V24_ROOT = VERSION_ROOT.parents[1] / "versions" / "20260718_prism_aligned_method1_closure_v24"
sys.path.insert(0, str(V24_ROOT / "src"))
from v24.io import iter_jsonl, write_json  # noqa: E402

METHODS = ("full_n", "dense_c", "lexical_c", "adaptive_rule")
METRICS = ("candidate_recall_frontier", "evidence_group_coverage_frontier", "candidate_recall_final", "evidence_group_coverage_final", "precision_at_k", "ndcg_at_10")


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def ci(values: list[float], seed: int) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(20):
        indices = rng.integers(0, len(array), size=(500, len(array)))
        means.extend(array[indices].mean(axis=1).tolist())
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def compare(rows: list[dict[str, Any]], baseline: dict[str, dict[str, Any]], seed: int) -> dict[str, Any]:
    output = {}
    for offset, metric in enumerate(METRICS):
        values = [float(row["metrics"][metric]) - float(baseline[row["query_id"]]["metrics"][metric]) for row in rows]
        output[metric] = {"mean_delta": mean(values), "ci_95": ci(values, seed + offset), "win_rate": mean([float(x > 0) for x in values]), "loss_rate": mean([float(x < 0) for x in values])}
    return output


def main() -> None:
    keyed = {}
    for path in sorted((VERSION_ROOT / "runs" / "v26_fresh_once").glob("qwen_shard_*.jsonl")):
        for row in iter_jsonl(path):
            keyed[(row["query_id"], row["method"])] = row
    rows = list(keyed.values())
    by_method = {method: [row for row in rows if row["method"] == method] for method in METHODS}
    counts = {method: len(value) for method, value in by_method.items()}
    if len(set(counts.values())) != 1 or not next(iter(counts.values()), 0):
        raise RuntimeError(f"incomplete fresh results: {counts}")
    baseline = {method: {row["query_id"]: row for row in by_method[method]} for method in ("dense_c", "full_n")}
    methods = {}
    for index, method in enumerate(METHODS):
        values = by_method[method]
        methods[method] = {"absolute": {metric: mean([float(row["metrics"][metric]) for row in values]) for metric in METRICS}, "paired_vs_dense": compare(values, baseline["dense_c"], 20260718 + index * 100), "paired_vs_full": compare(values, baseline["full_n"], 20260768 + index * 100), "cost": {"candidate_count": values[0]["cost"]["candidate_count"], "rerank_ms_mean_estimated": mean([float(row["cost"]["estimated_method_ms"]) for row in values]), "input_tokens_mean_estimated": mean([float(row["cost"]["estimated_input_tokens"]) for row in values])}, "safety_violations": {key: int(sum(row["safety"][key] for row in values)) for key in ("cross_namespace", "future_memory", "duplicates")}}
    ours, full = methods["lexical_c"], methods["full_n"]
    vd, vf = ours["paired_vs_dense"], ours["paired_vs_full"]
    gates = {"candidate_mean": vd["candidate_recall_frontier"]["mean_delta"] >= 0, "candidate_ci": vd["candidate_recall_frontier"]["ci_95"][0] >= 0, "final_coverage": vd["evidence_group_coverage_final"]["mean_delta"] > 0, "ndcg_dense_mean": vd["ndcg_at_10"]["mean_delta"] >= 0, "ndcg_dense_ci": vd["ndcg_at_10"]["ci_95"][0] >= -0.005, "ndcg_full_ci": vf["ndcg_at_10"]["ci_95"][0] >= -0.010, "loss_vs_win": vd["ndcg_at_10"]["loss_rate"] <= vd["ndcg_at_10"]["win_rate"], "cost": ours["cost"]["rerank_ms_mean_estimated"] < full["cost"]["rerank_ms_mean_estimated"], "candidate_reduction": ours["cost"]["candidate_count"] == 32, "safety": not any(ours["safety_violations"].values())}
    summary = {"status": "FRESH_HOLDOUT_PASSED" if all(gates.values()) else "FRESH_HOLDOUT_FAILED", "dataset": "LoCoMo independent fresh holdout", "formal_run_count": 1, "query_count": counts["lexical_c"], "selected_method": "lexical_c", "bootstrap_samples": 10000, "methods": methods, "gate_checks": gates, "cost_comparison": {"ours_ms": ours["cost"]["rerank_ms_mean_estimated"], "full_ms": full["cost"]["rerank_ms_mean_estimated"], "net_gain_ms": full["cost"]["rerank_ms_mean_estimated"] - ours["cost"]["rerank_ms_mean_estimated"], "candidate_reduction_vs_full": 0.75}}
    write_json(VERSION_ROOT / "aggregate" / "fresh_holdout_summary.json", summary)
    print(json.dumps({"status": summary["status"], "gates": gates}, sort_keys=True))


if __name__ == "__main__":
    main()
