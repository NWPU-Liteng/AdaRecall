from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

VERSION_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = VERSION_ROOT.parents[1]
V24_ROOT = PROJECT_ROOT / "versions" / "20260718_prism_aligned_method1_closure_v24"
sys.path.insert(0, str(V24_ROOT / "src"))

from v24.io import iter_jsonl  # noqa: E402


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _ci(values: list[float], seed: int, samples: int = 10000) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = [float(rng.choice(array, size=len(array), replace=True).mean()) for _ in range(samples)]
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _comparison(
    ours: list[dict[str, Any]], baseline: dict[tuple[str, str], dict[str, Any]], method: str, keys: list[str]
) -> dict[str, Any]:
    output = {}
    for index, key in enumerate(keys):
        values = [
            float(row["metrics"][key]) - float(baseline[(row["query_id"], method)]["metrics"][key])
            for row in ours
        ]
        output[key] = {
            "mean_delta": _mean(values),
            "ci_95": _ci(values, 20260718 + index + (100 if method == "full_n" else 0)),
            "win_rate": _mean([float(value > 0.0) for value in values]),
            "loss_rate": _mean([float(value < 0.0) for value in values]),
        }
    return output


def main() -> None:
    ours = list(iter_jsonl(VERSION_ROOT / "runs" / "v25_lexical_fusion_development" / "query_results.jsonl"))
    baseline_rows = list(iter_jsonl(V24_ROOT / "runs" / "v24_development_20260718" / "query_results.jsonl"))
    baseline = {(row["query_id"], row["method"]): row for row in baseline_rows}
    if len(ours) != 290 or any((row["query_id"], "dense_c") not in baseline for row in ours):
        raise RuntimeError("v25 or paired v24 development rows are incomplete")
    keys = [
        "candidate_recall_frontier",
        "evidence_group_coverage_frontier",
        "candidate_recall_final",
        "evidence_group_coverage_final",
        "precision_at_k",
        "ndcg_at_10",
    ]
    absolute = {
        "v25_lexical_fusion_c": {key: _mean([float(row["metrics"][key]) for row in ours]) for key in keys},
        "v24_dense_c": {
            key: _mean([float(baseline[(row["query_id"], "dense_c")]["metrics"][key]) for row in ours])
            for key in keys
        },
        "v24_full_n": {
            key: _mean([float(baseline[(row["query_id"], "full_n")]["metrics"][key]) for row in ours])
            for key in keys
        },
    }
    vs_dense = _comparison(ours, baseline, "dense_c", keys)
    vs_full = _comparison(ours, baseline, "full_n", keys)
    ours_ms = _mean([float(row["total_wall_ms"]) for row in ours])
    dense_ms = _mean([float(baseline[(row["query_id"], "dense_c")]["semantic_selection_ms"]) for row in ours])
    full_ms = _mean([float(baseline[(row["query_id"], "full_n")]["semantic_selection_ms"]) for row in ours])
    safety = not any(
        row["metrics"][key]
        for row in ours
        for key in ("cross_namespace_violation", "future_memory_violation", "duplicate_frontier_ids")
    )
    gates = {
        "candidate_vs_dense": bool(
            vs_dense["candidate_recall_frontier"]["mean_delta"] >= 0.0
            and vs_dense["candidate_recall_frontier"]["ci_95"][0] >= 0.0
        ),
        "final_quality_vs_dense": bool(
            vs_dense["ndcg_at_10"]["mean_delta"] >= 0.0
            and vs_dense["ndcg_at_10"]["ci_95"][0] >= -0.005
        ),
        "quality_vs_full_noninferior": bool(
            vs_full["ndcg_at_10"]["mean_delta"] >= -0.010
            and vs_full["ndcg_at_10"]["ci_95"][0] >= -0.010
        ),
        "cost_vs_full": bool(ours_ms < full_ms),
        "safety": safety,
    }
    development_closed = all(gates.values())
    state = {
        "status": "V25_DEVELOPMENT_CLOSED_AWAITING_FRESH_HOLDOUT" if development_closed else "V25_DEVELOPMENT_INCOMPLETE",
        "scientific_scope": "development only; v24 test was already revealed and was not read or rerun",
        "test_read": False,
        "new_dataset_downloaded": False,
        "query_count": len(ours),
        "method": "dense_lexical_safety_frontier_with_qwen_reranker",
        "algorithm_decision": "weak hypergraph signals remain diagnostics/layout signals, not the primary relevance selector",
        "absolute_metrics": absolute,
        "paired_vs_dense": vs_dense,
        "paired_vs_full": vs_full,
        "cost": {
            "v25_ms_mean": ours_ms,
            "dense_c_ms_mean": dense_ms,
            "full_n_ms_mean": full_ms,
            "net_gain_vs_full_ms_mean": full_ms - ours_ms,
            "candidate_reduction_vs_full": 0.75,
        },
        "gates": gates,
        "development_closed": development_closed,
        "failed_or_bounded_attempts": {
            "degree_calibrated_residual": "failed tuning-to-validation generalization",
            "degree_calibrated_fusion": "safe positive development signal but did not enlarge v24 final effect",
            "lexical_safety_frontier": "passed tuning, validation, Qwen final-quality, cost, and safety development gates",
        },
    }
    aggregate = VERSION_ROOT / "aggregate" / "v25_development_closure.json"
    aggregate.parent.mkdir(parents=True, exist_ok=True)
    aggregate.write_text(json.dumps(state, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report = [
        "# Method 1 v25 Development Closure",
        "",
        f"Status: `{state['status']}`",
        "",
        "v25 does not read or rerun the already revealed v24 test split. Results are development-only and require a fresh holdout for an independent final claim.",
        "",
        "## Main result",
        "",
        f"- Frontier candidate recall vs Dense-C: `{vs_dense['candidate_recall_frontier']['mean_delta']:+.6f}`; 95% CI `{vs_dense['candidate_recall_frontier']['ci_95']}`.",
        f"- Final evidence-group coverage vs Dense-C: `{vs_dense['evidence_group_coverage_final']['mean_delta']:+.6f}`; 95% CI `{vs_dense['evidence_group_coverage_final']['ci_95']}`.",
        f"- Final nDCG@10 vs Dense-C: `{vs_dense['ndcg_at_10']['mean_delta']:+.6f}`; 95% CI `{vs_dense['ndcg_at_10']['ci_95']}`.",
        f"- Final nDCG@10 vs Full-N: `{vs_full['ndcg_at_10']['mean_delta']:+.6f}`; 95% CI `{vs_full['ndcg_at_10']['ci_95']}`.",
        f"- Mean latency: v25 `{ours_ms:.2f} ms`, Full-N `{full_ms:.2f} ms`, net gain `{full_ms - ours_ms:.2f} ms/query`.",
        "- Candidate count: 32 vs Full-N 128 (75% reduction).",
        f"- Safety: `{'PASS' if safety else 'FAIL'}`.",
        "",
        "## Algorithm conclusion",
        "",
        "Degree/overlap calibration removes saturated broad-edge support, but hypergraph relevance still generalizes weakly. The promoted development method uses Dense plus a sparse lexical safety frontier for exact medical entities and dates, followed by the frozen Qwen reranker. Hypergraph evidence remains an auditable diagnostic/layout signal rather than a primary relevance signal.",
        "",
        "## Boundaries",
        "",
        "- Residual hypergraph replacement failed the held-out development partition and was rejected.",
        "- No new dataset was downloaded.",
        "- No v25 test run exists; a fresh holdout or new benchmark is required before claiming independent closure.",
    ]
    reports = VERSION_ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "V25_DEVELOPMENT_CLOSURE.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"status": state["status"], "gates": gates}, sort_keys=True))


if __name__ == "__main__":
    main()
