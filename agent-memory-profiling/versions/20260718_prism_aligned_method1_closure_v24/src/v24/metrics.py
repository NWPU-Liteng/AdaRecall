from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import EvaluationRecord, QueryRunResult


def _dcg(relevances: Sequence[float]) -> float:
    return float(sum(value / np.log2(index + 2.0) for index, value in enumerate(relevances)))


def quality_metrics(ids: Sequence[str], evaluation: EvaluationRecord | None, *, k: int | None = None) -> dict[str, Any]:
    selected = [str(value) for value in ids]
    cutoff = selected[: int(k)] if k is not None else selected
    if evaluation is None:
        return {
            "precision_at_k": None, "recall_at_k": None, "ndcg_at_10": None, "mrr_at_10": None,
            "evidence_group_coverage": None, "candidate_recall": None, "evidence_density": None,
        }
    gold = set(evaluation.gold_memory_ids)
    hits = [1.0 if value in gold else 0.0 for value in cutoff]
    ideal = sorted([1.0] * min(len(gold), len(cutoff)) + [0.0] * max(0, len(cutoff) - len(gold)), reverse=True)
    groups = [set(group) for group in evaluation.evidence_groups if group]
    group_hits = [bool(set(selected) & group) for group in groups]
    useful = set().union(*[set(group) for group in groups]) if groups else gold
    return {
        "precision_at_k": float(sum(hits) / max(len(cutoff), 1)),
        "recall_at_k": float(len(set(cutoff) & gold) / max(len(gold), 1)),
        "ndcg_at_10": float(_dcg(hits) / max(_dcg(ideal), 1e-12)),
        "mrr_at_10": float(next((1.0 / (index + 1) for index, hit in enumerate(hits) if hit), 0.0)),
        "evidence_group_coverage": float(np.mean(group_hits)) if group_hits else 0.0,
        "candidate_recall": float(len(set(selected) & gold) / max(len(gold), 1)),
        "evidence_density": float(len(set(selected) & useful) / max(len(selected), 1)),
    }


def reference_overlap(ids: Sequence[str], reference: Sequence[str]) -> float:
    left, right = set(str(value) for value in ids), set(str(value) for value in reference)
    return float(len(left & right) / max(len(right), 1))


def paired_summary(rows: Sequence[Mapping[str, Any]], value_key: str) -> dict[str, Any]:
    values = [float(row[value_key]) for row in rows if row.get(value_key) is not None]
    if not values:
        return {"count": 0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(array)), "mean": float(array.mean()), "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)), "min": float(array.min()), "max": float(array.max()),
    }


def summarize_results(results: Sequence[QueryRunResult]) -> dict[str, Any]:
    grouped: dict[str, list[QueryRunResult]] = defaultdict(list)
    for result in results:
        grouped[result.method].append(result)
    summary: dict[str, Any] = {"rows": len(results), "methods": {}}
    for method, method_rows in sorted(grouped.items()):
        metrics = {}
        keys = sorted({key for row in method_rows for key in row.metrics})
        for key in keys:
            values = [float(row.metrics[key]) for row in method_rows if isinstance(row.metrics.get(key), (int, float))]
            if values:
                metrics[key] = paired_summary([{key: value} for value in values], key)
        timings = {
            "seed_ms": paired_summary([row.to_mapping() for row in method_rows], "seed_ms"),
            "edge_activation_ms": paired_summary([row.to_mapping() for row in method_rows], "edge_activation_ms"),
            "diffusion_ms": paired_summary([row.to_mapping() for row in method_rows], "diffusion_ms"),
            "frontier_finalize_ms": paired_summary([row.to_mapping() for row in method_rows], "frontier_finalize_ms"),
            "rerank_ms": paired_summary([row.to_mapping() for row in method_rows], "rerank_ms"),
            "semantic_selection_ms": paired_summary([row.to_mapping() for row in method_rows], "semantic_selection_ms"),
        }
        summary["methods"][method] = {"count": len(method_rows), "metrics": metrics, "timings": timings}
    return summary
