from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


def percentile_signal(memory_ids: Sequence[str], raw_by_id: Mapping[str, float]) -> dict[str, float]:
    order = sorted(memory_ids, key=lambda value: (-float(raw_by_id[value]), str(value)))
    denominator = max(len(order) - 1, 1)
    return {memory_id: 1.0 - rank / denominator for rank, memory_id in enumerate(order)}


def order_signal(memory_ids: Sequence[str]) -> dict[str, float]:
    denominator = max(len(memory_ids) - 1, 1)
    return {str(memory_id): 1.0 - rank / denominator for rank, memory_id in enumerate(memory_ids)}


def graph_prior(frontier_row: Mapping, kind: str) -> dict[str, float]:
    provenance = {row["memory_id"]: row for row in frontier_row["graph_provenance"]}
    memory_ids = list(frontier_row["frontier_ids"])
    if set(provenance) != set(memory_ids):
        raise ValueError("graph provenance does not cover the frozen candidate set")
    if kind == "selection_order":
        raw = {memory_id: -float(provenance[memory_id]["selection_order"]) for memory_id in memory_ids}
    elif kind == "x_sem":
        raw = {memory_id: float(provenance[memory_id]["x_sem"]) for memory_id in memory_ids}
    elif kind == "submodular_total_marginal":
        raw = {
            memory_id: sum(
                float(provenance[memory_id][key])
                for key in ("relevance_marginal", "edge_coverage_marginal", "facility_marginal")
            )
            for memory_id in memory_ids
        }
    else:
        raise ValueError(f"unknown graph prior: {kind}")
    return percentile_signal(memory_ids, raw)


def posterior_rank(
    qwen_row: Mapping,
    frontier_row: Mapping,
    *,
    alpha_qwen: float,
    prior_kind: str,
    graph_method: bool,
) -> list[str]:
    if not 0.0 <= alpha_qwen <= 1.0:
        raise ValueError("alpha_qwen must be in [0, 1]")
    memory_ids = list(qwen_row["frontier_ids"])
    if memory_ids != list(frontier_row["frontier_ids"]):
        raise ValueError("posterior inputs changed candidate identity or order")
    qwen_raw = dict(zip(qwen_row["reranked_ids"], qwen_row["reranker_scores"]))
    if set(qwen_raw) != set(memory_ids):
        raise ValueError("full Qwen scores do not cover candidates")
    qwen = percentile_signal(memory_ids, qwen_raw)
    prior = graph_prior(frontier_row, prior_kind) if graph_method else order_signal(memory_ids)
    return sorted(
        memory_ids,
        key=lambda value: (
            -(alpha_qwen * qwen[value] + (1.0 - alpha_qwen) * prior[value]),
            -qwen[value],
            str(value),
        ),
    )
