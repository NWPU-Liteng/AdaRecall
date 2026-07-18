from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _percentile_rank(values: np.ndarray, memory_ids: list[str]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = sorted(range(len(values)), key=lambda i: (float(values[i]), str(memory_ids[i])))
    ranks = np.empty(len(values), dtype=np.float32)
    denominator = max(len(values) - 1, 1)
    for rank, index in enumerate(order):
        ranks[index] = float(rank) / float(denominator)
    return ranks


def calibrated_edge_weight(
    activation: float,
    *,
    edge_size: int,
    visible_count: int,
    seed_overlap: int,
) -> float:
    """Downweight broad edges and edges justified by only one seed."""
    size = max(int(edge_size), 1)
    visible = max(int(visible_count), size)
    idf = np.log1p(visible / size) / np.log1p(visible)
    overlap_confidence = 1.0 - np.exp(-max(int(seed_overlap) - 1, 0))
    return float(max(float(activation), 0.0) * idf * overlap_confidence)


@dataclass(frozen=True)
class ResidualConfig:
    max_replacements: int = 2
    protected_prefix: int = 24
    graph_margin: float = 0.10
    max_dense_drop: float = 0.04
    history_weight: float = 0.10
    minimum_graph_rank: float = 0.70


def select_residual_frontier(
    memory_ids: list[str],
    dense_scores: np.ndarray,
    semantic_support: np.ndarray,
    history_support: np.ndarray,
    *,
    cap: int,
    config: ResidualConfig,
) -> tuple[list[int], list[dict[str, Any]]]:
    """Start from Dense-C and admit only graph-supported residual challengers."""
    if not (len(memory_ids) == len(dense_scores) == len(semantic_support) == len(history_support)):
        raise ValueError("frontier component lengths differ")
    if cap <= 0 or not memory_ids:
        return [], []

    dense_order = sorted(range(len(memory_ids)), key=lambda i: (-float(dense_scores[i]), str(memory_ids[i])))
    anchor = dense_order[: min(int(cap), len(dense_order))]
    if config.max_replacements <= 0 or len(dense_order) <= len(anchor):
        return anchor, _trace(anchor, memory_ids, dense_scores, semantic_support, history_support, set())

    semantic_rank = _percentile_rank(semantic_support, memory_ids)
    history_rank = _percentile_rank(history_support, memory_ids)
    graph = (1.0 - float(config.history_weight)) * semantic_rank + float(config.history_weight) * history_rank
    protected = max(0, min(int(config.protected_prefix), len(anchor)))
    incumbents = list(reversed(anchor[protected:]))
    challengers = sorted(
        dense_order[len(anchor) :],
        key=lambda i: (-float(graph[i]), -float(dense_scores[i]), str(memory_ids[i])),
    )
    selected = set(anchor)
    replacements: set[int] = set()
    used_incumbents: set[int] = set()
    for challenger in challengers:
        if len(replacements) >= int(config.max_replacements):
            break
        if float(graph[challenger]) < float(config.minimum_graph_rank):
            continue
        incumbent = next((i for i in incumbents if i not in used_incumbents), None)
        if incumbent is None:
            break
        graph_gain = float(graph[challenger] - graph[incumbent])
        dense_drop = float(dense_scores[incumbent] - dense_scores[challenger])
        if graph_gain < float(config.graph_margin) or dense_drop > float(config.max_dense_drop):
            continue
        selected.remove(incumbent)
        selected.add(challenger)
        used_incumbents.add(incumbent)
        replacements.add(challenger)

    output = sorted(selected, key=lambda i: (-float(dense_scores[i]), str(memory_ids[i])))
    return output, _trace(output, memory_ids, dense_scores, semantic_support, history_support, replacements, graph)


def _trace(
    selected: list[int],
    memory_ids: list[str],
    dense_scores: np.ndarray,
    semantic_support: np.ndarray,
    history_support: np.ndarray,
    replacements: set[int],
    graph: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    return [
        {
            "memory_id": memory_ids[index],
            "dense_score": float(dense_scores[index]),
            "semantic_support": float(semantic_support[index]),
            "history_support": float(history_support[index]),
            "graph_score": float(graph[index]) if graph is not None else 0.0,
            "residual_replacement": bool(index in replacements),
        }
        for index in selected
    ]
