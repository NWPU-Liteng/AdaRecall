from __future__ import annotations

from typing import Any

import numpy as np

from .diffusion import rank_normalize


def fuse_relevance(
    memory_ids: list[str],
    dense_scores: np.ndarray,
    semantic_support: np.ndarray,
    history_support: np.ndarray,
    *,
    beta_dense: float = 0.7,
    history_weight: float = 0.25,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    dense_rank = rank_normalize(dense_scores, memory_ids)
    semantic_rank = rank_normalize(semantic_support, memory_ids)
    history_rank = rank_normalize(history_support, memory_ids)
    graph = (1.0 - float(history_weight)) * semantic_rank + float(history_weight) * history_rank
    graph_rank = rank_normalize(graph, memory_ids)
    fused = float(beta_dense) * dense_rank + (1.0 - float(beta_dense)) * graph_rank
    return fused.astype(np.float32), {
        "dense_rank": dense_rank,
        "semantic_rank": semantic_rank,
        "history_rank": history_rank,
        "graph_rank": graph_rank,
    }


def select_frontier(
    memory_ids: list[str],
    fused_scores: np.ndarray,
    *,
    cap: int,
    components: dict[str, np.ndarray] | None = None,
) -> tuple[list[int], list[dict[str, Any]]]:
    order = sorted(range(len(memory_ids)), key=lambda index: (-float(fused_scores[index]), str(memory_ids[index])))
    selected = order[: min(int(cap), len(order))]
    rows = []
    for index in selected:
        row = {"memory_id": memory_ids[index], "fused_score": float(fused_scores[index])}
        if components:
            for key, values in components.items():
                row[key] = float(values[index])
        rows.append(row)
    return selected, rows


def materialize_pool(
    visible_positions: np.ndarray,
    seed_positions: np.ndarray,
    active_edges: list[dict[str, Any]],
    *,
    cap: int,
    dense_scores_by_position: dict[int, float],
) -> np.ndarray:
    visible = set(int(value) for value in np.asarray(visible_positions, dtype=np.int64))
    positions = {int(value) for value in np.asarray(seed_positions, dtype=np.int64) if int(value) in visible}
    for edge in active_edges:
        members = edge.get("members", [])
        if isinstance(members, np.ndarray):
            positions.update(int(value) for value in members if int(value) in visible)
        else:
            positions.update(int(value) for value in members if int(value) in visible)
    ordered = sorted(positions, key=lambda position: (-float(dense_scores_by_position.get(position, 0.0)), position))
    if len(ordered) > int(cap):
        ordered = ordered[: int(cap)]
    return np.asarray(ordered, dtype=np.int64)
