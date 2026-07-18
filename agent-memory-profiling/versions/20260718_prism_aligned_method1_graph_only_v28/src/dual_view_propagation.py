from __future__ import annotations

import numpy as np


def graph_node_scores(
    x_sem: np.ndarray,
    x_hist: np.ndarray,
    arrival_orders: np.ndarray,
    *,
    semantic_weight: float = 1.0,
    history_weight: float = 0.0,
    agreement_weight: float = 0.0,
    recency_weight: float = 0.04,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    for name, value in {
        "semantic_weight": semantic_weight,
        "history_weight": history_weight,
        "agreement_weight": agreement_weight,
        "recency_weight": recency_weight,
    }.items():
        if value < 0:
            raise ValueError(f"{name} must be non-negative")
    semantic = np.maximum(np.asarray(x_sem, dtype=np.float64), 0.0)
    history = np.maximum(np.asarray(x_hist, dtype=np.float64), 0.0)
    if semantic.max(initial=0.0) > 0:
        semantic /= semantic.max()
    if history.max(initial=0.0) > 0:
        history /= history.max()
    agreement = np.sqrt(semantic * history)
    arrival = np.asarray(arrival_orders, dtype=np.float64)
    if len(arrival) and arrival.max() > arrival.min():
        recency = (arrival - arrival.min()) / (arrival.max() - arrival.min())
    else:
        recency = np.zeros_like(arrival)
    score = semantic_weight * semantic + history_weight * history + agreement_weight * agreement + recency_weight * recency
    return score.astype(np.float32), agreement.astype(np.float32), recency.astype(np.float32)
