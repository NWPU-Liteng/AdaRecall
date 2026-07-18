from __future__ import annotations

from typing import Iterable

import numpy as np


def diffuse(
    node_count: int,
    edge_members: Iterable[np.ndarray],
    edge_weights: Iterable[float],
    restart: np.ndarray,
    *,
    alpha: float = 0.8,
    steps: int = 2,
) -> np.ndarray:
    """Run fixed-step query-conditioned hypergraph propagation on a local pool."""
    x0 = np.asarray(restart, dtype=np.float32).reshape(-1)
    if len(x0) != int(node_count):
        raise ValueError("restart vector does not match local node count")
    x0 = np.maximum(x0, 0.0)
    x0 /= max(float(x0.sum()), 1e-12)
    members = [np.unique(np.asarray(value, dtype=np.int64)) for value in edge_members]
    weights = np.asarray([max(float(value), 0.0) for value in edge_weights], dtype=np.float32)
    degree_e = np.asarray([max(len(value), 1) for value in members], dtype=np.float32)
    degree_v = np.zeros(int(node_count), dtype=np.float32)
    for value, weight in zip(members, weights):
        degree_v[value] += float(weight)
    x = x0.copy()
    for _ in range(max(0, int(steps))):
        source = x / np.maximum(degree_v, 1e-12)
        next_values = np.zeros_like(x)
        for value, weight, edge_degree in zip(members, weights, degree_e):
            if not len(value) or weight <= 0.0:
                continue
            mass = float(source[value].sum())
            next_values[value] += float(weight / edge_degree) * mass
        x = float(alpha) * x0 + (1.0 - float(alpha)) * next_values
        total = float(x.sum())
        if total > 0.0:
            x /= total
    return x.astype(np.float32, copy=False)


def rank_normalize(values: np.ndarray, ids: list[str] | None = None) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if not len(values):
        return values.copy()
    ids = ids or [str(index) for index in range(len(values))]
    order = sorted(range(len(values)), key=lambda index: (-float(values[index]), str(ids[index])))
    result = np.zeros(len(values), dtype=np.float32)
    denominator = max(len(values) - 1, 1)
    for rank, index in enumerate(order):
        result[index] = 1.0 - float(rank) / float(denominator)
    return result
