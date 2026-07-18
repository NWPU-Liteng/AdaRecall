from __future__ import annotations

import numpy as np
from scipy import sparse

from .contracts import CausalGraphSnapshot
from .query_edge_alignment import EdgeActivation


def semantic_distribution(
    snapshot: CausalGraphSnapshot,
    activation: EdgeActivation,
    *,
    restart: float = 0.78,
    steps: int = 1,
    pool_cap: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = len(snapshot.positions)
    if not activation.active_columns or not count:
        return np.zeros(count, dtype=np.float32), np.zeros(count, dtype=np.float32), np.empty(0, dtype=np.int64)
    columns = np.asarray(activation.active_columns, dtype=np.int64)
    incidence = snapshot.incidence[:, columns].tocsr().astype(np.float64)
    degrees = np.asarray((incidence > 0).sum(axis=0)).reshape(-1)
    width_penalty = 1.0 / np.sqrt(np.maximum(degrees, 1.0))
    x0 = np.asarray(incidence @ (activation.weights.astype(np.float64) * width_penalty), dtype=np.float64).reshape(-1)
    if x0.sum() > 0:
        x0 /= x0.sum()
    node_degree = np.asarray(incidence.sum(axis=1)).reshape(-1)
    edge_degree = np.asarray(incidence.sum(axis=0)).reshape(-1)
    dv_inv = sparse.diags(1.0 / np.maximum(node_degree, 1e-12))
    de_inv = sparse.diags(1.0 / np.maximum(edge_degree, 1e-12))
    transition = (dv_inv @ incidence @ de_inv @ incidence.T).tocsr()
    x = x0.copy()
    for _ in range(int(steps)):
        x = float(restart) * x0 + (1.0 - float(restart)) * np.asarray(transition.T @ x).reshape(-1)
    positive = np.flatnonzero(x0 > 0)
    order = sorted(positive.tolist(), key=lambda i: (-float(x[i]), snapshot.memory_ids[i]))
    pool = np.asarray(order[: int(pool_cap)], dtype=np.int64)
    return x0.astype(np.float32), x.astype(np.float32), pool


def transition_row_sums(snapshot: CausalGraphSnapshot, active_columns: tuple[int, ...]) -> np.ndarray:
    if not active_columns:
        return np.empty(0, dtype=np.float64)
    incidence = snapshot.incidence[:, np.asarray(active_columns, dtype=np.int64)].tocsr().astype(np.float64)
    active_rows = np.flatnonzero(np.asarray(incidence.sum(axis=1)).reshape(-1) > 0)
    local = incidence[active_rows]
    dv_inv = sparse.diags(1.0 / np.maximum(np.asarray(local.sum(axis=1)).reshape(-1), 1e-12))
    de_inv = sparse.diags(1.0 / np.maximum(np.asarray(local.sum(axis=0)).reshape(-1), 1e-12))
    transition = dv_inv @ local @ de_inv @ local.T
    return np.asarray(transition.sum(axis=1)).reshape(-1)
