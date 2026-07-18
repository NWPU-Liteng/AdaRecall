from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
from scipy import sparse


def _row_rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-np.asarray(values), kind="stable")
    rank = np.empty(len(order), dtype=np.float32)
    rank[order] = np.arange(len(order), dtype=np.float32)
    return 1.0 - rank / max(float(len(order) - 1), 1.0)


@dataclass
class SemanticBuild:
    prototypes: np.ndarray
    incidence: sparse.csr_matrix
    diagnostics: dict[str, Any]


def _fit_centroids(values: np.ndarray, n_prototypes: int, seed: int) -> np.ndarray:
    normalized = np.asarray(values, dtype=np.float32)
    normalized = normalized / np.maximum(np.linalg.norm(normalized, axis=1, keepdims=True), 1e-12)
    n_prototypes = min(int(n_prototypes), max(1, len(normalized)))
    try:
        from sklearn.cluster import MiniBatchKMeans

        model = MiniBatchKMeans(
            n_clusters=n_prototypes, random_state=int(seed), batch_size=min(2048, max(64, len(normalized))),
            n_init=3, max_iter=80, reassignment_ratio=0.01,
        )
        model.fit(normalized)
        centers = np.asarray(model.cluster_centers_, dtype=np.float32)
    except Exception:
        rng = np.random.default_rng(int(seed))
        indices = rng.choice(len(normalized), size=n_prototypes, replace=False)
        centers = normalized[indices].copy()
        for _ in range(8):
            scores = normalized @ centers.T
            labels = np.argmax(scores, axis=1)
            for cluster in range(n_prototypes):
                members = normalized[labels == cluster]
                if len(members):
                    centers[cluster] = members.mean(axis=0)
    centers /= np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-12)
    return centers.astype(np.float32)


def _health(
    values: np.ndarray,
    prototypes: np.ndarray,
    incidence: sparse.csr_matrix,
    *,
    temperature: float,
) -> dict[str, Any]:
    degree = np.asarray(incidence.sum(axis=0)).reshape(-1)
    nonempty = degree > 0
    pairwise = prototypes @ prototypes.T
    upper = pairwise[np.triu_indices(len(prototypes), k=1)]
    row_values = incidence.data
    entropy = np.zeros(incidence.shape[0], dtype=np.float64)
    cursor = 0
    effective = np.zeros(incidence.shape[0], dtype=np.int64)
    for row in range(incidence.shape[0]):
        end = incidence.indptr[row + 1]
        weights = incidence.data[cursor:end]
        cursor = end
        if len(weights):
            probabilities = weights / max(float(weights.sum()), 1e-12)
            entropy[row] = float(-(probabilities * np.log(np.maximum(probabilities, 1e-12))).sum())
            effective[row] = int(np.count_nonzero(weights > 1e-5))
    max_degree_ratio = float(degree.max() / max(float(degree.sum()), 1e-12)) if len(degree) else 0.0
    diagnostics = {
        "prototype_count": int(len(prototypes)),
        "nonempty_prototypes": int(nonempty.sum()),
        "nonempty_prototype_fraction": float(nonempty.mean()) if len(nonempty) else 0.0,
        "member_count_min": int(degree.min()) if len(degree) else 0,
        "member_count_median": float(np.median(degree)) if len(degree) else 0.0,
        "member_count_max": int(degree.max()) if len(degree) else 0,
        "prototype_pairwise_cosine_mean": float(upper.mean()) if len(upper) else 1.0,
        "prototype_pairwise_cosine_max": float(upper.max()) if len(upper) else 1.0,
        "assignment_entropy_mean": float(entropy.mean()) if len(entropy) else 0.0,
        "assignment_entropy_p95": float(np.percentile(entropy, 95)) if len(entropy) else 0.0,
        "effective_membership_count_mean": float(effective.mean()) if len(effective) else 0.0,
        "incidence_nnz": int(incidence.nnz),
        "max_prototype_mass_fraction": max_degree_ratio,
        "temperature": float(temperature),
        "finite": bool(np.isfinite(prototypes).all() and np.isfinite(row_values).all()),
    }
    diagnostics["prototype_collapse_flag"] = bool(
        diagnostics["nonempty_prototype_fraction"] < 0.80
        or diagnostics["prototype_pairwise_cosine_mean"] > 0.985
        or diagnostics["max_prototype_mass_fraction"] > 0.35
        or not diagnostics["finite"]
    )
    diagnostics["health_passed"] = not diagnostics["prototype_collapse_flag"]
    return diagnostics


def fit_spherical_soft(
    embeddings: np.ndarray,
    *,
    n_prototypes: int = 128,
    top_memberships: int = 4,
    temperature: float = 0.10,
    seed: int = 20260718,
) -> SemanticBuild:
    started = perf_counter()
    values = np.asarray(embeddings, dtype=np.float32)
    values = values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
    prototypes = _fit_centroids(values, int(n_prototypes), int(seed))
    logits = (values @ prototypes.T) / max(float(temperature), 1e-4)
    r = min(max(1, int(top_memberships)), logits.shape[1])
    top = np.argpartition(-logits, r - 1, axis=1)[:, :r]
    selected = np.take_along_axis(logits, top, axis=1)
    selected -= selected.max(axis=1, keepdims=True)
    weights = np.exp(selected)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    rows = np.repeat(np.arange(len(values), dtype=np.int64), r)
    cols = top.reshape(-1).astype(np.int64)
    data = weights.reshape(-1).astype(np.float32)
    incidence = sparse.csr_matrix((data, (rows, cols)), shape=(len(values), len(prototypes)), dtype=np.float32)
    incidence.sum_duplicates()
    diagnostics = _health(values, prototypes, incidence, temperature=float(temperature))
    diagnostics["build_ms"] = float((perf_counter() - started) * 1000.0)
    return SemanticBuild(prototypes, incidence, diagnostics)


class SemanticHypergraph:
    def __init__(self, prototypes: np.ndarray, incidence: sparse.spmatrix, memory_ids: list[str]) -> None:
        self.prototypes = np.array(prototypes, dtype=np.float32, copy=True)
        self.prototypes /= np.maximum(np.linalg.norm(self.prototypes, axis=1, keepdims=True), 1e-12)
        self.incidence = sparse.csr_matrix(incidence, dtype=np.float32)
        self.edge_members = self.incidence.T.tocsr()
        self.memory_ids = list(memory_ids)
        self.position_by_id = {memory_id: index for index, memory_id in enumerate(self.memory_ids)}

    def active_edges(
        self,
        query_vector: np.ndarray,
        seed_positions: np.ndarray,
        seed_scores: np.ndarray,
        *,
        top_edges: int = 16,
        tau: float = 0.10,
    ) -> tuple[list[dict[str, Any]], float]:
        started = perf_counter()
        seeds = np.asarray(seed_positions, dtype=np.int64)
        if seeds.size == 0:
            return [], 0.0
        seed_values = np.asarray(seed_scores, dtype=np.float32)
        seed_values = np.maximum(seed_values - float(seed_values.min(initial=0.0)), 0.0)
        if not np.any(seed_values):
            seed_values = np.ones_like(seed_values)
        seed_values /= max(float(seed_values.sum()), 1e-12)
        local_incidence = self.incidence[seeds]
        edge_ids = np.unique(local_incidence.indices).astype(np.int64)
        if edge_ids.size == 0:
            return [], float((perf_counter() - started) * 1000.0)
        seed_mass = np.asarray(local_incidence.T @ seed_values).reshape(-1)
        query = np.array(query_vector, dtype=np.float32, copy=True).reshape(-1)
        query /= max(float(np.linalg.norm(query)), 1e-12)
        query_match = 1.0 / (1.0 + np.exp(-((self.prototypes[edge_ids] @ query) / max(float(tau), 1e-4))))
        scores = query_match * (1e-3 + seed_mass[edge_ids])
        order = np.argsort(-scores, kind="stable")[: int(top_edges)]
        active = []
        for index in order:
            edge_id = int(edge_ids[index])
            members = self.edge_members.indices[self.edge_members.indptr[edge_id]:self.edge_members.indptr[edge_id + 1]]
            active.append({
                "edge_id": edge_id, "edge_type": "semantic", "activation": float(scores[index]),
                "query_match": float(query_match[index]), "seed_mass": float(seed_mass[edge_id]),
                "members": members.astype(np.int64, copy=False),
            })
        return active, float((perf_counter() - started) * 1000.0)
