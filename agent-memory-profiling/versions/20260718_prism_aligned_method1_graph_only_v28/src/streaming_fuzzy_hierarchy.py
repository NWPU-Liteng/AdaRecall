from __future__ import annotations

import math
from collections import defaultdict
from typing import Sequence

import numpy as np
from scipy import sparse

from .contracts import CausalGraphSnapshot, EdgeDescriptor


def dynamic_coarse_k(visible_count: int, minimum: int = 32, maximum: int = 64) -> int:
    if visible_count <= 0:
        return 0
    return min(visible_count, max(minimum, min(maximum, math.ceil(math.sqrt(visible_count)))))


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    return array / np.maximum(np.linalg.norm(array, axis=1, keepdims=True), 1e-12)


def _soft_memberships(values: np.ndarray, centroids: np.ndarray, top_r: int, temperature: float) -> tuple[np.ndarray, np.ndarray]:
    logits = values @ centroids.T / max(float(temperature), 1e-4)
    count = min(int(top_r), centroids.shape[0])
    ids = np.argpartition(-logits, count - 1, axis=1)[:, :count]
    selected = np.take_along_axis(logits, ids, axis=1)
    selected -= selected.max(axis=1, keepdims=True)
    probabilities = np.exp(selected)
    probabilities /= np.maximum(probabilities.sum(axis=1, keepdims=True), 1e-12)
    order = np.argsort(ids, axis=1)
    return np.take_along_axis(ids, order, axis=1), np.take_along_axis(probabilities, order, axis=1)


def _merge_at_compaction(
    centroids: np.ndarray,
    coarse_ids: np.ndarray,
    *,
    similarity_threshold: float,
    maximum_degree: int,
    minimum_age: int,
) -> tuple[np.ndarray, int]:
    if len(centroids) < 2:
        return centroids, 0
    degrees = np.asarray([np.sum(np.any(coarse_ids == i, axis=1)) for i in range(len(centroids))], dtype=np.int64)
    similarities = centroids @ centroids.T
    candidates = []
    for left in range(len(centroids)):
        for right in range(left + 1, len(centroids)):
            candidates.append((float(similarities[left, right]), left, right))
    candidates.sort(key=lambda row: (-row[0], row[1], row[2]))
    consumed: set[int] = set()
    groups: list[tuple[int, ...]] = []
    partners: dict[int, int] = {}
    for similarity, left, right in candidates:
        if similarity < similarity_threshold:
            break
        if left in consumed or right in consumed:
            continue
        if min(int(degrees[left]), int(degrees[right])) < minimum_age:
            continue
        union_degree = int(np.sum(np.any(np.isin(coarse_ids, [left, right]), axis=1)))
        if union_degree > maximum_degree:
            continue
        consumed.update((left, right))
        partners[left] = right
    index = 0
    while index < len(centroids):
        if index in partners:
            groups.append((index, partners[index]))
        elif index not in consumed:
            groups.append((index,))
        index += 1
    if len(groups) == len(centroids):
        return centroids, 0
    merged = []
    for group in groups:
        weights = degrees[np.asarray(group, dtype=np.int64)].astype(np.float64)
        centroid = np.average(centroids[np.asarray(group, dtype=np.int64)], axis=0, weights=np.maximum(weights, 1.0))
        centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
        merged.append(centroid)
    return np.asarray(merged, dtype=np.float32), len(centroids) - len(groups)


def _edge_stats(
    edge_id: str,
    parent_id: str | None,
    centroid: np.ndarray,
    member_rows: np.ndarray,
    member_weights: np.ndarray,
    embeddings: np.ndarray,
    global_positions: np.ndarray,
    arrivals: np.ndarray,
    medoids_per_edge: int,
    version: int,
) -> EdgeDescriptor:
    centroid = np.asarray(centroid, dtype=np.float32)
    centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
    similarities = embeddings[member_rows] @ centroid
    medoid_order = sorted(range(len(member_rows)), key=lambda i: (-float(similarities[i]), int(global_positions[member_rows[i]])))
    medoids = tuple(int(global_positions[member_rows[i]]) for i in medoid_order[:medoids_per_edge])
    weights = np.asarray(member_weights, dtype=np.float64)
    weights /= max(float(weights.sum()), 1e-12)
    entropy = float(-(weights * np.log(np.maximum(weights, 1e-12))).sum() / max(math.log(len(weights)), 1.0))
    cohesion = float(np.average(np.clip(similarities, -1.0, 1.0), weights=weights))
    confidence = float(np.clip((cohesion + 1.0) * 0.5 * (1.0 - 0.35 * entropy), 0.05, 1.0))
    return EdgeDescriptor(
        edge_id=edge_id,
        parent_id=parent_id,
        centroid=centroid,
        medoid_positions=medoids,
        degree=int(len(member_rows)),
        entropy=entropy,
        cohesion=cohesion,
        confidence=confidence,
        version=int(version),
        arrival_min=int(arrivals[member_rows].min()),
        arrival_max=int(arrivals[member_rows].max()),
    )


def build_causal_snapshot(
    all_embeddings: np.ndarray,
    all_memory_ids: Sequence[str],
    all_namespaces: Sequence[str],
    all_arrivals: np.ndarray,
    *,
    namespace_id: str,
    cutoff_order: int,
    memberships_per_memory: int = 3,
    temperature: float = 0.08,
    medoids_per_edge: int = 3,
    max_degree_before_split: int = 192,
    split_children: int = 2,
    merge_similarity: float = 0.985,
    minimum_merge_age: int = 16,
    compaction_boundary: bool = True,
    seed: int = 20260718,
) -> CausalGraphSnapshot:
    from sklearn.cluster import MiniBatchKMeans

    namespace_id = str(namespace_id)
    global_positions = np.asarray(
        [i for i, namespace in enumerate(all_namespaces) if str(namespace) == namespace_id and int(all_arrivals[i]) <= int(cutoff_order)],
        dtype=np.int64,
    )
    if not len(global_positions):
        snapshot = CausalGraphSnapshot(
            namespace_id=namespace_id,
            cutoff_order=int(cutoff_order),
            positions=global_positions,
            memory_ids=(),
            embeddings=np.empty((0, np.asarray(all_embeddings).shape[1]), dtype=np.float32),
            arrival_orders=np.empty(0, dtype=np.int64),
            incidence=sparse.csr_matrix((0, 0), dtype=np.float32),
            edges=(),
            coarse_edge_count=0,
            health={"coarse_k": 0, "nonempty_edge_rate": 1.0, "isolated_memories": 0, "finite": True},
            source_max_arrival=-1,
        )
        snapshot.assert_causal()
        return snapshot

    values = _normalize_rows(np.asarray(all_embeddings[global_positions], dtype=np.float32))
    arrivals = np.asarray(all_arrivals[global_positions], dtype=np.int64)
    coarse_k = dynamic_coarse_k(len(values))
    model = MiniBatchKMeans(
        n_clusters=coarse_k,
        random_state=int(seed) + int(cutoff_order) * 1009 + sum(ord(c) for c in namespace_id),
        batch_size=min(512, len(values)),
        n_init=3,
        max_iter=80,
        reassignment_ratio=0.0,
    )
    labels = model.fit_predict(values)
    centroids = _normalize_rows(np.asarray(model.cluster_centers_, dtype=np.float32))
    coarse_ids, coarse_weights = _soft_memberships(values, centroids, memberships_per_memory, temperature)
    merge_count = 0
    if compaction_boundary:
        centroids, merge_count = _merge_at_compaction(
            centroids,
            coarse_ids,
            similarity_threshold=float(merge_similarity),
            maximum_degree=int(max_degree_before_split),
            minimum_age=int(minimum_merge_age),
        )
        coarse_k = len(centroids)
        coarse_ids, coarse_weights = _soft_memberships(values, centroids, memberships_per_memory, temperature)

    coarse_edges: list[EdgeDescriptor] = []
    child_edges: list[EdgeDescriptor] = []
    coarse_columns: list[tuple[np.ndarray, np.ndarray]] = []
    child_columns: list[tuple[np.ndarray, np.ndarray]] = []
    split_count = 0
    for edge_index in range(coarse_k):
        member_rows, membership_cols = np.where(coarse_ids == edge_index)
        member_weights = coarse_weights[member_rows, membership_cols]
        edge_id = f"ns{namespace_id}:c{edge_index}:v{cutoff_order}"
        coarse_edges.append(
            _edge_stats(
                edge_id,
                None,
                centroids[edge_index],
                member_rows,
                member_weights,
                values,
                global_positions,
                arrivals,
                medoids_per_edge,
                cutoff_order,
            )
        )
        coarse_columns.append((member_rows.astype(np.int64), member_weights.astype(np.float32)))

        if len(member_rows) <= max_degree_before_split:
            continue
        child_count = max(int(split_children), math.ceil(len(member_rows) / max_degree_before_split))
        child_count = min(child_count, len(member_rows))
        child_model = MiniBatchKMeans(
            n_clusters=child_count,
            random_state=int(seed) + edge_index * 7919 + int(cutoff_order),
            batch_size=min(256, len(member_rows)),
            n_init=3,
            max_iter=60,
            reassignment_ratio=0.0,
        )
        child_labels = child_model.fit_predict(values[member_rows])
        child_centroids = _normalize_rows(np.asarray(child_model.cluster_centers_, dtype=np.float32))
        for child_index in range(child_count):
            child_local = np.flatnonzero(child_labels == child_index)
            if not len(child_local):
                continue
            child_rows = member_rows[child_local]
            # Child membership partitions the parent's already normalized mass.
            child_weights = member_weights[child_local]
            child_id = f"{edge_id}:s{child_index}"
            child_edges.append(
                _edge_stats(
                    child_id,
                    edge_id,
                    child_centroids[child_index],
                    child_rows,
                    child_weights,
                    values,
                    global_positions,
                    arrivals,
                    medoids_per_edge,
                    cutoff_order,
                )
            )
            child_columns.append((child_rows.astype(np.int64), child_weights.astype(np.float32)))
            split_count += 1

    # Coarse memberships remain the stochastic incidence. Child columns carry
    # the parent's mass only when activated and are stored as structural paths.
    edges = coarse_edges + child_edges
    incidence_columns = coarse_columns + child_columns
    row_ids: list[int] = []
    col_ids: list[int] = []
    data: list[float] = []
    for column, (member_rows, weights) in enumerate(incidence_columns):
        row_ids.extend(member_rows.tolist())
        col_ids.extend([column] * len(member_rows))
        data.extend(weights.tolist())
    incidence = sparse.csr_matrix((np.asarray(data, dtype=np.float32), (row_ids, col_ids)), shape=(len(values), len(edges)), dtype=np.float32)
    degrees = np.asarray((incidence > 0).sum(axis=0)).reshape(-1)
    coarse_degrees = degrees[:coarse_k]
    pairwise = centroids @ centroids.T
    upper = pairwise[np.triu_indices(coarse_k, 1)] if coarse_k > 1 else np.asarray([], dtype=np.float32)
    row_sums = np.asarray(incidence[:, :coarse_k].sum(axis=1)).reshape(-1)
    health = {
        "coarse_k": coarse_k,
        "edge_count": len(edges),
        "nonempty_edge_rate": float(np.mean(coarse_degrees > 0)),
        "degree_p50": float(np.median(coarse_degrees)),
        "degree_p95": float(np.quantile(coarse_degrees, 0.95)),
        "degree_max": int(coarse_degrees.max()),
        "centroid_pairwise_cosine_mean": float(upper.mean()) if len(upper) else 0.0,
        "centroid_pairwise_cosine_max": float(upper.max()) if len(upper) else 0.0,
        "membership_entropy_mean": float(np.mean([edge.entropy for edge in edges[:coarse_k]])),
        "cohesion_mean": float(np.mean([edge.cohesion for edge in edges[:coarse_k]])),
        "split_count": split_count,
        "merge_count": merge_count,
        "isolated_memories": int(np.sum(row_sums <= 0)),
        "finite": bool(np.isfinite(centroids).all() and np.isfinite(incidence.data).all()),
        "prototype_drift": None,
    }
    snapshot = CausalGraphSnapshot(
        namespace_id=namespace_id,
        cutoff_order=int(cutoff_order),
        positions=global_positions,
        memory_ids=tuple(str(all_memory_ids[i]) for i in global_positions),
        embeddings=values,
        arrival_orders=arrivals,
        incidence=incidence,
        edges=tuple(edges),
        coarse_edge_count=coarse_k,
        health=health,
        source_max_arrival=int(arrivals.max()),
    )
    snapshot.assert_causal()
    return snapshot


def snapshot_fingerprint(snapshot: CausalGraphSnapshot) -> str:
    import hashlib

    digest = hashlib.sha256()
    digest.update(snapshot.namespace_id.encode("utf-8"))
    digest.update(str(snapshot.cutoff_order).encode("ascii"))
    digest.update("\n".join(snapshot.memory_ids).encode("utf-8"))
    digest.update(snapshot.incidence.indptr.tobytes())
    digest.update(snapshot.incidence.indices.tobytes())
    digest.update(snapshot.incidence.data.tobytes())
    for edge in snapshot.edges:
        digest.update(edge.edge_id.encode("utf-8"))
        digest.update(edge.centroid.tobytes())
    return digest.hexdigest()
