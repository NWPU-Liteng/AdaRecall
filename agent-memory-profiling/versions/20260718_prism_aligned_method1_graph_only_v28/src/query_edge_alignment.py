from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contracts import CausalGraphSnapshot


@dataclass(frozen=True)
class EdgeActivation:
    active_columns: tuple[int, ...]
    active_edge_ids: tuple[str, ...]
    weights: np.ndarray
    raw_scores: np.ndarray
    entropy: float
    margin: float
    high_confidence: bool
    centroid_comparisons: int
    medoid_comparisons: int


def _rank_normalize(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return array.astype(np.float32)
    order = np.argsort(np.argsort(array, kind="mergesort"), kind="mergesort")
    return (order / max(len(array) - 1, 1)).astype(np.float32)


def align_query_to_edges(
    snapshot: CausalGraphSnapshot,
    query_embedding: np.ndarray,
    all_embeddings: np.ndarray,
    *,
    centroid_weight: float = 0.72,
    medoid_weight: float = 0.28,
    temperature: float = 0.18,
    active_high: int = 4,
    active_low: int = 8,
    confidence_entropy_max: float = 0.68,
    confidence_margin_min: float = 0.08,
) -> EdgeActivation:
    if not snapshot.coarse_edge_count:
        return EdgeActivation((), (), np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32), 0.0, 0.0, False, 0, 0)
    query = np.asarray(query_embedding, dtype=np.float32)
    query /= max(float(np.linalg.norm(query)), 1e-12)
    edges = snapshot.edges[: snapshot.coarse_edge_count]
    centroids = np.vstack([edge.centroid for edge in edges]).astype(np.float32)
    centroid_scores = centroids @ query
    medoid_scores = np.empty(len(edges), dtype=np.float32)
    medoid_comparisons = 0
    for i, edge in enumerate(edges):
        rows = np.asarray(edge.medoid_positions, dtype=np.int64)
        medoids = np.asarray(all_embeddings[rows], dtype=np.float32)
        medoids /= np.maximum(np.linalg.norm(medoids, axis=1, keepdims=True), 1e-12)
        scores = medoids @ query
        # The medoid signal is aggregated at edge level and cannot identify a candidate.
        medoid_scores[i] = float(0.65 * scores.max() + 0.35 * scores.mean())
        medoid_comparisons += len(rows)
    raw = centroid_weight * _rank_normalize(centroid_scores) + medoid_weight * _rank_normalize(medoid_scores)
    degree = np.asarray([edge.degree for edge in edges], dtype=np.float32)
    specificity = np.log((len(snapshot.positions) + 1.0) / (degree + 1.0))
    specificity /= max(float(specificity.max()), 1e-12)
    confidence = np.asarray([edge.confidence for edge in edges], dtype=np.float32)
    logits = raw / max(float(temperature), 1e-4)
    logits -= logits.max()
    probabilities = np.exp(logits)
    probabilities /= max(float(probabilities.sum()), 1e-12)
    entropy = float(-(probabilities * np.log(np.maximum(probabilities, 1e-12))).sum() / max(np.log(len(probabilities)), 1.0))
    order = sorted(range(len(edges)), key=lambda i: (-float(raw[i]), edges[i].edge_id))
    margin = float(raw[order[0]] - raw[order[1]]) if len(order) > 1 else 1.0
    high_confidence = entropy <= confidence_entropy_max and margin >= confidence_margin_min
    # With fewer memories than the frontier budget, every visible graph node
    # must remain reachable; activating all coarse edges is the causal graph
    # equivalent of returning the full visible set.
    active_count = len(order) if len(snapshot.positions) <= 32 else min(active_high if high_confidence else active_low, len(order))
    coarse_columns = tuple(order[:active_count])
    coarse_weights = probabilities[np.asarray(coarse_columns)] * specificity[np.asarray(coarse_columns)] * confidence[np.asarray(coarse_columns)]
    coarse_weights /= max(float(coarse_weights.sum()), 1e-12)
    # A split parent routes through the single child whose descriptor best
    # matches the query. Its weight replaces, rather than adds to, parent mass.
    active_columns_list: list[int] = []
    active_weights_list: list[float] = []
    for coarse_column, coarse_mass in zip(coarse_columns, coarse_weights):
        parent_id = edges[coarse_column].edge_id
        children = [(i, edge) for i, edge in enumerate(snapshot.edges) if edge.parent_id == parent_id]
        if not children:
            active_columns_list.append(coarse_column)
            active_weights_list.append(float(coarse_mass))
            continue
        child_scores = []
        for child_column, child in children:
            centroid_score = float(child.centroid @ query)
            rows = np.asarray(child.medoid_positions, dtype=np.int64)
            medoids = np.asarray(all_embeddings[rows], dtype=np.float32)
            medoids /= np.maximum(np.linalg.norm(medoids, axis=1, keepdims=True), 1e-12)
            descriptor_score = centroid_weight * centroid_score + medoid_weight * float((medoids @ query).mean())
            child_scores.append((descriptor_score, child.edge_id, child_column))
            medoid_comparisons += len(rows)
        child_column = max(child_scores, key=lambda row: (row[0], row[1]))[2]
        active_columns_list.append(child_column)
        active_weights_list.append(float(coarse_mass))
    active_columns = tuple(active_columns_list)
    active_weights = np.asarray(active_weights_list, dtype=np.float32)
    return EdgeActivation(
        active_columns=active_columns,
        active_edge_ids=tuple(snapshot.edges[i].edge_id for i in active_columns),
        weights=active_weights.astype(np.float32),
        raw_scores=raw.astype(np.float32),
        entropy=entropy,
        margin=margin,
        high_confidence=high_confidence,
        centroid_comparisons=len(edges),
        medoid_comparisons=medoid_comparisons,
    )
