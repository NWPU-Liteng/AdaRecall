from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .contracts import CandidateProvenance, CausalGraphSnapshot, GraphSelectionResult


@dataclass(frozen=True)
class ObjectiveWeights:
    relevance: float = 1.0
    edge_coverage: float = 0.25
    facility: float = 0.08

    def validate(self) -> None:
        if min(self.relevance, self.edge_coverage, self.facility) < 0:
            raise ValueError("submodular objective weights must be non-negative")


def incidence_similarity(incidence: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(incidence, dtype=np.float64), 0.0)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    normalized = values / np.maximum(norms, 1e-12)
    similarity = np.clip(normalized @ normalized.T, 0.0, 1.0)
    np.fill_diagonal(similarity, 1.0)
    return similarity.astype(np.float32)


def _objective(
    chosen: Sequence[int],
    relevance: np.ndarray,
    incidence: np.ndarray,
    edge_weights: np.ndarray,
    similarity: np.ndarray,
    omega: np.ndarray,
    weights: ObjectiveWeights,
) -> float:
    if not chosen:
        return 0.0
    rows = np.asarray(chosen, dtype=np.int64)
    relevance_value = float(relevance[rows].sum())
    coverage_value = float((edge_weights * np.minimum(1.0, incidence[rows].sum(axis=0))).sum())
    facility_value = float((omega * similarity[:, rows].max(axis=1)).sum())
    return weights.relevance * relevance_value + weights.edge_coverage * coverage_value + weights.facility * facility_value


def select_graph_only(
    snapshot: CausalGraphSnapshot,
    pool: np.ndarray,
    graph_scores: np.ndarray,
    x0: np.ndarray,
    x_sem: np.ndarray,
    x_hist: np.ndarray,
    agreement: np.ndarray,
    recency: np.ndarray,
    active_columns: tuple[int, ...],
    active_edge_weights: np.ndarray,
    *,
    history_edge_ids_by_local: dict[int, tuple[str, ...]] | None = None,
    budget: int = 32,
    weights: ObjectiveWeights = ObjectiveWeights(),
) -> GraphSelectionResult:
    weights.validate()
    pool = np.asarray(pool, dtype=np.int64)
    target = min(int(budget), len(snapshot.positions))
    underflow = len(snapshot.positions) >= budget and len(pool) < budget
    if underflow or target == 0:
        return GraphSelectionResult([], [], [], len(pool), tuple(snapshot.edges[i].edge_id for i in active_columns), underflow, {"dense_fallback_slots": 0})
    pool = pool[:]
    incidence = snapshot.incidence[pool][:, np.asarray(active_columns, dtype=np.int64)].toarray().astype(np.float32)
    relevance = np.maximum(np.asarray(graph_scores)[pool], 0.0).astype(np.float32)
    similarity = incidence_similarity(incidence)
    omega = np.maximum(relevance, 1e-6)
    omega /= omega.sum()
    edge_weights = np.asarray(active_edge_weights, dtype=np.float32)
    covered = np.zeros(len(active_columns), dtype=np.float32)
    facility_best = np.zeros(len(pool), dtype=np.float32)
    selected: list[int] = []
    selected_set: set[int] = set()
    marginal_parts: dict[int, tuple[float, float, float]] = {}

    def marginal(candidate: int) -> tuple[float, tuple[float, float, float]]:
        relevance_gain = weights.relevance * float(relevance[candidate])
        new_coverage = np.minimum(1.0, covered + incidence[candidate])
        coverage_gain = weights.edge_coverage * float((edge_weights * (new_coverage - covered)).sum())
        facility_gain = weights.facility * float((omega * np.maximum(similarity[:, candidate] - facility_best, 0.0)).sum())
        return relevance_gain + coverage_gain + facility_gain, (relevance_gain, coverage_gain, facility_gain)

    heap: list[tuple[float, str, int, int]] = []
    for candidate in range(len(pool)):
        gain, _ = marginal(candidate)
        heapq.heappush(heap, (-gain, snapshot.memory_ids[int(pool[candidate])], candidate, 0))
    evaluations = 0
    while len(selected) < target and heap:
        negative_bound, memory_id, candidate, stamp = heapq.heappop(heap)
        if candidate in selected_set:
            continue
        if stamp == len(selected):
            _, parts = marginal(candidate)
            selected.append(candidate)
            selected_set.add(candidate)
            marginal_parts[candidate] = parts
            covered[:] = np.minimum(1.0, covered + incidence[candidate])
            facility_best[:] = np.maximum(facility_best, similarity[:, candidate])
            continue
        gain, _ = marginal(candidate)
        evaluations += 1
        heapq.heappush(heap, (-gain, memory_id, candidate, len(selected)))

    selected_local = [int(pool[index]) for index in selected]
    provenance = []
    history_edge_ids_by_local = history_edge_ids_by_local or {}
    for order, (pool_index, local_position) in enumerate(zip(selected, selected_local), start=1):
        active_memberships = incidence[pool_index]
        semantic_edges = tuple(
            snapshot.edges[active_columns[i]].edge_id for i in np.flatnonzero(active_memberships > 0)
        )
        relevance_gain, coverage_gain, facility_gain = marginal_parts[pool_index]
        provenance.append(
            CandidateProvenance(
                memory_id=snapshot.memory_ids[local_position],
                semantic_edge_ids=semantic_edges,
                history_edge_ids=history_edge_ids_by_local.get(local_position, ()),
                x0=float(x0[local_position]),
                x_sem=float(x_sem[local_position]),
                x_hist=float(x_hist[local_position]),
                agreement=float(agreement[local_position]),
                recency=float(recency[local_position]),
                relevance_marginal=relevance_gain,
                edge_coverage_marginal=coverage_gain,
                facility_marginal=facility_gain,
                selection_order=order,
            )
        )
    if any(not row.semantic_edge_ids for row in provenance):
        raise AssertionError("selected candidate lacks an active semantic-edge path")
    return GraphSelectionResult(
        memory_ids=[snapshot.memory_ids[i] for i in selected_local],
        local_positions=selected_local,
        provenance=provenance,
        pool_size=len(pool),
        active_edge_ids=tuple(snapshot.edges[i].edge_id for i in active_columns),
        underflow=False,
        audit={
            "graph_owned_slots": len(selected_local),
            "dense_fallback_slots": 0,
            "candidate_query_memory_similarity_calls": 0,
            "memory_ann_calls": 0,
            "celf_recomputations": evaluations,
            "objective_value": _objective(selected, relevance, incidence, edge_weights, similarity, omega, weights),
        },
    )


def exact_optimum_value(
    relevance: np.ndarray,
    incidence: np.ndarray,
    edge_weights: np.ndarray,
    budget: int,
    weights: ObjectiveWeights = ObjectiveWeights(),
) -> tuple[float, tuple[int, ...]]:
    similarity = incidence_similarity(incidence)
    omega = np.maximum(np.asarray(relevance, dtype=np.float64), 1e-6)
    omega /= omega.sum()
    best_value = -1.0
    best_choice: tuple[int, ...] = ()
    for choice in itertools.combinations(range(len(relevance)), min(budget, len(relevance))):
        value = _objective(choice, relevance, incidence, edge_weights, similarity, omega, weights)
        if value > best_value:
            best_value, best_choice = value, choice
    return best_value, best_choice
