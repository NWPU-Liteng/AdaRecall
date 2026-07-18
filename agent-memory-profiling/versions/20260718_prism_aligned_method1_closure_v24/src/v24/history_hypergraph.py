from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Iterable

import numpy as np

from .contracts import HistoryEvent


@dataclass
class HistoryBuild:
    edges: list[dict[str, Any]]
    vectors: np.ndarray
    inverted: dict[str, list[int]]
    diagnostics: dict[str, Any]


def build_history_edges(
    events: Iterable[HistoryEvent],
    memory_ids: set[str],
    query_vectors: dict[str, np.ndarray],
    *,
    max_event_size: int = 32,
) -> HistoryBuild:
    started = perf_counter()
    edges: list[dict[str, Any]] = []
    vectors: list[np.ndarray] = []
    inverted: dict[str, list[int]] = {}
    dropped = 0
    for event in events:
        members = []
        seen = set()
        for memory_id in event.selected_memory_ids:
            memory_id = str(memory_id)
            if memory_id in memory_ids and memory_id not in seen:
                seen.add(memory_id)
                members.append(memory_id)
            if len(members) >= int(max_event_size):
                break
        if len(members) < 2:
            dropped += 1
            continue
        context = query_vectors.get(event.query_embedding_id)
        if context is None:
            dropped += 1
            continue
        # Query arrays may be memory-mapped read-only; normalization must not
        # mutate the shared embedding store.
        context = np.array(context, dtype=np.float32, copy=True)
        context /= max(float(np.linalg.norm(context)), 1e-12)
        edge_id = len(edges)
        edges.append({
            "edge_id": edge_id,
            "event_id": event.event_id,
            "namespace_id": event.namespace_id,
            "query_id": event.query_id,
            "members": members,
            "support": 1,
            "task_success": event.task_success,
            "event_kind": event.event_kind,
            "reliability": float(np.log1p(len(members)) / np.log1p(max_event_size)),
            "arrival_order": int(event.arrival_order),
        })
        vectors.append(context)
        for memory_id in members:
            inverted.setdefault(memory_id, []).append(edge_id)
    matrix = np.vstack(vectors).astype(np.float32) if vectors else np.empty((0, 0), dtype=np.float32)
    sizes = [len(edge["members"]) for edge in edges]
    diagnostics = {
        "edge_count": len(edges),
        "singleton_events_dropped": int(dropped),
        "event_size_mean": float(np.mean(sizes)) if sizes else 0.0,
        "event_size_median": float(np.median(sizes)) if sizes else 0.0,
        "event_size_max": int(max(sizes)) if sizes else 0,
        "support_mean": 1.0 if edges else 0.0,
        "event_kind_counts": {kind: sum(edge["event_kind"] == kind for edge in edges) for kind in {edge["event_kind"] for edge in edges}},
        "task_success_available": bool(any(edge["task_success"] is not None for edge in edges)),
        "source_semantics": "co_candidate_only",
        "build_ms": float((perf_counter() - started) * 1000.0),
    }
    return HistoryBuild(edges, matrix, inverted, diagnostics)


class HistoryHypergraph:
    def __init__(self, edges: list[dict[str, Any]], vectors: np.ndarray, inverted: dict[str, list[int]]) -> None:
        self.edges = edges
        self.vectors = np.array(vectors, dtype=np.float32, copy=True)
        if self.vectors.size:
            self.vectors /= np.maximum(np.linalg.norm(self.vectors, axis=1, keepdims=True), 1e-12)
        self.inverted = {str(key): [int(value) for value in values] for key, values in inverted.items()}

    def active_edges(
        self,
        query_vector: np.ndarray,
        seed_ids: list[str],
        seed_scores: np.ndarray,
        *,
        top_edges: int = 32,
        tau: float = 0.10,
    ) -> tuple[list[dict[str, Any]], float]:
        started = perf_counter()
        edge_ids = sorted({edge_id for memory_id in seed_ids for edge_id in self.inverted.get(str(memory_id), [])})
        if not edge_ids:
            return [], 0.0
        query = np.array(query_vector, dtype=np.float32, copy=True).reshape(-1)
        query /= max(float(np.linalg.norm(query)), 1e-12)
        score_by_id = {str(memory_id): float(score) for memory_id, score in zip(seed_ids, seed_scores)}
        values = []
        for edge_id in edge_ids:
            edge = self.edges[edge_id]
            overlap = [score_by_id[memory_id] for memory_id in edge["members"] if memory_id in score_by_id]
            seed_mass = float(np.mean(overlap)) if overlap else 0.0
            query_match = float(1.0 / (1.0 + np.exp(-float((self.vectors[edge_id] @ query) / max(float(tau), 1e-4)))))
            activation = float(edge["reliability"]) * query_match * (1e-3 + max(seed_mass, 0.0))
            values.append((activation, query_match, seed_mass, edge_id))
        values.sort(key=lambda item: (-item[0], item[3]))
        active = []
        for activation, query_match, seed_mass, edge_id in values[: int(top_edges)]:
            edge = self.edges[edge_id]
            active.append({
                "edge_id": int(edge_id), "edge_type": "history", "activation": float(activation),
                "query_match": float(query_match), "seed_mass": float(seed_mass),
                "reliability": float(edge["reliability"]), "members": list(edge["members"]),
                "namespace_id": edge["namespace_id"],
            })
        return active, float((perf_counter() - started) * 1000.0)
