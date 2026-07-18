from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .contracts import CausalGraphSnapshot


@dataclass(frozen=True)
class HistoryPropagation:
    scores: np.ndarray
    active_edge_ids_by_local: dict[int, tuple[str, ...]]
    usable_events: int
    fold_excluded_events: int
    future_excluded_events: int


def propagate_coretrieval(
    snapshot: CausalGraphSnapshot,
    x_sem: np.ndarray,
    query_embedding: np.ndarray,
    events: Iterable[Mapping[str, Any]],
    *,
    excluded_query_ids: set[str],
    query_embeddings_by_id: Mapping[str, np.ndarray] | None = None,
    minimum_support: int = 2,
    maximum_edge_size: int = 32,
    restart: float = 0.82,
) -> HistoryPropagation:
    local_by_id = {memory_id: i for i, memory_id in enumerate(snapshot.memory_ids)}
    query = np.asarray(query_embedding, dtype=np.float32).copy()
    query /= max(float(np.linalg.norm(query)), 1e-12)
    contributions = np.zeros(len(snapshot.positions), dtype=np.float64)
    edge_ids: dict[int, list[str]] = {}
    usable = excluded = future = 0
    pair_support: dict[tuple[str, str], int] = {}
    materialized = list(events)
    for event in materialized:
        if str(event["query_id"]) in excluded_query_ids:
            continue
        if str(event["namespace_id"]) != snapshot.namespace_id:
            continue
        if int(event.get("arrival_order", event.get("cutoff_order", 0))) > snapshot.cutoff_order:
            continue
        members = sorted(set(str(value) for value in event["memory_ids"]))
        for left_index, left in enumerate(members):
            for right in members[left_index + 1 :]:
                pair_support[(left, right)] = pair_support.get((left, right), 0) + 1
    for event in materialized:
        if str(event["query_id"]) in excluded_query_ids:
            excluded += 1
            continue
        if str(event["namespace_id"]) != snapshot.namespace_id:
            continue
        if int(event.get("arrival_order", event.get("cutoff_order", 0))) > snapshot.cutoff_order:
            future += 1
            continue
        members = [local_by_id[str(value)] for value in event["memory_ids"] if str(value) in local_by_id]
        if not members or len(members) > maximum_edge_size:
            continue
        member_ids = sorted(str(value) for value in event["memory_ids"] if str(value) in local_by_id)
        repeated_pairs = [
            pair_support.get((left, right), 0)
            for left_index, left in enumerate(member_ids)
            for right in member_ids[left_index + 1 :]
            if pair_support.get((left, right), 0) >= minimum_support
        ]
        if not repeated_pairs:
            continue
        semantic_overlap = float(np.asarray(x_sem)[np.asarray(members, dtype=np.int64)].sum())
        if semantic_overlap <= 0:
            continue
        context_match = 1.0
        if query_embeddings_by_id is not None and str(event["query_id"]) in query_embeddings_by_id:
            context = np.asarray(query_embeddings_by_id[str(event["query_id"])], dtype=np.float32).copy()
            context /= max(float(np.linalg.norm(context)), 1e-12)
            context_match = max(float(query @ context), 0.0)
        reliability = min(1.0, float(np.mean(np.log1p(repeated_pairs))) / np.log(4.0))
        weight = context_match * semantic_overlap * reliability / len(members)
        if weight <= 0:
            continue
        usable += 1
        event_id = str(event.get("event_id", event["query_id"]))
        for member in members:
            contributions[member] += weight
            edge_ids.setdefault(member, []).append(event_id)
    if contributions.sum() > 0:
        contributions /= contributions.sum()
    x_hist = float(restart) * np.asarray(x_sem, dtype=np.float64) + (1.0 - float(restart)) * contributions
    return HistoryPropagation(
        scores=x_hist.astype(np.float32),
        active_edge_ids_by_local={key: tuple(sorted(set(value))) for key, value in edge_ids.items()},
        usable_events=usable,
        fold_excluded_events=excluded,
        future_excluded_events=future,
    )
