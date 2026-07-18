from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np

from .contracts import CausalGraphSnapshot


@dataclass(frozen=True)
class TileLayout:
    name: str
    tile_by_local_position: np.ndarray
    members_by_tile: dict[int, tuple[int, ...]]
    cutoff_order: int
    source_max_arrival: int


def _pack(order: list[int], capacity: int, name: str, snapshot: CausalGraphSnapshot) -> TileLayout:
    mapping = np.full(len(snapshot.positions), -1, dtype=np.int64)
    members = {}
    for tile, start in enumerate(range(0, len(order), capacity)):
        rows = tuple(order[start : start + capacity])
        members[tile] = rows
        mapping[np.asarray(rows, dtype=np.int64)] = tile
    if np.any(mapping < 0):
        raise AssertionError("layout did not place every visible memory")
    return TileLayout(name, mapping, members, snapshot.cutoff_order, snapshot.source_max_arrival)


def chronological_layout(snapshot: CausalGraphSnapshot, capacity: int = 32) -> TileLayout:
    order = sorted(range(len(snapshot.positions)), key=lambda i: (int(snapshot.arrival_orders[i]), snapshot.memory_ids[i]))
    return _pack(order, capacity, "chronological", snapshot)


def balanced_graph_layout(
    snapshot: CausalGraphSnapshot,
    events: Iterable[Mapping[str, Any]],
    *,
    excluded_query_ids: set[str],
    capacity: int = 32,
    use_semantic: bool,
    use_coretrieval: bool,
    name: str,
) -> TileLayout:
    local_by_id = {memory_id: i for i, memory_id in enumerate(snapshot.memory_ids)}
    primary = np.asarray(snapshot.incidence[:, : snapshot.coarse_edge_count].argmax(axis=1)).reshape(-1)
    adjacency: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    if use_coretrieval:
        for event in events:
            if str(event["query_id"]) in excluded_query_ids:
                continue
            if str(event["namespace_id"]) != snapshot.namespace_id:
                continue
            if int(event.get("arrival_order", event.get("cutoff_order", 0))) > snapshot.cutoff_order:
                continue
            rows = [local_by_id[str(value)] for value in event["memory_ids"] if str(value) in local_by_id]
            for left in rows:
                for right in rows:
                    if left != right:
                        adjacency[left][right] += 1.0
    remaining = set(range(len(snapshot.positions)))
    semantic_buckets: dict[int, set[int]] = defaultdict(set)
    for row in remaining:
        semantic_buckets[int(primary[row])].add(row)
    order: list[int] = []
    while remaining:
        seed = max(remaining, key=lambda i: (sum(adjacency[i].values()), -int(snapshot.arrival_orders[i]), snapshot.memory_ids[i]))
        tile = [seed]
        remaining.remove(seed)
        semantic_buckets[int(primary[seed])].discard(seed)
        while remaining and len(tile) < capacity:
            candidates: set[int] = set()
            if use_semantic:
                for row in tile:
                    candidates.update(semantic_buckets[int(primary[row])])
            if use_coretrieval:
                for row in tile:
                    candidates.update(value for value in adjacency[row] if value in remaining)
            if not candidates:
                candidates = remaining
            choice = max(
                candidates,
                key=lambda i: (
                    sum(adjacency[row].get(i, 0.0) for row in tile) if use_coretrieval else 0.0,
                    sum(int(primary[i] == primary[row]) for row in tile) if use_semantic else 0,
                    -int(snapshot.arrival_orders[i]),
                    snapshot.memory_ids[i],
                ),
            )
            tile.append(choice)
            remaining.remove(choice)
            semantic_buckets[int(primary[choice])].discard(choice)
        order.extend(tile)
    return _pack(order, capacity, name, snapshot)


def layout_metrics(layout: TileLayout, selected_local_positions: list[int]) -> dict[str, Any]:
    tiles = sorted({int(layout.tile_by_local_position[value]) for value in selected_local_positions})
    return {"activated_tiles": len(tiles), "tile_ids": tiles, "candidate_count": len(selected_local_positions)}


def candidate_identity_fingerprint(memory_ids: list[str]) -> str:
    import hashlib

    return hashlib.sha256("\n".join(memory_ids).encode("utf-8")).hexdigest()


def layout_fingerprint(layout: TileLayout, memory_ids: Sequence[str]) -> str:
    import hashlib

    if len(memory_ids) != len(layout.tile_by_local_position):
        raise ValueError("layout fingerprint memory count mismatch")
    digest = hashlib.sha256()
    digest.update(layout.name.encode("utf-8"))
    digest.update(str(layout.cutoff_order).encode("ascii"))
    for memory_id, tile_id in zip(memory_ids, layout.tile_by_local_position):
        digest.update(str(memory_id).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(int(tile_id)).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()
