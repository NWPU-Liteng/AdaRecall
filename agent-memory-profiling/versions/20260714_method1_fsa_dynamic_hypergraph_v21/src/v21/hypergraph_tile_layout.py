"""Leakage-safe logical layout for fuzzy semantic and evidence hyperedges."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Mapping, Sequence

import numpy as np


class _DisjointSet:
    def __init__(self, values: set[str]) -> None:
        self.parent = {value: value for value in values}
        self.members = {value: {value} for value in values}

    def find(self, value: str) -> str:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            parent = self.parent[value]
            self.parent[value] = root
            value = parent
        return root

    def union(self, left: str, right: str, capacity: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if len(self.members[left_root]) + len(self.members[right_root]) > capacity:
            return
        if (len(self.members[left_root]), left_root) < (len(self.members[right_root]), right_root):
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        self.members[left_root].update(self.members.pop(right_root))


@dataclass(frozen=True)
class HypergraphTileLayout:
    memory_to_tile: Mapping[str, str]
    tile_members: Mapping[str, tuple[str, ...]]
    primary_prototype: Mapping[str, int]
    outer_train_hyperedge_span_mean: float
    outer_train_hyperedge_count: int


def build_hypergraph_tile_layout(
    memory_ids: Sequence[str],
    persona_by_memory: Mapping[str, str],
    incidence: np.ndarray,
    outer_train_hyperedges: Sequence[Sequence[str]],
    *,
    capacity: int = 16,
    structural_weight: int = 8,
    structural_core_capacity: int | None = None,
    semantic_vectors: np.ndarray | None = None,
    semantic_alias_fill: bool = False,
    fallback_positions: Mapping[str, int] | None = None,
    tile_prefix: str = "v21_hyper",
) -> HypergraphTileLayout:
    """Pack train hyperedge components, then fuzzy-primary semantic members.

    Evidence hyperedges only decide components for outer-train memories. Every
    other memory is placed from frozen UOT incidence, so heldout query labels
    are unnecessary and no access/co-access trace is constructed.
    """
    ids = tuple(map(str, memory_ids))
    values = np.asarray(incidence, dtype=np.float32)
    semantic = values if semantic_vectors is None else np.asarray(semantic_vectors, dtype=np.float32)
    core_capacity = capacity if structural_core_capacity is None else int(structural_core_capacity)
    if capacity <= 0 or structural_weight <= 0:
        raise ValueError("capacity and structural weight must be positive")
    if values.ndim != 2 or values.shape[0] != len(ids):
        raise ValueError("incidence rows must align with memory IDs")
    if semantic.ndim != 2 or semantic.shape[0] != len(ids) or not 0 < core_capacity <= capacity:
        raise ValueError("semantic vectors/core capacity must align with the layout")
    if len(set(ids)) != len(ids) or set(ids) != set(map(str, persona_by_memory)):
        raise ValueError("memory IDs/persona mapping must be complete and unique")
    index_by_id = {memory_id: position for position, memory_id in enumerate(ids)}
    primary_values = np.argmax(values, axis=1).astype(np.int64)
    primary = {memory_id: int(primary_values[position]) for position, memory_id in enumerate(ids)}

    edges_by_persona: dict[str, list[tuple[str, ...]]] = defaultdict(list)
    pair_weights_by_persona: dict[str, Counter[tuple[str, str]]] = defaultdict(Counter)
    for edge_values in outer_train_hyperedges:
        edge = tuple(dict.fromkeys(map(str, edge_values)))
        if len(edge) < 2 or any(memory_id not in index_by_id for memory_id in edge):
            continue
        personas = {str(persona_by_memory[memory_id]) for memory_id in edge}
        if len(personas) != 1:
            raise ValueError("one evidence hyperedge cannot cross personas")
        persona = next(iter(personas))
        edges_by_persona[persona].append(edge)
        for pair in combinations(sorted(edge), 2):
            pair_weights_by_persona[persona][pair] += int(structural_weight)

    ids_by_persona: dict[str, list[str]] = defaultdict(list)
    for memory_id in ids:
        ids_by_persona[str(persona_by_memory[memory_id])].append(memory_id)

    assignment: dict[str, str] = {}
    tile_members: dict[str, list[str]] = {}
    tile_number = 0
    for persona in sorted(ids_by_persona, key=lambda item: (int(item) if item.isdigit() else 10**9, item)):
        persona_ids = ids_by_persona[persona]
        active_nodes = {memory_id for edge in edges_by_persona.get(persona, ()) for memory_id in edge}
        components: list[set[str]] = []
        if active_nodes:
            dsu = _DisjointSet(active_nodes)
            for (left, right), weight in sorted(
                pair_weights_by_persona[persona].items(),
                key=lambda item: (-item[1], item[0]),
            ):
                if weight > 0:
                    dsu.union(left, right, core_capacity)
            components = sorted(
                (set(component) for component in dsu.members.values()),
                key=lambda component: (-len(component), min(component)),
            )

        bins: list[list[str]] = []
        for component in components:
            component_ids = sorted(component)
            component_vector = semantic[[index_by_id[memory_id] for memory_id in component_ids]].mean(axis=0)
            feasible = [position for position, tile in enumerate(bins) if len(tile) + len(component_ids) <= core_capacity]
            if feasible:
                def bin_score(position: int) -> tuple[float, int, int]:
                    tile = bins[position]
                    tile_vector = semantic[[index_by_id[memory_id] for memory_id in tile]].mean(axis=0)
                    structural = sum(
                        pair_weights_by_persona[persona].get(tuple(sorted((left, right))), 0)
                        for left in component_ids for right in tile
                    )
                    return float(structural) + float(component_vector @ tile_vector), len(tile), -position

                bins[max(feasible, key=bin_score)].extend(component_ids)
            else:
                bins.append(component_ids)

        assigned_active = {memory_id for tile in bins for memory_id in tile}
        if semantic_alias_fill and bins:
            candidates = [memory_id for memory_id in persona_ids if memory_id not in assigned_active]
            if candidates:
                roots = np.stack([
                    semantic[[index_by_id[memory_id] for memory_id in tile]].mean(axis=0)
                    for tile in bins
                ]).astype(np.float32)
                roots /= np.maximum(np.linalg.norm(roots, axis=1, keepdims=True), 1e-12)
                candidate_matrix = semantic[[index_by_id[memory_id] for memory_id in candidates]].astype(np.float32, copy=True)
                candidate_matrix /= np.maximum(np.linalg.norm(candidate_matrix, axis=1, keepdims=True), 1e-12)
                scores = candidate_matrix @ roots.T
                preferences = np.argsort(-scores, axis=1, kind="stable")
                candidate_order = sorted(
                    range(len(candidates)),
                    key=lambda position: (-float(scores[position, preferences[position, 0]]), candidates[position]),
                )
                aliases: set[str] = set()
                for candidate_position in candidate_order:
                    for bin_position in preferences[candidate_position]:
                        target = int(bin_position)
                        if len(bins[target]) < capacity:
                            memory_id = candidates[candidate_position]
                            bins[target].append(memory_id)
                            aliases.add(memory_id)
                            break
                assigned_active.update(aliases)
        if fallback_positions is not None:
            if set(map(str, fallback_positions)) != set(ids):
                raise ValueError("fallback positions must cover every memory")
            remaining_order = sorted(
                (memory_id for memory_id in persona_ids if memory_id not in assigned_active),
                key=lambda memory_id: (int(fallback_positions[memory_id]), memory_id),
            )
            for tile in bins:
                while remaining_order and len(tile) < capacity:
                    tile.append(remaining_order.pop(0))
        else:
            remaining_by_primary: dict[int, list[str]] = defaultdict(list)
            for memory_id in persona_ids:
                if memory_id not in assigned_active:
                    remaining_by_primary[primary[memory_id]].append(memory_id)
            for prototype, members in remaining_by_primary.items():
                members.sort(key=lambda memory_id: (
                    -float(values[index_by_id[memory_id], prototype]),
                    tuple(map(int, np.argsort(-values[index_by_id[memory_id]], kind="stable")[:3])),
                    memory_id,
                ))

            # Fill any residual structural capacity with fuzzy-primary members.
            for tile in bins:
                if len(tile) >= capacity:
                    continue
                centroid = values[[index_by_id[memory_id] for memory_id in tile]].mean(axis=0)
                prototype_order = list(map(int, np.argsort(-centroid, kind="stable")))
                for prototype in prototype_order:
                    pool = remaining_by_primary.get(prototype, [])
                    while pool and len(tile) < capacity:
                        tile.append(pool.pop(0))
                    if len(tile) == capacity:
                        break

            remaining_order = []
            for prototype in range(values.shape[1]):
                remaining_order.extend(remaining_by_primary.get(prototype, []))
        for start in range(0, len(remaining_order), capacity):
            bins.append(remaining_order[start : start + capacity])

        flattened = [memory_id for tile in bins for memory_id in tile]
        if len(flattened) != len(persona_ids) or len(set(flattened)) != len(persona_ids):
            raise RuntimeError(f"layout lost or duplicated memories for persona {persona}")
        for tile in bins:
            if not tile:
                continue
            tile_id = f"{tile_prefix}_t{tile_number}"
            tile_number += 1
            tile_members[tile_id] = list(tile)
            for memory_id in tile:
                assignment[memory_id] = tile_id

    if len(assignment) != len(ids) or max(map(len, tile_members.values()), default=0) > capacity:
        raise RuntimeError("layout violates completeness or capacity")
    spans = [len({assignment[memory_id] for memory_id in edge}) for edges in edges_by_persona.values() for edge in edges]
    return HypergraphTileLayout(
        memory_to_tile=assignment,
        tile_members={tile_id: tuple(members) for tile_id, members in tile_members.items()},
        primary_prototype=primary,
        outer_train_hyperedge_span_mean=float(np.mean(spans)) if spans else 0.0,
        outer_train_hyperedge_count=len(spans),
    )


def logical_rank_block_activations(
    entries_visited_by_prototype: Sequence[int],
    *,
    entry_bytes: int = 4,
    tile_bytes: int = 65536,
) -> int:
    if entry_bytes <= 0 or tile_bytes <= 0:
        raise ValueError("entry and tile bytes must be positive")
    return int(sum((max(0, int(entries)) * entry_bytes + tile_bytes - 1) // tile_bytes for entries in entries_visited_by_prototype))
