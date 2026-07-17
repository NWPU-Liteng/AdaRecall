"""Persona-sharded view of the frozen prototype rank index."""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from .sparse_semantic_index import (
    SparseSemanticEdge,
    SparseSemanticLookupRequest,
    SparseSemanticLookupResult,
)


class PersonaShardedSparseSemanticIndex:
    """Preserve global rank order while removing cross-persona scan entries."""

    def __init__(
        self,
        incidence: np.ndarray,
        persona_by_memory_position: Sequence[str],
    ) -> None:
        values = np.asarray(incidence, dtype=np.float32)
        personas = tuple(map(str, persona_by_memory_position))
        if values.ndim != 2 or values.shape[0] != len(personas):
            raise ValueError("incidence and persona rows must align")
        positions_by_persona: dict[str, np.ndarray] = {}
        for persona in sorted(set(personas), key=lambda item: (int(item) if item.isdigit() else 10**9, item)):
            positions = np.asarray([position for position, value in enumerate(personas) if value == persona], dtype=np.int32)
            positions_by_persona[persona] = positions
        shards: dict[str, np.ndarray] = {}
        for persona, positions in positions_by_persona.items():
            ranks = np.empty((values.shape[1], len(positions)), dtype=np.int32)
            for prototype in range(values.shape[1]):
                local_order = np.lexsort((positions, -values[positions, prototype]))
                ranks[prototype] = positions[local_order]
            ranks.setflags(write=False)
            shards[persona] = ranks
        self._incidence = values
        self._personas = personas
        self._shards = shards
        self._metadata_bytes = int(sum(shard.nbytes for shard in shards.values()))

    @property
    def prototype_count(self) -> int:
        return int(self._incidence.shape[1])

    @property
    def memory_count(self) -> int:
        return int(self._incidence.shape[0])

    @property
    def metadata_bytes(self) -> int:
        return self._metadata_bytes

    @property
    def shard_sizes(self) -> Mapping[str, int]:
        return {persona: int(shard.shape[1]) for persona, shard in self._shards.items()}

    def lookup(self, persona_id: str, request: SparseSemanticLookupRequest) -> SparseSemanticLookupResult:
        persona = str(persona_id)
        if persona not in self._shards:
            raise KeyError(f"unknown persona shard {persona}")
        if any(prototype >= self.prototype_count for prototype in request.active_prototype_ids):
            raise ValueError("lookup prototype outside index")
        if any(self._personas[position] != persona for position in request.eligible_memory_positions):
            raise ValueError("eligible memory crosses the requested persona shard")
        eligible_mask = np.zeros(self.memory_count, dtype=bool)
        eligible_mask[np.fromiter(request.eligible_memory_positions, dtype=np.int64)] = True
        chunk_size = max(64, min(2048, int(request.members_per_edge) * 32))
        edges: list[SparseSemanticEdge] = []
        visited_total = 0
        for prototype in request.active_prototype_ids:
            ranks = self._shards[persona][prototype]
            selected: list[int] = []
            scores: list[float] = []
            visited = 0
            while visited < len(ranks) and len(selected) < request.members_per_edge:
                stop = min(len(ranks), visited + chunk_size)
                prefix = ranks[visited:stop]
                hits_array = np.flatnonzero(eligible_mask[prefix])
                remaining = int(request.members_per_edge) - len(selected)
                if len(hits_array) > remaining:
                    hits_array = hits_array[:remaining]
                if len(hits_array):
                    chosen = prefix[hits_array].astype(np.int64, copy=False)
                    selected.extend(map(int, chosen.tolist()))
                    scores.extend(map(float, self._incidence[chosen, prototype].tolist()))
                visited = stop
                if len(selected) == request.members_per_edge:
                    visited = int(visited - (len(prefix) - int(hits_array[-1]) - 1))
                    break
            visited_total += visited
            edges.append(SparseSemanticEdge(int(prototype), tuple(selected), tuple(scores), visited))
        return SparseSemanticLookupResult(tuple(edges), len(edges), visited_total, self.metadata_bytes)


class PersonaDispatchSparseSemanticIndex:
    """Drop-in lookup adapter that derives the one legal persona from eligibility."""

    def __init__(self, index: PersonaShardedSparseSemanticIndex, persona_by_memory_position: Sequence[str]) -> None:
        self._index = index
        self._personas = tuple(map(str, persona_by_memory_position))

    @property
    def metadata_bytes(self) -> int:
        return self._index.metadata_bytes

    def lookup(self, request: SparseSemanticLookupRequest) -> SparseSemanticLookupResult:
        personas = {self._personas[position] for position in request.eligible_memory_positions}
        if len(personas) != 1:
            raise ValueError("eligible lookup must resolve to exactly one persona shard")
        return self._index.lookup(next(iter(personas)), request)


class PersonaShardedSnapshotDispatch:
    """Serving adapter for an atomically published persona-sharded overlay snapshot.

    The request remains position-based, while the immutable snapshot owns the
    persona/prototype rank streams and merges base plus overlay entries.
    """

    def __init__(self, publisher, memory_ids: Sequence[str], persona_by_memory_position: Sequence[str]) -> None:
        ids = tuple(map(str, memory_ids))
        personas = tuple(map(str, persona_by_memory_position))
        if len(ids) != len(personas) or len(ids) != len(set(ids)):
            raise ValueError("memory IDs and personas must be aligned and unique")
        self._publisher = publisher
        self._ids = ids
        self._positions = {memory_id: position for position, memory_id in enumerate(ids)}
        self._personas = personas

    @property
    def memory_count(self) -> int:
        return len(self._ids)

    @property
    def prototype_count(self) -> int:
        return int(self._publisher.current().prototype_count)

    @property
    def metadata_bytes(self) -> int:
        snapshot = self._publisher.current()
        base_entries = sum(len(ranks) for ranks in snapshot._base_ranks.values())
        overlay_entries = sum(len(ranks) for ranks in snapshot._overlay_ranks.values())
        return int((base_entries + overlay_entries) * np.dtype(np.int32).itemsize)

    def lookup(self, request: SparseSemanticLookupRequest) -> SparseSemanticLookupResult:
        if not request.eligible_memory_positions:
            raise ValueError("eligible lookup must contain at least one memory")
        if any(position < 0 or position >= len(self._ids) for position in request.eligible_memory_positions):
            raise ValueError("eligible memory position outside dispatch namespace")
        eligible_ids = frozenset(self._ids[position] for position in request.eligible_memory_positions)
        personas = {self._personas[position] for position in request.eligible_memory_positions}
        if len(personas) != 1:
            raise ValueError("eligible lookup must resolve to exactly one persona shard")
        result = self._publisher.current().lookup(next(iter(personas)), request.active_prototype_ids, eligible_ids, request.members_per_edge)
        edges = tuple(
            SparseSemanticEdge(
                edge.prototype_id,
                tuple(self._positions[memory_id] for memory_id in edge.memory_ids),
                edge.incidence_scores,
                edge.entries_visited,
            )
            for edge in result.edges
        )
        return SparseSemanticLookupResult(edges, len(edges), result.entries_visited, self.metadata_bytes)


def build_top_prototype_feature_tiles(
    incidence: np.ndarray,
    persona_by_memory_position: Sequence[str],
    *,
    replication_factor: int = 2,
    tile_capacity: int = 16,
) -> tuple[dict[tuple[int, int], str], dict[str, int]]:
    """Assign bounded feature replicas to top-incidence prototype Tiles."""
    values = np.asarray(incidence, dtype=np.float32)
    personas = tuple(map(str, persona_by_memory_position))
    if values.ndim != 2 or values.shape[0] != len(personas):
        raise ValueError("incidence and persona rows must align")
    if not 1 <= replication_factor <= values.shape[1] or tile_capacity <= 0:
        raise ValueError("invalid replication factor or Tile capacity")
    top = np.argpartition(-values, replication_factor - 1, axis=1)[:, :replication_factor]
    assignments: dict[tuple[int, int], str] = {}
    tile_counts: dict[str, int] = {}
    for persona in sorted(set(personas), key=lambda item: (int(item) if item.isdigit() else 10**9, item)):
        persona_positions = np.asarray([position for position, value in enumerate(personas) if value == persona], dtype=np.int64)
        for prototype in range(values.shape[1]):
            members = [int(position) for position in persona_positions if prototype in top[int(position)]]
            members.sort(key=lambda position: (-float(values[position, prototype]), position))
            for rank, position in enumerate(members):
                tile_id = f"feature:p{persona}:h{prototype}:t{rank // tile_capacity}"
                assignments[(prototype, position)] = tile_id
                tile_counts[tile_id] = tile_counts.get(tile_id, 0) + 1
    if len(assignments) != values.shape[0] * replication_factor:
        raise RuntimeError("feature replica assignment is incomplete")
    return assignments, tile_counts
