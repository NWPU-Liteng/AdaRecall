"""Content-update overlay for the static FSA sparse semantic index.

This module implements the update mechanics allowed by the v21 dynamic
feasibility audit. It does not infer an access stream or a causal insertion
order. New memory embeddings must be projected and assigned prototype
incidence by the frozen upstream model before they are passed here.
"""
from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class IncrementalSparseEdge:
    prototype_id: int
    memory_ids: tuple[str, ...]
    incidence_scores: tuple[float, ...]
    entries_visited: int


@dataclass(frozen=True)
class IncrementalSparseLookupResult:
    edges: tuple[IncrementalSparseEdge, ...]
    entries_visited: int
    snapshot_version: int


@dataclass(frozen=True)
class CompactedSparseIndex:
    memory_ids: tuple[str, ...]
    incidence: np.ndarray
    prototype_memory_rank_index: np.ndarray


class IncrementalSparseSnapshot:
    """Immutable query view over a frozen base index plus a small overlay."""

    def __init__(
        self,
        *,
        version: int,
        base_memory_ids: Sequence[str],
        base_incidence: np.ndarray,
        base_rank_index: np.ndarray,
        overlay_incidence: Mapping[str, np.ndarray],
        tombstones: frozenset[str],
    ) -> None:
        base_ids = tuple(map(str, base_memory_ids))
        incidence = np.asarray(base_incidence, dtype=np.float32)
        ranks = np.asarray(base_rank_index)
        if incidence.ndim != 2 or ranks.shape != (incidence.shape[1], incidence.shape[0]):
            raise ValueError("base incidence/rank shapes do not align")
        if len(base_ids) != incidence.shape[0] or len(set(base_ids)) != len(base_ids):
            raise ValueError("base memory IDs must be unique and align with incidence")
        overlay: dict[str, np.ndarray] = {}
        for memory_id, values in overlay_incidence.items():
            key = str(memory_id)
            vector = np.array(values, dtype=np.float32, copy=True)
            if vector.shape != (incidence.shape[1],) or not np.isfinite(vector).all() or np.any(vector < 0.0):
                raise ValueError("overlay incidence must be finite, nonnegative, and prototype-aligned")
            vector.setflags(write=False)
            overlay[key] = vector
        if set(base_ids).intersection(overlay):
            raise ValueError("overlay IDs cannot replace the frozen base")
        if incidence.flags.writeable or not incidence.flags.c_contiguous:
            incidence = np.array(incidence, copy=True)
        if ranks.flags.writeable or ranks.dtype != np.int64 or not ranks.flags.c_contiguous:
            ranks = np.array(ranks, dtype=np.int64, copy=True)
        incidence.setflags(write=False)
        ranks.setflags(write=False)
        self.version = int(version)
        self.base_memory_ids = base_ids
        self.base_incidence = incidence
        self.base_rank_index = ranks
        self.overlay_incidence = MappingProxyType(overlay)
        self.tombstones = frozenset(map(str, tombstones))
        self._overlay_ranks = tuple(
            tuple(sorted(overlay, key=lambda memory_id: (-float(overlay[memory_id][prototype]), memory_id)))
            for prototype in range(incidence.shape[1])
        )

    @property
    def prototype_count(self) -> int:
        return int(self.base_incidence.shape[1])

    def lookup(
        self,
        active_prototypes: Sequence[int],
        eligible_memory_ids: frozenset[str],
        members_per_prototype: int,
    ) -> IncrementalSparseLookupResult:
        """Merge bounded base-rank and overlay-rank streams for each prototype."""
        if members_per_prototype <= 0:
            raise ValueError("members_per_prototype must be positive")
        eligible = frozenset(map(str, eligible_memory_ids))
        total_visited = 0
        edges: list[IncrementalSparseEdge] = []
        for prototype_value in dict.fromkeys(map(int, active_prototypes)):
            prototype = int(prototype_value)
            if not 0 <= prototype < self.prototype_count:
                raise ValueError("active prototype is outside the index")
            base_cursor = 0
            overlay_cursor = 0
            visited = 0

            def next_base() -> tuple[str, float] | None:
                nonlocal base_cursor, visited
                ranks = self.base_rank_index[prototype]
                while base_cursor < len(ranks):
                    position = int(ranks[base_cursor])
                    base_cursor += 1
                    visited += 1
                    memory_id = self.base_memory_ids[position]
                    if memory_id not in self.tombstones and memory_id in eligible:
                        return memory_id, float(self.base_incidence[position, prototype])
                return None

            def next_overlay() -> tuple[str, float] | None:
                nonlocal overlay_cursor, visited
                ranks = self._overlay_ranks[prototype]
                while overlay_cursor < len(ranks):
                    memory_id = ranks[overlay_cursor]
                    overlay_cursor += 1
                    visited += 1
                    if memory_id not in self.tombstones and memory_id in eligible:
                        return memory_id, float(self.overlay_incidence[memory_id][prototype])
                return None

            base_item = next_base()
            overlay_item = next_overlay()
            selected: list[tuple[str, float]] = []
            while len(selected) < members_per_prototype and (base_item is not None or overlay_item is not None):
                if base_item is None:
                    use_overlay = True
                elif overlay_item is None:
                    use_overlay = False
                else:
                    use_overlay = (-overlay_item[1], overlay_item[0]) < (-base_item[1], base_item[0])
                if use_overlay:
                    selected.append(overlay_item)  # type: ignore[arg-type]
                    overlay_item = next_overlay()
                else:
                    selected.append(base_item)  # type: ignore[arg-type]
                    base_item = next_base()
            total_visited += visited
            edges.append(IncrementalSparseEdge(
                prototype_id=prototype,
                memory_ids=tuple(item[0] for item in selected),
                incidence_scores=tuple(item[1] for item in selected),
                entries_visited=visited,
            ))
        return IncrementalSparseLookupResult(tuple(edges), total_visited, self.version)

    def compact(self) -> CompactedSparseIndex:
        """Offline rebuild that folds overlay entries and tombstones into a new base."""
        active_base = [
            position for position, memory_id in enumerate(self.base_memory_ids)
            if memory_id not in self.tombstones
        ]
        overlay_ids = sorted(memory_id for memory_id in self.overlay_incidence if memory_id not in self.tombstones)
        memory_ids = tuple(self.base_memory_ids[position] for position in active_base) + tuple(overlay_ids)
        blocks: list[np.ndarray] = []
        if active_base:
            blocks.append(self.base_incidence[np.asarray(active_base, dtype=np.int64)])
        if overlay_ids:
            blocks.append(np.stack([self.overlay_incidence[memory_id] for memory_id in overlay_ids]))
        incidence = np.concatenate(blocks, axis=0).astype(np.float32, copy=False) if blocks else np.empty((0, self.prototype_count), dtype=np.float32)
        ranks = np.empty((self.prototype_count, len(memory_ids)), dtype=np.int64)
        for prototype in range(self.prototype_count):
            ranks[prototype] = np.asarray(sorted(
                range(len(memory_ids)),
                key=lambda position: (-float(incidence[position, prototype]), memory_ids[position]),
            ), dtype=np.int64)
        return CompactedSparseIndex(memory_ids, incidence, ranks)


class IncrementalSparseOverlay:
    """Mutable staging area; queries only see explicitly published snapshots."""

    def __init__(self, base_memory_ids: Sequence[str], base_incidence: np.ndarray, base_rank_index: np.ndarray) -> None:
        self._base_memory_ids = tuple(map(str, base_memory_ids))
        self._base_id_set = frozenset(self._base_memory_ids)
        self._base_incidence = np.array(base_incidence, dtype=np.float32, copy=True)
        self._base_rank_index = np.array(base_rank_index, dtype=np.int64, copy=True)
        self._base_incidence.setflags(write=False)
        self._base_rank_index.setflags(write=False)
        self._overlay: dict[str, np.ndarray] = {}
        self._tombstones: set[str] = set()
        self._version = 0

    def upsert(self, memory_id: str, prototype_incidence: np.ndarray) -> None:
        key = str(memory_id)
        if key in self._base_id_set:
            raise ValueError("the frozen base cannot be replaced; delete and append under a new memory ID")
        vector = np.asarray(prototype_incidence, dtype=np.float32)
        prototype_count = int(self._base_incidence.shape[1])
        if vector.shape != (prototype_count,) or not np.isfinite(vector).all() or np.any(vector < 0.0):
            raise ValueError("prototype incidence must be finite, nonnegative, and prototype-aligned")
        self._overlay[key] = np.array(vector, copy=True)
        self._tombstones.discard(key)

    def delete(self, memory_id: str) -> None:
        key = str(memory_id)
        if key not in self._base_id_set and key not in self._overlay:
            raise KeyError(key)
        self._tombstones.add(key)

    def snapshot(self) -> IncrementalSparseSnapshot:
        self._version += 1
        return IncrementalSparseSnapshot(
            version=self._version,
            base_memory_ids=self._base_memory_ids,
            base_incidence=self._base_incidence,
            base_rank_index=self._base_rank_index,
            overlay_incidence=self._overlay,
            tombstones=frozenset(self._tombstones),
        )


class AtomicSparseSnapshotPublisher:
    """Lock-protected publication gives each query one consistent index view."""

    def __init__(self, initial: IncrementalSparseSnapshot) -> None:
        self._lock = Lock()
        self._snapshot = initial

    def current(self) -> IncrementalSparseSnapshot:
        with self._lock:
            return self._snapshot

    def publish(self, snapshot: IncrementalSparseSnapshot) -> None:
        with self._lock:
            if snapshot.version <= self._snapshot.version:
                raise ValueError("snapshot versions must increase monotonically")
            self._snapshot = snapshot


class AppendOnlyLogicalTileOverlay:
    """Stable logical placement for appended payloads until offline compaction."""

    def __init__(self, *, capacity: int = 16, tile_prefix: str = "overlay") -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self.tile_prefix = str(tile_prefix)
        self._assignments: dict[str, str] = {}
        self._next_ordinal = 0

    def assign(self, memory_id: str) -> str:
        key = str(memory_id)
        if key not in self._assignments:
            tile = f"{self.tile_prefix}_{self._next_ordinal // self.capacity:08d}"
            self._assignments[key] = tile
            self._next_ordinal += 1
        return self._assignments[key]

    def snapshot(self) -> Mapping[str, str]:
        return MappingProxyType(dict(self._assignments))


@dataclass(frozen=True)
class CompactedPersonaShardedIndex:
    memory_ids: tuple[str, ...]
    persona_ids: tuple[str, ...]
    incidence: np.ndarray


class PersonaShardedIncrementalSnapshot:
    """Immutable append/delete view with one rank stream per persona and prototype."""

    def __init__(self, *, version: int, base_memory_ids: Sequence[str], base_persona_ids: Sequence[str], base_incidence: np.ndarray, overlay_entries: Mapping[str, tuple[str, np.ndarray]], tombstones: frozenset[str]) -> None:
        base_ids = tuple(map(str, base_memory_ids))
        personas = tuple(map(str, base_persona_ids))
        incidence = np.asarray(base_incidence, dtype=np.float32)
        if incidence.ndim != 2 or len(base_ids) != len(personas) or len(base_ids) != len(incidence):
            raise ValueError("base IDs, personas, and incidence must align")
        if len(set(base_ids)) != len(base_ids):
            raise ValueError("base memory IDs must be unique")
        overlay: dict[str, tuple[str, np.ndarray]] = {}
        for memory_id, (persona_id, values) in overlay_entries.items():
            vector = np.array(values, dtype=np.float32, copy=True)
            if vector.shape != (incidence.shape[1],) or not np.isfinite(vector).all() or np.any(vector < 0.0):
                raise ValueError("overlay incidence must be finite, nonnegative, and prototype-aligned")
            vector.setflags(write=False)
            overlay[str(memory_id)] = (str(persona_id), vector)
        if set(base_ids).intersection(overlay):
            raise ValueError("overlay IDs cannot replace the frozen base")
        incidence = np.array(incidence, copy=True)
        incidence.setflags(write=False)
        self.version = int(version)
        self.base_memory_ids = base_ids
        self.base_persona_ids = personas
        self.base_incidence = incidence
        self.overlay_entries = MappingProxyType(overlay)
        self.tombstones = frozenset(map(str, tombstones))
        self._base_persona_by_id = dict(zip(base_ids, personas, strict=True))
        self._base_position_by_id = {memory_id: position for position, memory_id in enumerate(base_ids)}
        self._base_ranks = {
            persona: tuple(
                tuple(sorted(
                    (memory_id for memory_id, value in zip(base_ids, personas, strict=True) if value == persona),
                    key=lambda memory_id: (-float(incidence[self._base_position_by_id[memory_id], prototype]), memory_id),
                ))
                for prototype in range(incidence.shape[1])
            )
            for persona in sorted(set(personas))
        }
        self._overlay_ranks = {
            persona: tuple(
                tuple(sorted(
                    (memory_id for memory_id, (value, _incidence) in overlay.items() if value == persona),
                    key=lambda memory_id: (-float(overlay[memory_id][1][prototype]), memory_id),
                ))
                for prototype in range(incidence.shape[1])
            )
            for persona in sorted({persona for persona, _values in overlay.values()})
        }

    @property
    def prototype_count(self) -> int:
        return int(self.base_incidence.shape[1])

    def lookup(self, persona_id: str, active_prototypes: Sequence[int], eligible_memory_ids: frozenset[str], members_per_prototype: int) -> IncrementalSparseLookupResult:
        persona = str(persona_id)
        if members_per_prototype <= 0:
            raise ValueError("members_per_prototype must be positive")
        eligible = frozenset(map(str, eligible_memory_ids))
        for memory_id in eligible:
            actual = self._base_persona_by_id.get(memory_id)
            if actual is None and memory_id in self.overlay_entries:
                actual = self.overlay_entries[memory_id][0]
            if actual is not None and actual != persona:
                raise ValueError("eligible memory crosses the requested persona shard")
        edges, total_visited = [], 0
        empty = tuple(() for _ in range(self.prototype_count))
        for prototype in dict.fromkeys(map(int, active_prototypes)):
            if not 0 <= prototype < self.prototype_count:
                raise ValueError("active prototype is outside the index")
            base = self._base_ranks.get(persona, empty)[prototype]
            overlay = self._overlay_ranks.get(persona, empty)[prototype]
            base_cursor = overlay_cursor = visited = 0

            def next_base() -> tuple[str, float] | None:
                nonlocal base_cursor, visited
                while base_cursor < len(base):
                    memory_id = base[base_cursor]
                    base_cursor += 1
                    visited += 1
                    if memory_id not in self.tombstones and memory_id in eligible:
                        position = self._base_position_by_id[memory_id]
                        return memory_id, float(self.base_incidence[position, prototype])
                return None

            def next_overlay() -> tuple[str, float] | None:
                nonlocal overlay_cursor, visited
                while overlay_cursor < len(overlay):
                    memory_id = overlay[overlay_cursor]
                    overlay_cursor += 1
                    visited += 1
                    if memory_id not in self.tombstones and memory_id in eligible:
                        return memory_id, float(self.overlay_entries[memory_id][1][prototype])
                return None

            base_item, overlay_item = next_base(), next_overlay()
            selected: list[tuple[str, float]] = []
            while len(selected) < members_per_prototype and (base_item is not None or overlay_item is not None):
                use_overlay = base_item is None or (overlay_item is not None and (-overlay_item[1], overlay_item[0]) < (-base_item[1], base_item[0]))
                if use_overlay:
                    selected.append(overlay_item)  # type: ignore[arg-type]
                    overlay_item = next_overlay()
                else:
                    selected.append(base_item)  # type: ignore[arg-type]
                    base_item = next_base()
            total_visited += visited
            edges.append(IncrementalSparseEdge(prototype, tuple(item[0] for item in selected), tuple(item[1] for item in selected), visited))
        return IncrementalSparseLookupResult(tuple(edges), total_visited, self.version)

    def compact(self) -> CompactedPersonaShardedIndex:
        base_rows = [(memory_id, persona, self.base_incidence[position]) for position, (memory_id, persona) in enumerate(zip(self.base_memory_ids, self.base_persona_ids, strict=True)) if memory_id not in self.tombstones]
        overlay_rows = [(memory_id, persona, values) for memory_id, (persona, values) in sorted(self.overlay_entries.items()) if memory_id not in self.tombstones]
        rows = base_rows + overlay_rows
        memory_ids = tuple(row[0] for row in rows)
        persona_ids = tuple(row[1] for row in rows)
        incidence = np.stack([row[2] for row in rows]).astype(np.float32) if rows else np.empty((0, self.prototype_count), dtype=np.float32)
        return CompactedPersonaShardedIndex(memory_ids, persona_ids, incidence)


class PersonaShardedIncrementalOverlay:
    """Mutable staging area for persona-prototype shard publication."""

    def __init__(self, base_memory_ids: Sequence[str], base_persona_ids: Sequence[str], base_incidence: np.ndarray) -> None:
        self._base_memory_ids = tuple(map(str, base_memory_ids))
        self._base_persona_ids = tuple(map(str, base_persona_ids))
        self._base_id_set = frozenset(self._base_memory_ids)
        self._base_incidence = np.array(base_incidence, dtype=np.float32, copy=True)
        if len(self._base_memory_ids) != len(self._base_persona_ids) or len(self._base_memory_ids) != len(self._base_incidence):
            raise ValueError("base IDs, personas, and incidence must align")
        self._overlay: dict[str, tuple[str, np.ndarray]] = {}
        self._tombstones: set[str] = set()
        self._version = 0

    def upsert(self, memory_id: str, persona_id: str, prototype_incidence: np.ndarray) -> None:
        key = str(memory_id)
        if key in self._base_id_set:
            raise ValueError("the frozen base cannot be replaced; delete and append under a new memory ID")
        vector = np.asarray(prototype_incidence, dtype=np.float32)
        if vector.shape != (self._base_incidence.shape[1],) or not np.isfinite(vector).all() or np.any(vector < 0.0):
            raise ValueError("prototype incidence must be finite, nonnegative, and prototype-aligned")
        self._overlay[key] = (str(persona_id), np.array(vector, copy=True))
        self._tombstones.discard(key)

    def delete(self, memory_id: str) -> None:
        key = str(memory_id)
        if key not in self._base_id_set and key not in self._overlay:
            raise KeyError(key)
        self._tombstones.add(key)

    def snapshot(self) -> PersonaShardedIncrementalSnapshot:
        self._version += 1
        return PersonaShardedIncrementalSnapshot(version=self._version, base_memory_ids=self._base_memory_ids, base_persona_ids=self._base_persona_ids, base_incidence=self._base_incidence, overlay_entries=self._overlay, tombstones=frozenset(self._tombstones))


class AtomicPersonaShardedSnapshotPublisher:
    def __init__(self, initial: PersonaShardedIncrementalSnapshot) -> None:
        self._lock = Lock()
        self._snapshot = initial

    def current(self) -> PersonaShardedIncrementalSnapshot:
        with self._lock:
            return self._snapshot

    def publish(self, snapshot: PersonaShardedIncrementalSnapshot) -> None:
        with self._lock:
            if snapshot.version <= self._snapshot.version:
                raise ValueError("snapshot versions must increase monotonically")
            self._snapshot = snapshot
