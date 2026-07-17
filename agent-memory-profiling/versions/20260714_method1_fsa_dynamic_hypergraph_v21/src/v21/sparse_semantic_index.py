from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .eligibility import FORBIDDEN_INFERENCE_FIELDS


@dataclass(frozen=True)
class SparseSemanticLookupRequest:
    """Online request containing only label-free sparse-index inputs."""

    active_prototype_ids: tuple[int, ...]
    eligible_memory_positions: frozenset[int]
    members_per_edge: int

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any], prototype_count: int, memory_count: int) -> "SparseSemanticLookupRequest":
        forbidden = sorted(FORBIDDEN_INFERENCE_FIELDS.intersection(row.keys()))
        extra = sorted({"query_id", "gold_bundle", "bundle_id"}.intersection(row.keys()))
        if forbidden or extra:
            raise ValueError(f"sparse semantic lookup contains supervision-only fields: {forbidden + extra}")
        active = tuple(map(int, row["active_prototype_ids"]))
        eligible = frozenset(map(int, row["eligible_memory_positions"]))
        members = int(row["members_per_edge"])
        if not active or len(active) != len(set(active)) or not all(0 <= value < prototype_count for value in active):
            raise ValueError("active prototypes must be unique valid indices")
        if members < 1:
            raise ValueError("members_per_edge must be positive")
        if not eligible or not all(0 <= value < memory_count for value in eligible):
            raise ValueError("eligible memory positions must be nonempty valid indices")
        return cls(active, eligible, members)


@dataclass(frozen=True)
class SparseSemanticEdge:
    prototype_id: int
    memory_positions: tuple[int, ...]
    incidence_scores: tuple[float, ...]
    rank_entries_visited: int


@dataclass(frozen=True)
class SparseSemanticLookupResult:
    edges: tuple[SparseSemanticEdge, ...]
    activated_semantic_edges: int
    incidence_entries_visited: int
    metadata_bytes: int


def build_prototype_rank_index(incidence: np.ndarray) -> np.ndarray:
    """Offline-only prototype -> memory rank lists, with deterministic tie ordering."""
    values = np.asarray(incidence, dtype=np.float32)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("incidence must be a finite nonnegative [memories, prototypes] matrix")
    memory_positions = np.arange(values.shape[0], dtype=np.int32)
    rankings = np.empty((values.shape[1], values.shape[0]), dtype=np.int32)
    for prototype in range(values.shape[1]):
        # lexsort applies the index as a deterministic ascending tie-breaker.
        rankings[prototype] = np.lexsort((memory_positions, -values[:, prototype])).astype(np.int32, copy=False)
    return rankings


class SparseSemanticInvertedIndex:
    """Reads bounded prefix segments of prototype rank lists at online inference."""

    def __init__(self, rank_index: np.ndarray, incidence: np.ndarray):
        ranks = np.asarray(rank_index, dtype=np.int32)
        values = np.asarray(incidence, dtype=np.float32)
        if ranks.ndim != 2 or values.ndim != 2 or ranks.shape != (values.shape[1], values.shape[0]):
            raise ValueError("rank index shape must be [prototypes, memories] for the incidence matrix")
        if np.any(ranks < 0) or np.any(ranks >= values.shape[0]):
            raise ValueError("rank index contains an invalid memory position")
        self._ranks = ranks
        self._incidence = values

    @property
    def prototype_count(self) -> int:
        return int(self._ranks.shape[0])

    @property
    def memory_count(self) -> int:
        return int(self._ranks.shape[1])

    @property
    def metadata_bytes(self) -> int:
        return int(self._ranks.nbytes)

    def lookup(self, request: SparseSemanticLookupRequest) -> SparseSemanticLookupResult:
        if any(prototype >= self.prototype_count for prototype in request.active_prototype_ids):
            raise ValueError("lookup prototype outside index")
        # Build only a label-free eligibility membership mask.  Rank prefixes
        # are consumed in bounded chunks and the loop stops as soon as the
        # requested number of eligible members is found; no incidence matrix
        # scan is performed here.
        eligible_mask = np.zeros(self.memory_count, dtype=bool)
        eligible_mask[np.fromiter(request.eligible_memory_positions, dtype=np.int64)] = True
        chunk_size = max(64, min(2048, int(request.members_per_edge) * 32))
        edges = []
        visited_total = 0
        for prototype in request.active_prototype_ids:
            ranks = self._ranks[prototype]
            selected: list[int] = []
            scores: list[float] = []
            visited = 0
            while visited < len(ranks) and len(selected) < request.members_per_edge:
                stop = min(len(ranks), visited + chunk_size)
                prefix = ranks[visited:stop]
                hits = np.flatnonzero(eligible_mask[prefix])
                remaining = int(request.members_per_edge) - len(selected)
                if len(hits) > remaining:
                    hits = hits[:remaining]
                if len(hits):
                    chosen = prefix[hits].astype(np.int64, copy=False)
                    selected.extend(map(int, chosen.tolist()))
                    scores.extend(map(float, self._incidence[chosen, prototype].tolist()))
                visited = stop
                if len(selected) == request.members_per_edge:
                    # The counter is exact even when the final vectorized
                    # chunk contains additional ineligible rank entries.
                    visited = int(visited - (len(prefix) - int(hits[-1]) - 1))
                    break
            visited_total += visited
            edges.append(SparseSemanticEdge(int(prototype), tuple(selected), tuple(scores), visited))
        return SparseSemanticLookupResult(tuple(edges), len(edges), visited_total, self.metadata_bytes)
