from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class EdgeDescriptor:
    edge_id: str
    parent_id: str | None
    centroid: np.ndarray
    medoid_positions: tuple[int, ...]
    degree: int
    entropy: float
    cohesion: float
    confidence: float
    version: int
    arrival_min: int
    arrival_max: int


@dataclass
class CausalGraphSnapshot:
    namespace_id: str
    cutoff_order: int
    positions: np.ndarray
    memory_ids: tuple[str, ...]
    embeddings: np.ndarray
    arrival_orders: np.ndarray
    incidence: sparse.csr_matrix
    edges: tuple[EdgeDescriptor, ...]
    coarse_edge_count: int
    health: dict[str, Any]
    source_max_arrival: int

    def assert_causal(self) -> None:
        if len(self.positions) != len(self.memory_ids):
            raise AssertionError("snapshot position/ID mismatch")
        if self.source_max_arrival > self.cutoff_order:
            raise AssertionError("future memory entered graph snapshot")
        if self.incidence.shape != (len(self.positions), len(self.edges)):
            raise AssertionError("incidence shape mismatch")
        # Coarse fuzzy memberships form the stochastic assignment. Child
        # columns duplicate parent mass as a hierarchical routing view.
        row_sums = np.asarray(self.incidence[:, : self.coarse_edge_count].sum(axis=1)).reshape(-1)
        if len(row_sums) and not np.allclose(row_sums, 1.0, atol=1e-5):
            raise AssertionError("fuzzy memberships must sum to one")
        if np.any(self.incidence.data < 0) or not np.isfinite(self.incidence.data).all():
            raise AssertionError("invalid fuzzy membership")


@dataclass(frozen=True)
class CandidateProvenance:
    memory_id: str
    semantic_edge_ids: tuple[str, ...]
    history_edge_ids: tuple[str, ...]
    x0: float
    x_sem: float
    x_hist: float
    agreement: float
    recency: float
    relevance_marginal: float
    edge_coverage_marginal: float
    facility_marginal: float
    selection_order: int
    graph_owned: bool = True


@dataclass
class GraphSelectionResult:
    memory_ids: list[str]
    local_positions: list[int]
    provenance: list[CandidateProvenance]
    pool_size: int
    active_edge_ids: tuple[str, ...]
    underflow: bool
    audit: dict[str, Any] = field(default_factory=dict)
