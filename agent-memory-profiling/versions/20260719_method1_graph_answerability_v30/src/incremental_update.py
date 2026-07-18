from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .contracts import CausalGraphSnapshot
from .streaming_fuzzy_hierarchy import build_causal_snapshot, snapshot_fingerprint


@dataclass
class ImmutableSnapshotLedger:
    fingerprints: dict[tuple[str, int], str] = field(default_factory=dict)

    def freeze(self, snapshot: CausalGraphSnapshot) -> str:
        key = (snapshot.namespace_id, snapshot.cutoff_order)
        value = snapshot_fingerprint(snapshot)
        previous = self.fingerprints.setdefault(key, value)
        if previous != value:
            raise AssertionError("past causal snapshot was modified")
        return value

    def assert_unchanged(self, snapshot: CausalGraphSnapshot) -> None:
        key = (snapshot.namespace_id, snapshot.cutoff_order)
        if key not in self.fingerprints:
            raise KeyError("snapshot was not frozen")
        if self.fingerprints[key] != snapshot_fingerprint(snapshot):
            raise AssertionError("future insertion changed a past snapshot")


def replay_snapshot_builds(
    embeddings: np.ndarray,
    memory_ids: Sequence[str],
    namespaces: Sequence[str],
    arrivals: np.ndarray,
    namespace_id: str,
    cutoffs: Sequence[int],
    *,
    build_kwargs: dict,
) -> tuple[dict[int, CausalGraphSnapshot], list[dict[str, float | int]]]:
    snapshots = {}
    rows = []
    for cutoff in sorted(set(int(value) for value in cutoffs)):
        started = time.perf_counter_ns()
        snapshot = build_causal_snapshot(
            embeddings,
            memory_ids,
            namespaces,
            arrivals,
            namespace_id=namespace_id,
            cutoff_order=cutoff,
            **build_kwargs,
        )
        elapsed_ms = (time.perf_counter_ns() - started) / 1e6
        snapshots[cutoff] = snapshot
        rows.append(
            {
                "cutoff_order": cutoff,
                "visible_memories": len(snapshot.positions),
                "build_ms": elapsed_ms,
                "affected_semantic_edges": snapshot.coarse_edge_count,
                "prototype_drift": snapshot.health.get("prototype_drift"),
                "snapshot_hash": snapshot_fingerprint(snapshot),
            }
        )
    return snapshots, rows
