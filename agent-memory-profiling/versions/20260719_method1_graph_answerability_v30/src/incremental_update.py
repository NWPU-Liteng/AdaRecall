from __future__ import annotations

import hashlib
import json
import os
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .contracts import CausalGraphSnapshot
from .streaming_fuzzy_hierarchy import build_causal_snapshot, dynamic_coarse_k, snapshot_fingerprint


def _unit_vector(value: np.ndarray) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32).copy()
    vector /= max(float(np.linalg.norm(vector)), 1e-12)
    return vector


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


@dataclass
class ImmutableArtifactLedger:
    """Tracks compacted snapshot receipts without retaining every graph in RAM."""

    file_hashes: dict[Path, str] = field(default_factory=dict)

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1 << 20):
                digest.update(chunk)
        return digest.hexdigest()

    def freeze(self, path: Path, payload: Mapping[str, object]) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        value = self._sha256(path)
        self.file_hashes[path] = value
        return value

    def assert_all_unchanged(self) -> None:
        for path, expected in self.file_hashes.items():
            if self._sha256(path) != expected:
                raise AssertionError(f"future write modified compacted snapshot artifact: {path}")


@dataclass
class OnlinePrototypeState:
    namespace_id: str
    dimension: int
    memberships_per_memory: int
    temperature: float
    centroids: np.ndarray = field(init=False)
    sufficient_sums: np.ndarray = field(init=False)
    sufficient_mass: np.ndarray = field(init=False)
    edge_ids: list[str] = field(default_factory=list)
    seen_count: int = 0
    seen_memory_ids: set[str] = field(default_factory=set)
    last_arrival_order: int = -1

    def __post_init__(self) -> None:
        self.centroids = np.empty((0, self.dimension), dtype=np.float32)
        self.sufficient_sums = np.empty((0, self.dimension), dtype=np.float64)
        self.sufficient_mass = np.empty(0, dtype=np.float64)

    @classmethod
    def from_snapshot(
        cls,
        snapshot: CausalGraphSnapshot,
        *,
        memberships_per_memory: int,
        temperature: float,
    ) -> "OnlinePrototypeState":
        state = cls(
            namespace_id=snapshot.namespace_id,
            dimension=snapshot.embeddings.shape[1],
            memberships_per_memory=memberships_per_memory,
            temperature=temperature,
        )
        coarse = snapshot.incidence[:, : snapshot.coarse_edge_count].tocsc()
        sums = []
        masses = []
        for edge_index in range(snapshot.coarse_edge_count):
            start, stop = coarse.indptr[edge_index : edge_index + 2]
            rows = coarse.indices[start:stop]
            weights = coarse.data[start:stop].astype(np.float64)
            mass = max(float(weights.sum()), 1e-12)
            sums.append(np.sum(snapshot.embeddings[rows].astype(np.float64) * weights[:, None], axis=0))
            masses.append(mass)
        state.sufficient_sums = np.asarray(sums, dtype=np.float64)
        state.sufficient_mass = np.asarray(masses, dtype=np.float64)
        state.centroids = np.asarray([_unit_vector(value) for value in state.sufficient_sums], dtype=np.float32)
        state.edge_ids = [edge.edge_id for edge in snapshot.edges[: snapshot.coarse_edge_count]]
        state.seen_count = len(snapshot.memory_ids)
        state.seen_memory_ids = set(snapshot.memory_ids)
        state.last_arrival_order = int(snapshot.source_max_arrival)
        return state

    def _append_prototype(self, vector: np.ndarray, edge_id: str) -> int:
        self.centroids = np.vstack((self.centroids, vector[None, :]))
        self.sufficient_sums = np.vstack((self.sufficient_sums, vector.astype(np.float64)[None, :]))
        self.sufficient_mass = np.append(self.sufficient_mass, 1.0)
        self.edge_ids.append(edge_id)
        return len(self.edge_ids) - 1

    def insert(
        self,
        *,
        memory_id: str,
        namespace_id: str,
        arrival_order: int,
        embedding: np.ndarray,
    ) -> dict[str, object]:
        if str(namespace_id) != self.namespace_id:
            raise ValueError("namespace violation in incremental prototype update")
        if memory_id in self.seen_memory_ids:
            raise ValueError("duplicate memory insertion")
        if int(arrival_order) < self.last_arrival_order:
            raise ValueError("arrival order moved backwards")
        vector = _unit_vector(embedding)
        if len(vector) != self.dimension:
            raise ValueError("embedding dimension mismatch")

        self.seen_count += 1
        target_k = dynamic_coarse_k(self.seen_count)
        if len(self.centroids) < target_k:
            edge_index = self._append_prototype(vector, f"ns{self.namespace_id}:delta-c{len(self.edge_ids)}")
            affected = [edge_index]
            weights = np.asarray([1.0], dtype=np.float64)
            drift = 0.0
        else:
            logits = self.centroids @ vector / max(float(self.temperature), 1e-4)
            count = min(int(self.memberships_per_memory), len(logits))
            affected_array = np.argpartition(-logits, count - 1)[:count]
            affected_array = affected_array[np.argsort(-logits[affected_array], kind="stable")]
            selected = logits[affected_array].astype(np.float64)
            selected -= selected.max()
            weights = np.exp(selected)
            weights /= max(float(weights.sum()), 1e-12)
            before = self.centroids[affected_array].copy()
            for edge_index, weight in zip(affected_array, weights):
                self.sufficient_sums[edge_index] += float(weight) * vector
                self.sufficient_mass[edge_index] += float(weight)
                self.centroids[edge_index] = _unit_vector(self.sufficient_sums[edge_index])
            drift = max(
                0.0,
                max(1.0 - float(np.dot(old, self.centroids[index])) for old, index in zip(before, affected_array)),
            )
            affected = affected_array.tolist()

        self.seen_memory_ids.add(memory_id)
        self.last_arrival_order = int(arrival_order)
        return {
            "affected_semantic_edge_ids": [self.edge_ids[index] for index in affected],
            "membership_weights": weights.tolist(),
            "prototype_drift_max": float(drift),
            "coarse_k_online": len(self.centroids),
        }


class DeltaTileJournal:
    """Append-only physical delta Tiles, rotated after each exact boundary rebuild."""

    HEADER = struct.Struct("<QQQ")

    def __init__(self, root: Path, *, tile_capacity: int = 32, alignment: int = 4096) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=False)
        self.tile_capacity = int(tile_capacity)
        self.alignment = int(alignment)
        self.segment_index = 0
        self.segment_entries = 0
        self.total_entries = 0
        self._handle = None
        self._path: Path | None = None

    def _ensure_open(self) -> None:
        if self._handle is None:
            self._path = self.root / f"delta_segment_{self.segment_index:04d}.bin"
            self._handle = self._path.open("xb", buffering=0)

    def append(self, memory_id: str, payload: str, embedding: np.ndarray) -> dict[str, int]:
        self._ensure_open()
        handle = self._handle
        assert handle is not None
        tile_id = self.segment_entries // self.tile_capacity
        padding_bytes = 0
        if self.segment_entries % self.tile_capacity == 0:
            padding_bytes = (-handle.tell()) % self.alignment
            if padding_bytes:
                handle.write(b"\0" * padding_bytes)
        payload_blob = str(payload).encode("utf-8")
        embedding_blob = np.asarray(embedding, dtype=np.float32).tobytes(order="C")
        metadata_blob = json.dumps(
            {"memory_id": str(memory_id), "segment_entry": self.segment_entries, "tile_id": tile_id},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        header = self.HEADER.pack(len(payload_blob), len(embedding_blob), len(metadata_blob))
        started = time.perf_counter_ns()
        handle.write(header)
        handle.write(payload_blob)
        handle.write(embedding_blob)
        handle.write(metadata_blob)
        write_latency_ns = time.perf_counter_ns() - started
        metadata_bytes = len(header) + len(metadata_blob)
        logical_bytes = len(payload_blob) + len(embedding_blob)
        physical_bytes = logical_bytes + metadata_bytes + padding_bytes
        self.segment_entries += 1
        self.total_entries += 1
        return {
            "delta_segment": self.segment_index,
            "delta_tile_id": tile_id,
            "payload_bytes": len(payload_blob),
            "embedding_bytes": len(embedding_blob),
            "metadata_bytes": metadata_bytes,
            "padding_bytes": padding_bytes,
            "logical_bytes": logical_bytes,
            "physical_write_bytes": physical_bytes,
            "write_latency_ns": write_latency_ns,
        }

    def flush_and_rotate(self) -> dict[str, int | float | str | None]:
        if self._handle is None:
            return {
                "delta_segment": None,
                "entries": 0,
                "segment_bytes": 0,
                "fsync_ms": 0.0,
                "segment_path": None,
            }
        handle = self._handle
        path = self._path
        assert path is not None
        started = time.perf_counter_ns()
        handle.flush()
        os.fsync(handle.fileno())
        fsync_ms = (time.perf_counter_ns() - started) / 1e6
        segment_bytes = handle.tell()
        handle.close()
        result = {
            "delta_segment": self.segment_index,
            "entries": self.segment_entries,
            "segment_bytes": segment_bytes,
            "fsync_ms": fsync_ms,
            "segment_path": path.name,
        }
        self.segment_index += 1
        self.segment_entries = 0
        self._handle = None
        self._path = None
        return result

    def close(self) -> None:
        if self._handle is not None:
            self.flush_and_rotate()


@dataclass
class FrozenPolicyEventLedger:
    namespace_by_memory_id: Mapping[str, str]
    event_ids: set[str] = field(default_factory=set)

    def apply(self, event: Mapping[str, object], *, current_cutoff: int) -> dict[str, object]:
        event_id = str(event["event_id"])
        namespace_id = str(event["namespace_id"])
        if event_id in self.event_ids:
            raise ValueError("duplicate replay event")
        if int(event["arrival_order"]) > int(current_cutoff):
            raise ValueError("future event entered replay")
        if str(event["event_kind"]) != "co_retrieval":
            raise ValueError("frozen-policy replay must be named co_retrieval")
        if bool(event.get("native_agent_access", False)):
            raise ValueError("frozen-policy replay cannot claim native agent access")
        memory_ids = [str(value) for value in event["memory_ids"]]
        if len(memory_ids) != len(set(memory_ids)):
            raise ValueError("duplicate memory in replay event")
        if any(self.namespace_by_memory_id.get(value) != namespace_id for value in memory_ids):
            raise ValueError("namespace violation in replay event")
        self.event_ids.add(event_id)
        return {
            "event_id": event_id,
            "event_kind": "co_retrieval",
            "native_agent_access": False,
            "affected_history_edges": 1,
            "memory_count": len(memory_ids),
        }


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
    """Compatibility helper for exact boundary rebuild tests and diagnostics."""

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
