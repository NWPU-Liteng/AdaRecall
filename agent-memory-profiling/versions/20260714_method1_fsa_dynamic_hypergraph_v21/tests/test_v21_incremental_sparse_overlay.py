from __future__ import annotations

import numpy as np
from unittest import TestCase

from v21.incremental_sparse_overlay import (
    AppendOnlyLogicalTileOverlay,
    AtomicPersonaShardedSnapshotPublisher,
    AtomicSparseSnapshotPublisher,
    IncrementalSparseOverlay,
    PersonaShardedIncrementalOverlay,
)
from v21.persona_sharded_index import PersonaShardedSnapshotDispatch, PersonaShardedSparseSemanticIndex
from v21.sparse_semantic_index import SparseSemanticLookupRequest


def fixture_index() -> tuple[list[str], np.ndarray, np.ndarray]:
    ids = ["m0", "m1", "m2"]
    incidence = np.asarray([[0.9, 0.1], [0.7, 0.8], [0.2, 0.6]], dtype=np.float32)
    ranks = np.asarray([[0, 1, 2], [1, 2, 0]], dtype=np.int64)
    return ids, incidence, ranks


def test_overlay_merges_by_incidence_and_honors_eligibility() -> None:
    ids, incidence, ranks = fixture_index()
    overlay = IncrementalSparseOverlay(ids, incidence, ranks)
    overlay.upsert("m3", np.asarray([0.8, 0.95], dtype=np.float32))
    snapshot = overlay.snapshot()
    result = snapshot.lookup([0, 1], frozenset({"m0", "m1", "m2", "m3"}), 3)
    assert result.edges[0].memory_ids == ("m0", "m3", "m1")
    assert result.edges[1].memory_ids == ("m3", "m1", "m2")
    eligible = snapshot.lookup([0], frozenset({"m1", "m3"}), 2)
    assert eligible.edges[0].memory_ids == ("m3", "m1")
    assert eligible.entries_visited > 0


def test_tombstone_snapshot_isolation_and_atomic_publication() -> None:
    ids, incidence, ranks = fixture_index()
    overlay = IncrementalSparseOverlay(ids, incidence, ranks)
    first = overlay.snapshot()
    overlay.delete("m0")
    second = overlay.snapshot()
    eligible = frozenset(ids)
    assert first.lookup([0], eligible, 1).edges[0].memory_ids == ("m0",)
    assert second.lookup([0], eligible, 1).edges[0].memory_ids == ("m1",)
    publisher = AtomicSparseSnapshotPublisher(first)
    publisher.publish(second)
    assert publisher.current().version == second.version
    with TestCase().assertRaises(ValueError):
        publisher.publish(first)


def test_compaction_reproduces_overlay_lookup() -> None:
    ids, incidence, ranks = fixture_index()
    overlay = IncrementalSparseOverlay(ids, incidence, ranks)
    overlay.upsert("m3", np.asarray([0.8, 0.95], dtype=np.float32))
    overlay.delete("m1")
    before = overlay.snapshot()
    compacted = before.compact()
    rebuilt = IncrementalSparseOverlay(compacted.memory_ids, compacted.incidence, compacted.prototype_memory_rank_index).snapshot()
    eligible = frozenset(compacted.memory_ids)
    before_edges = before.lookup([0, 1], eligible, 3).edges
    rebuilt_edges = rebuilt.lookup([0, 1], eligible, 3).edges
    assert [edge.memory_ids for edge in before_edges] == [edge.memory_ids for edge in rebuilt_edges]
    assert np.allclose(
        [score for edge in before_edges for score in edge.incidence_scores],
        [score for edge in rebuilt_edges for score in edge.incidence_scores],
    )


def test_frozen_base_cannot_be_replaced_and_tiles_are_stable() -> None:
    ids, incidence, ranks = fixture_index()
    overlay = IncrementalSparseOverlay(ids, incidence, ranks)
    with TestCase().assertRaises(ValueError):
        overlay.upsert("m0", np.asarray([0.1, 0.1], dtype=np.float32))
    tiles = AppendOnlyLogicalTileOverlay(capacity=2)
    assert tiles.assign("m3") == "overlay_00000000"
    assert tiles.assign("m4") == "overlay_00000000"
    assert tiles.assign("m5") == "overlay_00000001"
    assert tiles.assign("m3") == "overlay_00000000"


def test_persona_sharded_overlay_merges_only_the_requested_shard() -> None:
    ids = ("p0_a", "p0_b", "p1_a", "p1_b")
    personas = ("0", "0", "1", "1")
    incidence = np.asarray([[0.9, 0.1], [0.4, 0.8], [0.95, 0.2], [0.3, 0.7]], dtype=np.float32)
    overlay = PersonaShardedIncrementalOverlay(ids, personas, incidence)
    overlay.upsert("p0_new", "0", np.asarray([0.99, 0.5], dtype=np.float32))
    overlay.upsert("p1_new", "1", np.asarray([1.0, 1.0], dtype=np.float32))
    snapshot = overlay.snapshot()
    result = snapshot.lookup("0", [0], frozenset({"p0_a", "p0_b", "p0_new"}), 3)
    assert result.edges[0].memory_ids == ("p0_new", "p0_a", "p0_b")
    with TestCase().assertRaises(ValueError):
        snapshot.lookup("0", [0], frozenset({"p0_a", "p1_a"}), 2)


def test_persona_sharded_overlay_delete_publish_and_compact_are_exact() -> None:
    ids = ("p0_a", "p0_b", "p1_a")
    personas = ("0", "0", "1")
    incidence = np.asarray([[0.9, 0.1], [0.4, 0.8], [0.95, 0.2]], dtype=np.float32)
    overlay = PersonaShardedIncrementalOverlay(ids, personas, incidence)
    first = overlay.snapshot()
    overlay.upsert("p0_new", "0", np.asarray([0.99, 0.5], dtype=np.float32))
    overlay.delete("p0_a")
    second = overlay.snapshot()
    publisher = AtomicPersonaShardedSnapshotPublisher(first)
    publisher.publish(second)
    before = publisher.current().lookup("0", [0, 1], frozenset({"p0_a", "p0_b", "p0_new"}), 3)
    assert before.edges[0].memory_ids == ("p0_new", "p0_b")
    compacted = second.compact()
    rebuilt = PersonaShardedSparseSemanticIndex(compacted.incidence, compacted.persona_ids)
    positions = frozenset(index for index, persona in enumerate(compacted.persona_ids) if persona == "0")
    after = rebuilt.lookup("0", SparseSemanticLookupRequest((0, 1), positions, 3))
    compacted_ids = tuple(compacted.memory_ids[position] for position in after.edges[0].memory_positions)
    assert compacted_ids == before.edges[0].memory_ids


def test_persona_snapshot_dispatch_serves_published_overlay_and_maps_positions() -> None:
    ids = ("p0_a", "p0_b", "p1_a")
    personas = ("0", "0", "1")
    incidence = np.asarray([[0.9, 0.1], [0.4, 0.8], [0.95, 0.2]], dtype=np.float32)
    overlay = PersonaShardedIncrementalOverlay(ids, personas, incidence)
    initial = overlay.snapshot()
    publisher = AtomicPersonaShardedSnapshotPublisher(initial)
    dispatch = PersonaShardedSnapshotDispatch(publisher, ids, personas)
    request = SparseSemanticLookupRequest((0,), frozenset({0, 1}), 2)
    assert dispatch.lookup(request).edges[0].memory_positions == (0, 1)
    overlay.upsert("p0_new", "0", np.asarray([0.99, 0.5], dtype=np.float32))
    overlay_snapshot = overlay.snapshot()
    publisher.publish(overlay_snapshot)
    expanded_ids = ids + ("p0_new",)
    expanded_personas = personas + ("0",)
    dispatch = PersonaShardedSnapshotDispatch(publisher, expanded_ids, expanded_personas)
    expanded = dispatch.lookup(SparseSemanticLookupRequest((0,), frozenset({0, 1, 3}), 2))
    assert expanded.edges[0].memory_positions == (3, 0)
if __name__ == "__main__":
    tests = sorted((name, value) for name, value in globals().items() if name.startswith("test_") and callable(value))
    for name, test in tests:
        test()
        print(f"PASS {name}")
    print(f"{len(tests)} passed")
