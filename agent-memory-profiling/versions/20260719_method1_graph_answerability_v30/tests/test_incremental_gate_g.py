from __future__ import annotations

import json

import numpy as np
import pytest

from src.causal_tile_layout import balanced_graph_layout
from src.incremental_update import (
    DeltaTileJournal,
    FrozenPolicyEventLedger,
    ImmutableArtifactLedger,
    ImmutableSnapshotLedger,
    OnlinePrototypeState,
)
from src.streaming_fuzzy_hierarchy import build_causal_snapshot


def synthetic_snapshot():
    rng = np.random.default_rng(20260719)
    embeddings = rng.normal(size=(72, 12)).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    return build_causal_snapshot(
        embeddings,
        [f"m{index:03d}" for index in range(72)],
        ["a"] * 72,
        np.arange(1, 73),
        namespace_id="a",
        cutoff_order=72,
        seed=19,
    )


def test_online_top3_updates_are_normalized_and_namespace_safe() -> None:
    state = OnlinePrototypeState("a", 8, memberships_per_memory=3, temperature=0.08)
    for index in range(40):
        vector = np.zeros(8, dtype=np.float32)
        vector[index % 8] = 1.0
        result = state.insert(
            memory_id=f"m{index}",
            namespace_id="a",
            arrival_order=index,
            embedding=vector,
        )
    assert len(result["affected_semantic_edge_ids"]) == 3
    assert sum(result["membership_weights"]) == pytest.approx(1.0)
    assert np.allclose(np.linalg.norm(state.centroids, axis=1), 1.0)
    with pytest.raises(ValueError, match="duplicate"):
        state.insert(memory_id="m39", namespace_id="a", arrival_order=40, embedding=vector)
    with pytest.raises(ValueError, match="namespace"):
        state.insert(memory_id="other", namespace_id="b", arrival_order=40, embedding=vector)


def test_delta_tile_journal_measures_physical_bytes_and_fsync(tmp_path) -> None:
    journal = DeltaTileJournal(tmp_path / "delta", tile_capacity=2, alignment=64)
    rows = [
        journal.append(f"m{index}", f"payload-{index}", np.ones(5, dtype=np.float32))
        for index in range(3)
    ]
    boundary = journal.flush_and_rotate()
    assert [row["delta_tile_id"] for row in rows] == [0, 0, 1]
    assert all(
        row["physical_write_bytes"]
        == row["payload_bytes"] + row["embedding_bytes"] + row["metadata_bytes"] + row["padding_bytes"]
        for row in rows
    )
    assert boundary["entries"] == 3
    assert boundary["segment_bytes"] == sum(row["physical_write_bytes"] for row in rows)
    assert boundary["fsync_ms"] >= 0.0


def test_exact_boundary_snapshot_and_layout_stay_immutable(tmp_path) -> None:
    snapshot = synthetic_snapshot()
    ledger = ImmutableSnapshotLedger()
    expected = ledger.freeze(snapshot)
    state = OnlinePrototypeState.from_snapshot(
        snapshot,
        memberships_per_memory=3,
        temperature=0.08,
    )
    state.insert(
        memory_id="future",
        namespace_id=snapshot.namespace_id,
        arrival_order=snapshot.cutoff_order + 1,
        embedding=snapshot.embeddings[0],
    )
    ledger.assert_unchanged(snapshot)
    assert ledger.fingerprints[(snapshot.namespace_id, snapshot.cutoff_order)] == expected

    layout = balanced_graph_layout(
        snapshot,
        (),
        excluded_query_ids=set(),
        use_semantic=True,
        use_coretrieval=False,
        name="semantic",
    )
    artifact = ImmutableArtifactLedger()
    path = tmp_path / "snapshot.json"
    artifact.freeze(
        path,
        {
            "snapshot_hash": expected,
            "tile_by_local_position": layout.tile_by_local_position.tolist(),
        },
    )
    artifact.assert_all_unchanged()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["snapshot_hash"] == expected


def test_frozen_policy_agent_replay_has_explicit_semantics() -> None:
    ledger = FrozenPolicyEventLedger({"m1": "a", "m2": "a", "m3": "b"})
    event = {
        "event_id": "q1",
        "namespace_id": "a",
        "arrival_order": 5,
        "memory_ids": ["m1", "m2"],
        "event_kind": "co_retrieval",
        "native_agent_access": False,
    }
    result = ledger.apply(event, current_cutoff=5)
    assert result["affected_history_edges"] == 1
    assert result["native_agent_access"] is False
    with pytest.raises(ValueError, match="duplicate replay"):
        ledger.apply(event, current_cutoff=5)
    with pytest.raises(ValueError, match="future"):
        ledger.apply({**event, "event_id": "q2", "arrival_order": 6}, current_cutoff=5)
    with pytest.raises(ValueError, match="namespace"):
        ledger.apply({**event, "event_id": "q3", "memory_ids": ["m3"]}, current_cutoff=5)
    with pytest.raises(ValueError, match="named co_retrieval"):
        ledger.apply({**event, "event_id": "q4", "event_kind": "co_use"}, current_cutoff=5)
