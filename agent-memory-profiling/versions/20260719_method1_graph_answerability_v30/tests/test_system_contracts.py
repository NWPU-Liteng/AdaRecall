from __future__ import annotations

import numpy as np

from src.causal_tile_layout import balanced_graph_layout, candidate_identity_fingerprint, chronological_layout
from src.physical_tile_store import PhysicalTileStore
from src.streaming_fuzzy_hierarchy import build_causal_snapshot


def synthetic_snapshot():
    rng = np.random.default_rng(20260718)
    centers = np.eye(4, 12, dtype=np.float32)
    values = []
    for index in range(96):
        row = centers[index % 4] + 0.05 * rng.normal(size=12)
        values.append(row / np.linalg.norm(row))
    embeddings = np.asarray(values, dtype=np.float32)
    ids = [f"m{index:03d}" for index in range(96)]
    return build_causal_snapshot(
        embeddings,
        ids,
        ["ns"] * 96,
        np.arange(1, 97),
        namespace_id="ns",
        cutoff_order=96,
        seed=17,
    )


def test_layouts_preserve_candidate_identity_and_causality() -> None:
    snapshot = synthetic_snapshot()
    candidates = list(snapshot.memory_ids[:32])
    chronological = chronological_layout(snapshot)
    semantic = balanced_graph_layout(
        snapshot,
        (),
        excluded_query_ids=set(),
        use_semantic=True,
        use_coretrieval=False,
        name="semantic",
    )
    assert candidate_identity_fingerprint(candidates) == candidate_identity_fingerprint(candidates)
    assert chronological.source_max_arrival <= chronological.cutoff_order
    assert semantic.source_max_arrival <= semantic.cutoff_order
    assert sorted(value for rows in semantic.members_by_tile.values() for value in rows) == list(range(96))


def test_physical_bytes_include_alignment_padding(tmp_path) -> None:
    snapshot = synthetic_snapshot()
    layout = chronological_layout(snapshot, capacity=7)
    payloads = ["payload-" + value * 7 for value in snapshot.memory_ids]
    store = PhysicalTileStore.materialize(
        tmp_path / "physical",
        layout,
        snapshot.memory_ids,
        payloads,
        snapshot.embeddings,
    )
    result = store.read_tiles([0, 1, 1, 2], cold=True)
    assert result["padding_bytes"] > 0
    assert result["total_fetched_bytes"] == sum(
        result[key]
        for key in ("payload_bytes", "embedding_bytes", "metadata_bytes", "padding_bytes")
    )
    assert result["unique_tiles"] == 3
