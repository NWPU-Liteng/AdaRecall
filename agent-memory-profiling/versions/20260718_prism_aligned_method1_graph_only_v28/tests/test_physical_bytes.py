from __future__ import annotations

from src.causal_tile_layout import chronological_layout
from src.physical_tile_store import PhysicalTileStore
from src.streaming_fuzzy_hierarchy import build_causal_snapshot


def test_physical_store_reads_full_ranges(tmp_path, synthetic_data) -> None:
    embeddings, ids, namespaces, arrivals = synthetic_data
    snapshot = build_causal_snapshot(embeddings, ids, namespaces, arrivals, namespace_id="a", cutoff_order=40)
    layout = chronological_layout(snapshot, capacity=8)
    payloads = [f"payload-{value}" for value in snapshot.memory_ids]
    store = PhysicalTileStore.materialize(tmp_path / "store", layout, snapshot.memory_ids, payloads, snapshot.embeddings)
    result = store.read_tiles([0, 0, 1])
    expected = store.ranges[0].length + store.ranges[1].length
    assert result["unique_tiles"] == 2
    assert result["total_fetched_bytes"] == expected
    assert result["embedding_bytes"] == 16 * snapshot.embeddings.shape[1] * 4
    assert result["metadata_bytes"] > 0
