from __future__ import annotations

from src.causal_tile_layout import balanced_graph_layout, candidate_identity_fingerprint, chronological_layout, layout_metrics
from src.streaming_fuzzy_hierarchy import build_causal_snapshot


def test_layout_changes_tiles_not_candidate_identity(synthetic_data) -> None:
    embeddings, ids, namespaces, arrivals = synthetic_data
    snapshot = build_causal_snapshot(embeddings, ids, namespaces, arrivals, namespace_id="a", cutoff_order=64)
    selected = list(range(32))
    chronological = chronological_layout(snapshot)
    semantic = balanced_graph_layout(snapshot, (), excluded_query_ids=set(), use_semantic=True, use_coretrieval=False, name="semantic")
    assert candidate_identity_fingerprint([snapshot.memory_ids[i] for i in selected]) == candidate_identity_fingerprint([snapshot.memory_ids[i] for i in selected])
    assert layout_metrics(chronological, selected)["candidate_count"] == layout_metrics(semantic, selected)["candidate_count"] == 32
    assert chronological.source_max_arrival <= chronological.cutoff_order
    assert semantic.source_max_arrival <= semantic.cutoff_order
