from __future__ import annotations

from src.streaming_fuzzy_hierarchy import build_causal_snapshot
from src.streaming_fuzzy_hierarchy import _merge_at_compaction


def test_wide_edges_split_deterministically(synthetic_data) -> None:
    embeddings, ids, namespaces, arrivals = synthetic_data
    kwargs = dict(namespace_id="a", cutoff_order=128, max_degree_before_split=6, split_children=2, seed=99)
    left = build_causal_snapshot(embeddings, ids, namespaces, arrivals, **kwargs)
    right = build_causal_snapshot(embeddings, ids, namespaces, arrivals, **kwargs)
    assert left.health["split_count"] > 0
    assert [edge.edge_id for edge in left.edges] == [edge.edge_id for edge in right.edges]
    assert all((a.centroid == b.centroid).all() for a, b in zip(left.edges, right.edges))
    assert any(edge.parent_id is not None for edge in left.edges)


def test_merge_requires_similarity_degree_age_and_compaction() -> None:
    import numpy as np

    centroids = np.asarray([[1.0, 0.0], [0.9999, 0.01], [0.0, 1.0]], dtype=np.float32)
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
    assignments = np.asarray([[0], [0], [1], [1], [2], [2]], dtype=np.int64)
    merged, count = _merge_at_compaction(centroids, assignments, similarity_threshold=0.99, maximum_degree=4, minimum_age=2)
    assert count == 1
    assert len(merged) == 2
    blocked, count = _merge_at_compaction(centroids, assignments, similarity_threshold=0.99, maximum_degree=3, minimum_age=2)
    assert count == 0
    assert len(blocked) == 3
