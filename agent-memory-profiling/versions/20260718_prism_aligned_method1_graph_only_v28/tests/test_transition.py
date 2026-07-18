from __future__ import annotations

import numpy as np

from src.query_edge_alignment import align_query_to_edges
from src.semantic_propagation import transition_row_sums
from src.streaming_fuzzy_hierarchy import build_causal_snapshot


def test_active_transition_is_row_stochastic(synthetic_data) -> None:
    embeddings, ids, namespaces, arrivals = synthetic_data
    snapshot = build_causal_snapshot(embeddings, ids, namespaces, arrivals, namespace_id="a", cutoff_order=96)
    activation = align_query_to_edges(snapshot, embeddings[2], embeddings)
    row_sums = transition_row_sums(snapshot, activation.active_columns)
    assert len(row_sums) > 0
    assert np.allclose(row_sums, 1.0, atol=1e-5)
