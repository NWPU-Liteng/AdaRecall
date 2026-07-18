from __future__ import annotations

import numpy as np

from src.coretrieval_hypergraph import propagate_coretrieval
from src.streaming_fuzzy_hierarchy import build_causal_snapshot


def test_heldout_and_future_history_events_are_excluded(synthetic_data) -> None:
    embeddings, ids, namespaces, arrivals = synthetic_data
    snapshot = build_causal_snapshot(embeddings, ids, namespaces, arrivals, namespace_id="a", cutoff_order=64)
    x = np.full(len(snapshot.positions), 1.0 / len(snapshot.positions), dtype=np.float32)
    base = {"namespace_id": "a", "memory_ids": list(snapshot.memory_ids[:3])}
    events = [
        {**base, "query_id": "heldout", "arrival_order": 20},
        {**base, "query_id": "future", "arrival_order": 65},
        {**base, "query_id": "ok1", "arrival_order": 20},
        {**base, "query_id": "ok2", "arrival_order": 21},
    ]
    result = propagate_coretrieval(snapshot, x, embeddings[0], events, excluded_query_ids={"heldout"}, minimum_support=2)
    assert result.fold_excluded_events == 1
    assert result.future_excluded_events == 1
    assert result.usable_events == 2
