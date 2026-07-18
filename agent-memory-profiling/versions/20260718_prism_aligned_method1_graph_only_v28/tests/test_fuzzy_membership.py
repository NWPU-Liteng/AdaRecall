from __future__ import annotations

import numpy as np

from src.streaming_fuzzy_hierarchy import build_causal_snapshot


def test_top3_membership_is_stochastic_and_namespace_local(synthetic_data) -> None:
    embeddings, ids, namespaces, arrivals = synthetic_data
    snapshot = build_causal_snapshot(embeddings, ids, namespaces, arrivals, namespace_id="a", cutoff_order=80)
    coarse = snapshot.incidence[:, : snapshot.coarse_edge_count]
    assert snapshot.source_max_arrival == 80
    assert set(snapshot.memory_ids) == set(ids[:80])
    assert np.all(np.diff(coarse.indptr) <= 3)
    assert np.allclose(np.asarray(coarse.sum(axis=1)).reshape(-1), 1.0, atol=1e-5)
    assert np.all(coarse.data >= 0)
