from __future__ import annotations

import numpy as np

from src.incremental_update import ImmutableSnapshotLedger
from src.streaming_fuzzy_hierarchy import build_causal_snapshot, snapshot_fingerprint


def test_future_memory_does_not_change_past_snapshot(synthetic_data) -> None:
    embeddings, ids, namespaces, arrivals = synthetic_data
    past = build_causal_snapshot(embeddings, ids, namespaces, arrivals, namespace_id="a", cutoff_order=64)
    ledger = ImmutableSnapshotLedger()
    fingerprint = ledger.freeze(past)
    changed_future = embeddings.copy()
    changed_future[64:128] = np.roll(changed_future[64:128], 1, axis=1)
    rebuilt = build_causal_snapshot(changed_future, ids, namespaces, arrivals, namespace_id="a", cutoff_order=64)
    assert snapshot_fingerprint(rebuilt) == fingerprint
    ledger.assert_unchanged(rebuilt)
