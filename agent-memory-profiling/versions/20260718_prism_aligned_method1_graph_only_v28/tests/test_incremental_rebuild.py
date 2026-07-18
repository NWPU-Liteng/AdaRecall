from __future__ import annotations

from src.incremental_update import replay_snapshot_builds
from src.streaming_fuzzy_hierarchy import snapshot_fingerprint


def test_replay_matches_same_cutoff_full_rebuild(synthetic_data) -> None:
    embeddings, ids, namespaces, arrivals = synthetic_data
    kwargs = {"seed": 17}
    first, rows = replay_snapshot_builds(embeddings, ids, namespaces, arrivals, "a", [32, 64], build_kwargs=kwargs)
    second, _ = replay_snapshot_builds(embeddings, ids, namespaces, arrivals, "a", [64], build_kwargs=kwargs)
    assert snapshot_fingerprint(first[64]) == snapshot_fingerprint(second[64])
    assert len(rows) == 2
    assert rows[-1]["visible_memories"] == 64
