from __future__ import annotations

import numpy as np

from v21.persona_sharded_index import PersonaDispatchSparseSemanticIndex, PersonaShardedSparseSemanticIndex, build_top_prototype_feature_tiles
from v21.sparse_semantic_index import SparseSemanticInvertedIndex, SparseSemanticLookupRequest, build_prototype_rank_index


def test_persona_shard_preserves_results_and_reduces_entries_visited() -> None:
    incidence = np.asarray([
        [0.9, 0.1], [0.85, 0.2], [0.8, 0.3],
        [0.88, 0.15], [0.83, 0.25], [0.78, 0.35],
    ], dtype=np.float32)
    personas = ["1", "2", "1", "2", "1", "2"]
    global_index = SparseSemanticInvertedIndex(build_prototype_rank_index(incidence), incidence)
    sharded = PersonaShardedSparseSemanticIndex(incidence, personas)
    request = SparseSemanticLookupRequest((0, 1), frozenset({0, 2, 4}), 2)
    global_result = global_index.lookup(request)
    shard_result = sharded.lookup("1", request)
    assert [edge.memory_positions for edge in global_result.edges] == [edge.memory_positions for edge in shard_result.edges]
    assert [edge.incidence_scores for edge in global_result.edges] == [edge.incidence_scores for edge in shard_result.edges]
    assert shard_result.incidence_entries_visited < global_result.incidence_entries_visited
    assert sharded.metadata_bytes == global_index.metadata_bytes
    dispatched = PersonaDispatchSparseSemanticIndex(sharded, personas)
    assert dispatched.lookup(request).edges == shard_result.edges


def test_top_prototype_feature_replication_is_bounded() -> None:
    incidence = np.asarray([[0.9, 0.8, 0.1], [0.2, 0.7, 0.6], [0.5, 0.1, 0.9]], dtype=np.float32)
    assignments, tiles = build_top_prototype_feature_tiles(incidence, ["1", "1", "1"], replication_factor=2, tile_capacity=2)
    assert len(assignments) == 6
    assert max(tiles.values()) <= 2
    assert (0, 0) in assignments and (1, 0) in assignments
    assert (1, 1) in assignments and (2, 1) in assignments


if __name__ == "__main__":
    tests = sorted((name, value) for name, value in globals().items() if name.startswith("test_") and callable(value))
    for name, test in tests:
        test()
        print(f"PASS {name}")
    print(f"{len(tests)} passed")
