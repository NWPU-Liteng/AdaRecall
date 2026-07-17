from __future__ import annotations

import numpy as np

from v21.hypergraph_tile_layout import build_hypergraph_tile_layout, logical_rank_block_activations


def test_hyperedge_members_are_colocated_without_replication() -> None:
    ids = [f"m{index}" for index in range(8)]
    persona = {memory_id: "1" for memory_id in ids}
    incidence = np.asarray([
        [0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4],
        [0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6],
    ], dtype=np.float32)
    layout = build_hypergraph_tile_layout(ids, persona, incidence, [("m0", "m4", "m5")], capacity=4)
    assert len(layout.memory_to_tile) == len(ids)
    assert len(set(layout.memory_to_tile.values())) == 2
    assert len({layout.memory_to_tile[memory_id] for memory_id in ("m0", "m4", "m5")}) == 1
    assert max(map(len, layout.tile_members.values())) <= 4
    assert sum(map(len, layout.tile_members.values())) == len(ids)


def test_layout_is_deterministic_and_persona_isolated() -> None:
    ids = ["a0", "a1", "b0", "b1"]
    persona = {"a0": "1", "a1": "1", "b0": "2", "b1": "2"}
    incidence = np.asarray([[0.9, 0.1], [0.8, 0.2], [0.9, 0.1], [0.8, 0.2]], dtype=np.float32)
    first = build_hypergraph_tile_layout(ids, persona, incidence, [], capacity=2)
    second = build_hypergraph_tile_layout(ids, persona, incidence, [], capacity=2)
    assert first.memory_to_tile == second.memory_to_tile
    for members in first.tile_members.values():
        assert len({persona[memory_id] for memory_id in members}) == 1


def test_semantic_aliases_fill_reserved_hyperedge_capacity() -> None:
    ids = [f"m{index}" for index in range(8)]
    persona = {memory_id: "1" for memory_id in ids}
    incidence = np.asarray([
        [0.9, 0.1], [0.8, 0.2], [0.85, 0.15], [0.75, 0.25],
        [0.1, 0.9], [0.2, 0.8], [0.15, 0.85], [0.25, 0.75],
    ], dtype=np.float32)
    layout = build_hypergraph_tile_layout(
        ids, persona, incidence, [("m0", "m1"), ("m4", "m5")],
        capacity=4, structural_core_capacity=2, semantic_vectors=incidence,
        semantic_alias_fill=True,
    )
    assert len(layout.tile_members) == 2
    assert sorted(map(len, layout.tile_members.values())) == [4, 4]
    assert layout.memory_to_tile["m0"] == layout.memory_to_tile["m2"]
    assert layout.memory_to_tile["m4"] == layout.memory_to_tile["m6"]


def test_rank_block_activation_counts_only_visited_prefixes() -> None:
    assert logical_rank_block_activations([0, 1, 16384, 16385], entry_bytes=4, tile_bytes=65536) == 4


def test_source_order_fallback_preserves_unassigned_locality() -> None:
    ids = [f"m{index}" for index in range(8)]
    persona = {memory_id: "1" for memory_id in ids}
    incidence = np.eye(8, dtype=np.float32)
    positions = {memory_id: index for index, memory_id in enumerate(ids)}
    layout = build_hypergraph_tile_layout(
        ids, persona, incidence, [("m0", "m4")], capacity=4,
        structural_core_capacity=4, fallback_positions=positions,
    )
    assert layout.memory_to_tile["m0"] == layout.memory_to_tile["m4"]
    assert layout.memory_to_tile["m1"] == layout.memory_to_tile["m2"]
    assert layout.memory_to_tile["m3"] == layout.memory_to_tile["m5"]
    assert layout.memory_to_tile["m3"] == layout.memory_to_tile["m6"]


if __name__ == "__main__":
    tests = sorted((name, value) for name, value in globals().items() if name.startswith("test_") and callable(value))
    for name, test in tests:
        test()
        print(f"PASS {name}")
    print(f"{len(tests)} passed")
