from __future__ import annotations

import numpy as np

from v24.contracts import MemoryRecord, QueryRecord
from v24.dataset import Dataset, validate_no_cross_split_ids


def test_source_contract_does_not_put_gold_in_query_record() -> None:
    query = QueryRecord.from_source({
        "query_id": "q1", "persona_id": 7, "query_session_id": 3,
        "query_text": "What happened?", "gold_memory_ids": ["m1"],
        "evidence_groups": [["m1"]], "query_type": "entity_exact_match",
    }, "test")
    assert query.query == "What happened?"
    assert query.namespace_id == "7"
    assert query.cutoff_order == 3
    assert "gold_memory_ids" not in query.to_mapping()
    assert "evidence_groups" not in query.to_mapping()


def test_causal_eligibility_is_namespace_local() -> None:
    memories = [
        MemoryRecord("m1", "a", "old", 1),
        MemoryRecord("m2", "a", "future", 4),
        MemoryRecord("m3", "b", "other namespace", 1),
    ]
    values = np.eye(3, dtype=np.float32)
    dataset = Dataset(memories, values, ["m1", "m2", "m3"], {"test": []}, {"test": {}}, {"test": {}})
    query = QueryRecord("q", "a", "query", 2, "test")
    assert dataset.eligible_positions(query).tolist() == [0]


def test_split_manifest_detects_overlap() -> None:
    values = validate_no_cross_split_ids({
        "train": [QueryRecord("q1", "a", "x", 1, "train")],
        "test": [QueryRecord("q1", "a", "x", 1, "test")],
    })
    assert not values["passed"]
    assert values["overlap"]["test:train"] == ["q1"] or values["overlap"]["train:test"] == ["q1"]
