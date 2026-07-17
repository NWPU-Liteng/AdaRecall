from __future__ import annotations

import sys
from pathlib import Path


VERSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VERSION_ROOT / "src"))

from v21.eligibility import (  # noqa: E402
    InferenceEligibilityQuery,
    assert_train_only_incidence,
    generate_eligible_memory_ids,
)


class GoldPoison(dict):
    def __getitem__(self, key):
        if str(key).startswith("gold") or key in {"evidence_groups", "acceptable_memory_ids"}:
            raise AssertionError(f"eligibility generation attempted to read {key}")
        return super().__getitem__(key)


def test_eligible_memory_generation_does_not_read_gold() -> None:
    memories = [
        {"memory_id": "a", "persona_id": 1, "session_id": 1, "session_type": "medical", "chronological_rank": 0},
        {"memory_id": "b", "persona_id": 1, "session_id": 3, "session_type": "medical", "chronological_rank": 1},
        {"memory_id": "c", "persona_id": 2, "session_id": 1, "session_type": "medical", "chronological_rank": 0},
    ]
    query = GoldPoison({
        "persona_id": 1,
        "query_session_id": 2,
        "gold_memory_ids": ["b"],
        "evidence_groups": [{"acceptable_memory_ids": ["b"]}],
    })
    assert generate_eligible_memory_ids(
        memories,
        query,
        include_noise=True,
        allow_current_session_context=True,
    ) == ["a"]


def test_inference_path_cannot_load_evidence_group_labels() -> None:
    clean = {
        "query_id": "q",
        "persona_id": 1,
        "query_session_id": 2,
        "eligible_memory_ids": ["a"],
    }
    assert InferenceEligibilityQuery.from_mapping(clean).eligible_memory_ids == ("a",)
    contaminated = {**clean, "evidence_groups": [{"acceptable_memory_ids": ["a"]}]}
    try:
        InferenceEligibilityQuery.from_mapping(contaminated)
    except ValueError as error:
        assert "supervision-only" in str(error)
    else:
        raise AssertionError("inference input accepted evidence-group supervision")


def test_calibration_and_test_are_absent_from_train_incidence() -> None:
    train_ids = {"q1", "q2"}
    rows = [{"query_id": "q1"}, {"query_id": "q1"}, {"query_id": "q2"}]
    result = assert_train_only_incidence(rows, train_ids, source_split="train")
    assert result["unexpected_query_ids"] == 0
    try:
        assert_train_only_incidence([*rows, {"query_id": "q_calibration"}], train_ids, source_split="train")
    except ValueError as error:
        assert "unexpected" in str(error)
    else:
        raise AssertionError("non-train query entered train incidence")


if __name__ == "__main__":
    tests = sorted((name, value) for name, value in globals().items() if name.startswith("test_") and callable(value))
    for name, test in tests:
        test()
        print(f"PASS {name}")
    print(f"{len(tests)} passed")
