from __future__ import annotations

import sys
from pathlib import Path


VERSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VERSION_ROOT / "src"))

from v21.evidence_sets import QueryEvidenceSet, incidence_rows, summarize  # noqa: E402


def test_incidence_preserves_group_members_as_a_set() -> None:
    record = QueryEvidenceSet("q", "1", 3, "g", "medmmb_en_p1_s1_x2", ("medmmb_en_p1_s1_x2", "medmmb_en_p1_s1_x3"), 4)
    row = incidence_rows([record], folds=5)[0]
    assert row["acceptable_memory_ids"] == ["medmmb_en_p1_s1_x2", "medmmb_en_p1_s1_x3"]
    assert row["source_session_lags"] == [2, 2]


def test_summary_marks_multi_group_queries_without_temporal_claim() -> None:
    records = [
        QueryEvidenceSet("q1", "1", 3, "g1", "medmmb_en_p1_s1_x2", ("medmmb_en_p1_s1_x2", "medmmb_en_p1_s1_x3"), 4),
        QueryEvidenceSet("q1", "1", 3, "g2", "medmmb_en_p1_s2_x1", ("medmmb_en_p1_s2_x1", "medmmb_en_p1_s2_x2"), 4),
    ]
    result = summarize(records, {"q1": 2})
    assert result["queries_with_two_or_more_groups"] == 1
    assert result["members_per_group"] == {2: 2}


if __name__ == "__main__":
    tests = sorted((name, value) for name, value in globals().items() if name.startswith("test_") and callable(value))
    for name, test in tests:
        test()
        print(f"PASS {name}")
    print(f"{len(tests)} passed")
