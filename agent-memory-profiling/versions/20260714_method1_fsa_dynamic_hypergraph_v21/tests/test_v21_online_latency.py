from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter_ns


VERSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VERSION_ROOT / "src"))

from v21.online_latency import LATENCY_FIELDS, OnlineLatencyRecorder  # noqa: E402


def test_latency_recorder_has_complete_schema_and_nonnegative_total() -> None:
    recorder = OnlineLatencyRecorder("cpu")
    recorder.start_query()
    selection_start = perf_counter_ns()
    with recorder.stage("ann_seed_search_ms"):
        sum(range(100))
    result = recorder.finish_query(selection_start)
    assert tuple(result) == LATENCY_FIELDS
    assert result["ann_seed_search_ms"] >= 0.0
    assert result["query_e2e_ms"] >= result["selection_total_ms"] >= 0.0


def test_latency_recorder_rejects_derived_stage_override() -> None:
    recorder = OnlineLatencyRecorder("cpu")
    recorder.start_query()
    try:
        with recorder.stage("query_e2e_ms"):
            pass
    except ValueError:
        pass
    else:
        raise AssertionError("recorder allowed direct e2e overwrite")
    recorder.finish_query(perf_counter_ns())


if __name__ == "__main__":
    tests = sorted((name, value) for name, value in globals().items() if name.startswith("test_") and callable(value))
    for name, test in tests:
        test()
        print(f"PASS {name}")
    print(f"{len(tests)} passed")
