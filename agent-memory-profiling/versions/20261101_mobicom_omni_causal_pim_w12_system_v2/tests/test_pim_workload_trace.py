import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pim_backend.workload_trace import audit_workload_trace


def trace():
    return {
        "schema": "agent-memory-observed-memory-trace-v1",
        "request_id": "req-1",
        "clock": "monotonic_ns",
        "synthetic_timestamps": False,
        "provenance": {"kind": "observed_target_runtime", "runtime_id": "edge-build-1"},
        "events": [
            {"event_id": "retrieve", "request_id": "req-1", "kind": "memory",
             "start_ns": 0, "end_ns": 100, "dependencies": [], "pim_eligible": True,
             "memory_counters": {"read_bytes": 4096, "write_bytes": 0,
                                 "memory_wait_ns": 60}},
            {"event_id": "reader", "request_id": "req-1", "kind": "compute",
             "start_ns": 100, "end_ns": 200, "dependencies": ["retrieve"],
             "pim_eligible": False, "memory_counters": {"read_bytes": 0,
                                 "write_bytes": 0, "memory_wait_ns": 0}},
        ],
    }


class WorkloadTraceTests(unittest.TestCase):
    def test_qualifies_observed_memory_wait_on_critical_path(self):
        result = audit_workload_trace(trace())
        self.assertTrue(result["qualified_for_candidate_design"])
        self.assertEqual(result["critical_path_event_ids"], ["retrieve", "reader"])
        self.assertAlmostEqual(result["critical_path_memory_wait_fraction"], 0.3)
        self.assertEqual(result["pim_eligible_read_bytes"], 4096)
        self.assertFalse(result["pim_speedup_claim_allowed"])
        self.assertIsNone(result["paired_e2e_speedup"])

    def test_off_critical_path_wait_does_not_qualify(self):
        source = trace()
        source["events"].append({
            "event_id": "side", "request_id": "req-1", "kind": "memory",
            "start_ns": 0, "end_ns": 100, "dependencies": [], "pim_eligible": True,
            "memory_counters": {"read_bytes": 4096, "write_bytes": 0,
                                "memory_wait_ns": 100},
        })
        source["events"][1]["dependencies"] = []
        source["events"][1]["end_ns"] = 300
        result = audit_workload_trace(source)
        self.assertFalse(result["qualified_for_candidate_design"])

    def test_unrelated_memory_wait_does_not_qualify_eligible_bytes(self):
        source = trace()
        source["events"][0]["pim_eligible"] = False
        source["events"][1]["pim_eligible"] = True
        source["events"][1]["memory_counters"] = {
            "read_bytes": 4096, "write_bytes": 0, "memory_wait_ns": 0,
        }
        result = audit_workload_trace(source)
        self.assertFalse(result["qualified_for_candidate_design"])

    def test_synthetic_trace_is_rejected(self):
        source = trace()
        source["synthetic_timestamps"] = True
        with self.assertRaisesRegex(ValueError, "synthetic timestamps"):
            audit_workload_trace(source)

    def test_dependency_overlap_is_rejected(self):
        source = trace()
        source["events"][1]["start_ns"] = 90
        with self.assertRaisesRegex(ValueError, "starts before dependency"):
            audit_workload_trace(source)

    def test_old_replay_is_not_misclassified_as_observed_trace(self):
        result = audit_workload_trace({"schema": "evidence-state-route-replay-v1"})
        self.assertEqual(result["status"], "unsupported_or_missing_trace_schema")
        self.assertFalse(result["qualified_for_candidate_design"])


if __name__ == "__main__":
    unittest.main()
