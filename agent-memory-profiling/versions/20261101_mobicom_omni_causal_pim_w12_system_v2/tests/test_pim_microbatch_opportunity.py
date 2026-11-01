import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pim_backend.microbatch_opportunity import audit_microbatch_opportunity


def trace_with_ready(ready):
    return {
        "schema": "pim-online-ready-queue-trace-v1",
        "clock": "monotonic_ns",
        "provenance": {"kind": "observed_online_runtime", "runtime_id": "runtime-test"},
        "snapshots": [{"snapshot_ns": 20, "ready": ready}],
    }


def event(request_id, compatibility_key="layer0-qproj-bf16", phase="decode", token_index=0):
    return {
        "request_id": request_id,
        "token_index": token_index,
        "ready_since_ns": 10,
        "phase": phase,
        "compatibility_key": compatibility_key,
    }


class PimMicrobatchOpportunityTests(unittest.TestCase):
    def test_legacy_serialized_trace_does_not_establish_concurrency(self):
        result = audit_microbatch_opportunity({
            "schema": "evidence-state-route-replay-v1",
            "rows": [{"request_index": 0, "service_seconds": 1.0}],
        })
        self.assertFalse(result["qualified"])
        self.assertIsNone(result["has_natural_microbatch_opportunity"])
        self.assertFalse(result["paired_smoke_allowed"])

    def test_two_naturally_ready_compatible_decode_requests_open_smoke_gate(self):
        result = audit_microbatch_opportunity(trace_with_ready([
            event("request-a"), event("request-b"),
        ]))
        self.assertTrue(result["qualified"])
        self.assertTrue(result["has_natural_microbatch_opportunity"])
        self.assertTrue(result["paired_smoke_allowed"])
        self.assertEqual(result["max_compatible_decode_batch_opportunity"], 2)
        self.assertIsNone(result["end_to_end_speedup"])

    def test_incompatible_or_non_decode_requests_do_not_create_opportunity(self):
        result = audit_microbatch_opportunity(trace_with_ready([
            event("request-a", compatibility_key="layer0-qproj-bf16"),
            event("request-b", compatibility_key="layer0-kproj-bf16"),
            event("request-c", compatibility_key="layer0-qproj-bf16", phase="prefill"),
        ]))
        self.assertFalse(result["has_natural_microbatch_opportunity"])
        self.assertFalse(result["paired_smoke_allowed"])

    def test_synthetic_provenance_is_rejected(self):
        trace = trace_with_ready([event("request-a"), event("request-b")])
        trace["provenance"]["kind"] = "synthetic"
        with self.assertRaisesRegex(ValueError, "observed online runtime"):
            audit_microbatch_opportunity(trace)

    def test_future_ready_time_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "not be later"):
            audit_microbatch_opportunity(trace_with_ready([
                {**event("request-a"), "ready_since_ns": 21},
            ]))

    def test_duplicate_request_in_same_snapshot_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "only one ready token"):
            audit_microbatch_opportunity(trace_with_ready([
                event("request-a", token_index=0), event("request-a", token_index=1),
            ]))


if __name__ == "__main__":
    unittest.main()
