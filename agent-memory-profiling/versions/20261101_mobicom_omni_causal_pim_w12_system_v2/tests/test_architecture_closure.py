import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pim_backend.architecture_closure import (
    EvidencePimArchitectureConfig,
    model_materialization,
    project_rows,
    project_joint_retrieval_rows,
)


def rows():
    return [{
        "question_id": "q0",
        "person_id": "q0",
        "service_seconds": 100.0,
        "payload_bytes": 1024,
        "metadata_bytes": 64,
        "packed_bytes": 1088,
        "event_seconds": {
            "query_parse_and_retrieval": 20.0,
            "source_grounding": 50.0,
            "proof_materialize": 10.0,
            "reader_prefill": 5.0,
            "reader_decode": 15.0,
        },
    }]


class ArchitectureClosureTests(unittest.TestCase):
    def test_materialization_charges_read_write_setup_and_fence(self):
        result = model_materialization(1024, 64, EvidencePimArchitectureConfig())
        self.assertEqual(result["traffic_bytes"], 2176)
        self.assertGreater(result["memory_cycles"], 0)
        self.assertGreater(result["compute_cycles"], 0)
        self.assertGreater(result["seconds"], 0)

    def test_only_materialization_phase_changes(self):
        result = project_rows(rows(), EvidencePimArchitectureConfig())
        row = result["rows"][0]
        self.assertLess(row["candidate_service_seconds"], 100.0)
        self.assertLess(row["candidate_ttft_seconds"], 85.0)
        self.assertEqual(result["metrics"]["materialize_host_seconds_total"], 10.0)
        self.assertEqual(result["metrics"]["materialize_host_fraction"], 0.1)

    def test_gate_fails_without_quality_assumption(self):
        result = project_rows(rows(), EvidencePimArchitectureConfig(),
                              quality_contract_assumed=False)
        self.assertFalse(result["positive_architecture_gate_pass"])
        self.assertFalse(result["gates"]["quality_contract_assumed"])

    def test_capacity_is_per_request_and_explicit(self):
        source = rows()
        source[0]["packed_bytes"] = 3 * 1024 ** 3
        result = project_rows(source, EvidencePimArchitectureConfig())
        self.assertFalse(result["gates"]["capacity_all_admitted"])

    def test_joint_projection_exposes_unobserved_scan_fraction(self):
        result = project_joint_retrieval_rows(
            rows(), {"q0": 1000}, EvidencePimArchitectureConfig(),
            retrieval_scan_fraction=0.25)
        self.assertEqual(result["retrieval_scan_fraction_assumed"], 0.25)
        self.assertLess(result["metrics"]["candidate_service_seconds" ]
                        if "candidate_service_seconds" in result["metrics"] else
                        result["metrics"]["service_speedup"], 100.0)
        self.assertIn("retrieval_host_seconds_total", result["metrics"])
        self.assertIn("architecture models", result["claim_boundary"])


if __name__ == "__main__":
    unittest.main()
