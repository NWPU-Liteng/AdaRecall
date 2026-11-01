import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.replay_unified_hybrid_state_tiers import replay


class UnifiedHybridStateTierReplayTests(unittest.TestCase):
    def test_two_level_lru_counts_gpu_pim_and_host_routes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            churn = root / "churn.json"
            qualification = root / "qualification.json"
            output = root / "output.json"
            sequence = [0, 1, 2, 3, 0, 1, 4, 5, 0]
            gpu_lru = []
            access_records = []
            for index, agent in enumerate(sequence):
                cache_hit = agent in gpu_lru
                if cache_hit:
                    gpu_lru.remove(agent)
                elif len(gpu_lru) >= 2:
                    gpu_lru.pop(0)
                gpu_lru.append(agent)
                access_records.append({
                    "access_index": index,
                    "agent_index": agent,
                    "cache_hit": cache_hit,
                    "restore_seconds": 1.0,
                    "eviction_seconds": 0.5,
                    "total_seconds": 2.0,
                })
            churn.write_text(json.dumps({
                "status": "complete",
                "schedule": "test",
                "agent_count": 6,
                "target_tokens": 24,
                "hbm_capacity_agents": 2,
                "access_records": access_records,
            }))
            qualification.write_text(json.dumps({
                "pim_plan": {
                    "admission": {
                        "state_scope": "unified",
                        "available_bytes": 200,
                        "per_agent_state_bytes": 100,
                        "resident_agents": 2,
                    },
                },
                "pim_profile": {
                    "external_service_bytes_per_cycle": 32,
                    "clock_hz": 1,
                    "calibration_status": "test-only",
                },
            }))

            result = replay(churn, qualification, output)
            counts = result["tiered_candidate_counts"]
            self.assertEqual(counts["pim_served"], 2)
            self.assertEqual(counts["host_restores"], 7)
            self.assertEqual(counts["pim_to_host_writebacks"], 3)
            self.assertGreater(counts["gpu_to_pim_migration_bytes"], 0)
            self.assertIsNotNone(counts["raw_migration_transfer_lower_bound_seconds"])
            self.assertIsNone(counts["candidate_service_seconds"])


if __name__ == "__main__":
    unittest.main()
