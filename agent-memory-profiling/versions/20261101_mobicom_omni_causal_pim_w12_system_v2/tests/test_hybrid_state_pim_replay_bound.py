import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.replay_hybrid_state_pim_bound import replay


class HybridStatePimReplayBoundTests(unittest.TestCase):
    def test_bound_retains_full_attention_transfer_cost(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            churn = root / "churn.json"
            qualification = root / "qualification.json"
            components = root / "components.json"
            output = root / "output.json"
            churn.write_text(json.dumps({
                "status": "complete",
                "schedule": "round_robin",
                "agent_count": 2,
                "target_tokens": 16,
                "hbm_capacity_agents": 1,
                "access_records": [{
                    "access_index": 0,
                    "agent_index": 1,
                    "cycle_index": 0,
                    "cache_hit": False,
                    "total_seconds": 10.0,
                    "decode_seconds": 2.0,
                    "restore_seconds": 4.0,
                    "eviction_seconds": 2.0,
                }],
            }))
            qualification.write_text(json.dumps({
                "native_state": {
                    "total_bytes": 100,
                    "linear_attention_state_bytes": 10,
                },
                "pim_plan": {"pim_step": {"cycles": 8}},
                "pim_profile": {"clock_hz": 8},
            }))
            components.write_text(json.dumps({
                "derived": {"extended_pim_fraction_upper_mean": 0.5},
            }))

            result = replay(churn, qualification, components, output)
            summary = result["summary"]
            self.assertAlmostEqual(
                summary["capacity_only_linear_state"]["candidate_total_seconds"],
                2.0 + 3.6 + 1.8,
            )
            self.assertAlmostEqual(
                result["rows"][0]["retained_full_attention_restore_seconds"],
                3.6,
            )
            self.assertLess(
                summary["capacity_only_linear_state"]["speedup_total"],
                1.4,
            )


if __name__ == "__main__":
    unittest.main()
