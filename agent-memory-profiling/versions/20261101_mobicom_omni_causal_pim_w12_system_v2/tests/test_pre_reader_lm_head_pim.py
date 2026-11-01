import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from project_pre_reader_lm_head_pim import project_structured_call
from pim_backend.lpddr_bank_pim import BankPimConfig


class PreReaderLmHeadReplayTests(unittest.TestCase):
    def test_only_output_head_is_admitted_and_e2e_is_not_claimed(self):
        import json

        profile = json.loads((ROOT / "pim_backend/mobile_lpddr5_bank_profile.json").read_text())
        config = BankPimConfig(**profile["config"])
        trace = {"operators": [
            {"phase": "decode", "rows": 1, "name": "decoder", "in_features": 4,
             "out_features": 4, "element_bytes": 2, "weight_bytes": 32, "calls": 2},
            {"phase": "decode", "rows": 1, "name": "lm_head", "in_features": 4,
             "out_features": 8, "element_bytes": 2, "weight_bytes": 64, "calls": 2},
        ]}
        result = project_structured_call(trace, config)
        self.assertEqual(result["target"], "lm_head")
        self.assertTrue(result["capacity_feasible"])
        self.assertGreater(result["linear_phase_speedup_warm"], 0.0)
        self.assertEqual(result["boundary"],
                         "modeled output-head linear phase only; not an end-to-end latency result")

    def test_capacity_override_fails_open_to_host(self):
        import json

        profile = json.loads((ROOT / "pim_backend/mobile_lpddr5_bank_profile.json").read_text())
        config = BankPimConfig(**profile["config"])
        trace = {"operators": [
            {"phase": "decode", "rows": 1, "name": "decoder", "in_features": 4,
             "out_features": 4, "element_bytes": 2, "weight_bytes": 32, "calls": 2},
            {"phase": "decode", "rows": 1, "name": "lm_head", "in_features": 4,
             "out_features": 8, "element_bytes": 2, "weight_bytes": 64, "calls": 2},
        ]}
        result = project_structured_call(trace, BankPimConfig(
            **{**config.__dict__, "pim_capacity_bytes": 1}))
        self.assertFalse(result["capacity_feasible"])
        self.assertEqual(result["route"], "host_capacity_fallback")
        self.assertEqual(result["linear_phase_speedup_warm"], 1.0)


if __name__ == "__main__":
    unittest.main()
