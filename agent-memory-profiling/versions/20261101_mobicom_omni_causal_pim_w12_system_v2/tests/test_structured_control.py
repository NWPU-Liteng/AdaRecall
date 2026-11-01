import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pim_backend.structured_control import (StructuredControlConfig, mask_scan_cost,
                                            project_control_plane, residency_layout)


class StructuredControlTests(unittest.TestCase):
    def test_default_topology_is_two_eight_by_sixteen_stacks(self):
        config = StructuredControlConfig()
        self.assertEqual(config.bank_count, 256)
        self.assertEqual(config.bytes_per_cycle, 4096)

    def test_scan_is_parallel_and_bounded(self):
        config = StructuredControlConfig()
        one = mask_scan_cost(1, 248320, config)
        many = mask_scan_cost(256, 248320, config)
        self.assertEqual(one['tokens_per_bank'], 970)
        self.assertGreater(one['cycles'], one['command_fence_cycles'])
        self.assertGreater(many['cycles'], one['cycles'])
        self.assertLess(many['milliseconds_per_token'], 1.0)

    def test_residency_uses_one_weight_copy_and_charges_control_state(self):
        config = StructuredControlConfig()
        layout = residency_layout(5320474624, token_table_bytes=4 * 1024**2,
                                  grammar_state_bytes=64 * 1024**2,
                                  scratch_bytes=256 * 1024**2, config=config)
        self.assertTrue(layout['feasible'])
        self.assertEqual(layout['weight_copies'], 1)
        self.assertEqual(layout['total_bytes'], 5320474624 + 324 * 1024**2)

    def test_capacity_failure_is_explicit(self):
        layout = residency_layout(8 * 1024**3, token_table_bytes=1,
                                  grammar_state_bytes=0, scratch_bytes=0,
                                  config=StructuredControlConfig())
        self.assertFalse(layout['feasible'])
        self.assertEqual(layout['headroom_bytes'], -1)

    def test_control_projection_keeps_measurement_and_model_separate(self):
        result = project_control_plane(32.1958, 211, 248320)
        self.assertGreater(result['modeled_control_speedup'], 100)
        self.assertIn('not a measured PIM', result['boundary'])

    def test_invalid_inputs_fail_closed(self):
        with self.assertRaises(ValueError):
            mask_scan_cost(0, 248320)
        with self.assertRaises(ValueError):
            mask_scan_cost(1, 0)
        with self.assertRaises(ValueError):
            project_control_plane(float('nan'), 1, 10)
        with self.assertRaises(ValueError):
            residency_layout(-1, token_table_bytes=0, grammar_state_bytes=0, scratch_bytes=0)
        with self.assertRaises(ValueError):
            StructuredControlConfig(stacks=True)


if __name__ == '__main__':
    unittest.main()
