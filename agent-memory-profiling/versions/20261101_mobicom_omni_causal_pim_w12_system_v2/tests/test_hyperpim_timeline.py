import sys
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pim_backend.hyperpim_timeline import (
    HyperPimTimelineConfig, matrix_cycles, refresh_elapsed, simulate_token,
    with_residual_bits, with_weight_bits)


class HyperPimTimelineTests(unittest.TestCase):
    def test_refresh_blocks_periodically_and_preserves_active_work(self):
        config = HyperPimTimelineConfig(
            refresh_interval_cycles=100, refresh_block_cycles=20)
        short = refresh_elapsed(50, 0, config)
        crossed = refresh_elapsed(150, 0, config)
        self.assertEqual(short['refresh_cycles'], 0)
        self.assertEqual(short['elapsed_cycles'], 50)
        self.assertGreater(crossed['refresh_cycles'], 0)
        self.assertEqual(crossed['elapsed_cycles'],
                         crossed['active_cycles'] + crossed['refresh_cycles'])

    def test_matrix_charges_memory_mac_rows_scales_and_residual(self):
        config = HyperPimTimelineConfig(
            stacks=1, channels_per_stack=1, banks_per_channel=2,
            residual_bits=2, layer_fence_cycles=1)
        result = matrix_cycles(256, 4, config)
        self.assertEqual(result['active_banks'], 2)
        self.assertEqual(result['rows_per_bank'], 2)
        self.assertEqual(result['elements_per_bank'], 512)
        self.assertGreater(result['int8_mac_cycles'], 0)
        self.assertGreater(result['row_activation_cycles'], 0)
        self.assertGreater(result['residual_conversion_cycles'], 0)
        self.assertGreater(result['residual_mac_cycles'], 0)

    def test_residual_increases_token_time_but_not_weight_inventory(self):
        sequence = [
            {'name': 'model.layers.0.a', 'inputs': 256, 'outputs': 512, 'layer': 0},
            {'name': 'model.layers.0.b', 'inputs': 512, 'outputs': 256, 'layer': 0},
        ]
        base = HyperPimTimelineConfig(
            stacks=1, channels_per_stack=1, banks_per_channel=4)
        r0 = simulate_token(sequence, with_residual_bits(base, 0))
        r2 = simulate_token(sequence, with_residual_bits(base, 2))
        self.assertEqual(r0['totals']['weight_elements'],
                         r2['totals']['weight_elements'])
        self.assertGreater(r2['totals']['pim_linear_ms_per_token'],
                           r0['totals']['pim_linear_ms_per_token'])
        self.assertGreater(r2['totals']['residual_mac_cycles'], 0)
        self.assertGreaterEqual(r2['totals']['refresh_cycles'], 0)

    def test_twelve_bit_plane_charges_one_and_a_half_weight_bytes(self):
        config = HyperPimTimelineConfig(
            stacks=1, channels_per_stack=1, banks_per_channel=2)
        eight = matrix_cycles(2560, 64, config)
        twelve = matrix_cycles(2560, 64, with_weight_bits(config, 12))
        self.assertEqual(twelve['weight_bits'], 12)
        self.assertEqual(twelve['weight_bytes_per_bank'],
                         eight['weight_bytes_per_bank'] * 3 // 2)
        self.assertGreater(twelve['active_cycles'], eight['active_cycles'])
        with self.assertRaises(ValueError):
            with_weight_bits(config, 16)
        with self.assertRaises(ValueError):
            HyperPimTimelineConfig(weight_bits=12, residual_bits=2)

    def test_bf16_reconstruction_cycles_are_explicit_and_conservative(self):
        config = HyperPimTimelineConfig(
            stacks=1, channels_per_stack=1, banks_per_channel=4)
        w12 = with_weight_bits(config, 12)
        result = matrix_cycles(2560, 64, w12)
        self.assertTrue(w12.bf16_reconstruct_weights)
        self.assertGreater(result["bf16_reconstruction_cycles"], 0)
        self.assertGreater(result["bf16_rounding_cycles"], 0)
        self.assertGreater(result["bf16_group_reduction_cycles"], 0)
        self.assertGreater(result["bf16_mac_cycles"], 0)
        self.assertGreater(result["bf16_pipeline_overlap_cycles"], 0)
        self.assertEqual(
            result["bf16_reconstruction_cycles"],
            (result["elements_per_bank"]
             + w12.bf16_reconstruction_lanes_per_bank - 1)
            // w12.bf16_reconstruction_lanes_per_bank)
        self.assertGreater(result["effective_mac_cycles"], 0)
        joint = replace(
            w12, bf16_macs_per_bank_cycle=1,
            bf16_reconstruction_lanes_per_bank=8,
            bf16_round_lanes_per_bank=8,
            bf16_group_reduction_lanes_per_bank=8)
        joint_result = matrix_cycles(2560, 64, joint)
        self.assertGreater(joint_result["active_cycles"],
                           result["active_cycles"])
        with self.assertRaises(ValueError):
            HyperPimTimelineConfig(
                weight_bits=8, bf16_reconstruct_weights=True)

    def test_layer_fences_are_accounted_once_per_layer(self):
        one = [{'name': 'model.layers.0.a', 'inputs': 256, 'outputs': 256, 'layer': 0}]
        two = one + [{'name': 'model.layers.1.a', 'inputs': 256, 'outputs': 256, 'layer': 1}]
        config = HyperPimTimelineConfig(
            stacks=1, channels_per_stack=1, banks_per_channel=2,
            layer_fence_cycles=10, segment_fences_per_layer=4)
        first = simulate_token(one, config)
        second = simulate_token(two, config)
        self.assertEqual(first['totals']['layer_fence_cycles'], 40)
        self.assertEqual(second['totals']['layer_fence_cycles'], 80)

    def test_invalid_timeline_contract_fails_closed(self):
        with self.assertRaises(ValueError):
            HyperPimTimelineConfig(residual_bits=1)
        with self.assertRaises(ValueError):
            HyperPimTimelineConfig(refresh_block_cycles=3125)
        with self.assertRaises(ValueError):
            matrix_cycles(0, 1)
        with self.assertRaises(ValueError):
            simulate_token([])


if __name__ == '__main__':
    unittest.main()
