import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pim_backend.hyperpim_sc import (DecodeCostConfig, W8A16CostConfig,
                                     bank_resident_decoder_plan,
                                     grammar_residency, packed_bit_sliced_decoder_plan,
                                     phase_latency_projection,
                                     w8a16_mixed_precision_overhead)
from pim_backend.structured_control import StructuredControlConfig


class HyperPimScTests(unittest.TestCase):
    def test_reader_target_does_not_charge_grammar_scan(self):
        costs = DecodeCostConfig(40.0, 7.0, 6.0, 0.1, 4.0, safety_factor=1.0)
        self.assertAlmostEqual(costs.target_ms(structured=True), 17.1)
        self.assertAlmostEqual(costs.target_ms(structured=False), 17.0)
        self.assertAlmostEqual(costs.weight_stream_speedup(), 40.0 / 7.0)

    def test_bank_plan_keeps_one_copy_and_accounts_padding(self):
        config = StructuredControlConfig(stacks=1, channels_per_stack=2,
                                         banks_per_channel=2)
        plan = bank_resident_decoder_plan([
            {'name': 'a', 'out_features': 5, 'in_features': 4},
            {'name': 'b', 'out_features': 2, 'in_features': 3},
        ], config, scale_bytes_per_row=0)
        self.assertEqual(plan['weight_copies'], 1)
        self.assertEqual(plan['logical_weight_bytes'], 26)
        self.assertEqual(plan['bank_padded_weight_bytes'], 26)
        self.assertEqual(plan['bank_min_bytes'], 0)
        self.assertEqual(plan['bank_max_bytes'], 11)
        self.assertTrue(plan['feasible'])

    def test_w8a16_bank_plan_accounts_group_scales(self):
        config = StructuredControlConfig(stacks=1, channels_per_stack=2,
                                         banks_per_channel=2)
        plan = bank_resident_decoder_plan([
            {'name': 'a', 'out_features': 4, 'in_features': 256},
            {'name': 'b', 'out_features': 3, 'in_features': 130},
        ], config, group_size=128, scale_bytes_per_group=4)
        self.assertEqual(plan['logical_scale_groups'], 4 * 2 + 3 * 2)
        self.assertEqual(plan['logical_scale_bytes'], 56)
        self.assertEqual(plan['group_size'], 128)
        self.assertEqual(plan['weight_copies'], 1)

    def test_twelve_bit_row_packing_has_one_plane_and_fits_aligned_rows(self):
        matrices = [
            {'name': 'up', 'out_features': 16, 'in_features': 2560},
            {'name': 'down', 'out_features': 8, 'in_features': 9216},
        ]
        plan = packed_bit_sliced_decoder_plan(
            matrices, weight_bits=12, row_bytes=2048,
            capacity_bytes=64 * 1024**2)
        self.assertEqual(plan['weight_bits'], 12)
        self.assertEqual(plan['weight_copies'], 1)
        self.assertEqual(plan['matrices'][0]['physical_row_bytes'], 4096)
        self.assertEqual(plan['matrices'][1]['physical_row_bytes'], 14336)
        self.assertEqual(plan['logical_weight_bytes'], 16 * 3840 + 8 * 13824)
        self.assertAlmostEqual(plan['row_packing_overhead'],
                               (16 * 4096 + 8 * 14336)
                               / (16 * 3840 + 8 * 13824))
        self.assertTrue(plan['feasible'])

    def test_twelve_bit_capacity_model_rejects_invalid_contract(self):
        with self.assertRaises(ValueError):
            packed_bit_sliced_decoder_plan(
                [{'name': 'a', 'out_features': 1, 'in_features': 128}],
                weight_bits=16)
        with self.assertRaises(ValueError):
            packed_bit_sliced_decoder_plan(
                [{'name': 'a', 'out_features': 1, 'in_features': 0}])

    def test_w8a16_overhead_charges_conversion_scale_and_broadcast(self):
        config = StructuredControlConfig(stacks=1, channels_per_stack=1,
                                         banks_per_channel=1, clock_hz=1_000_000_000)
        result = w8a16_mixed_precision_overhead(
            [{'name': 'a', 'out_features': 1, 'in_features': 256}],
            1.0, config, W8A16CostConfig(
                group_size=128, conversion_elements_per_bank_cycle=16,
                group_apply_ops_per_bank_cycle=16, matrix_command_cycles=1,
                broadcast_bandwidth_bytes_per_second=512_000_000))
        self.assertEqual(result['conversion_cycles'], 16)
        self.assertEqual(result['scale_read_cycles'], 1)
        self.assertEqual(result['scale_apply_cycles'], 1)
        self.assertAlmostEqual(result['arithmetic_ms_per_token'], 19e-6)
        self.assertAlmostEqual(result['activation_broadcast_ms_per_token'], 1e-3)
        self.assertAlmostEqual(result['total_overhead_ms_per_token'], 1.019e-3)

    def test_w8a16_residual_bits_are_charged_in_capacity_and_mac(self):
        config = StructuredControlConfig(stacks=1, channels_per_stack=1,
                                         banks_per_channel=1)
        matrices = [{'name': 'a', 'out_features': 4, 'in_features': 256}]
        plan = bank_resident_decoder_plan(
            matrices, config, group_size=128, scale_bytes_per_group=4,
            residual_bits=2, residual_scale_bytes=4)
        self.assertEqual(plan['logical_residual_bytes'],
                         4 * (256 * 2 // 8 + 2 * 4))
        result = w8a16_mixed_precision_overhead(
            matrices, 2.0, config, W8A16CostConfig(
                group_size=128, residual_bits=2,
                broadcast_bandwidth_bytes_per_second=1e12))
        self.assertAlmostEqual(result['residual_mac_ms_per_token'], 2.0)
        self.assertAlmostEqual(result['pim_linear_with_overhead_ms_per_token'],
                               4.0 + result['total_overhead_ms_per_token'])
        self.assertGreater(result['residual_conversion_cycles'], 0)
        self.assertGreater(result['residual_scale_apply_cycles'], 0)

    def test_w8a16_mac_slowdown_is_explicit(self):
        config = StructuredControlConfig(stacks=1, channels_per_stack=1,
                                         banks_per_channel=1)
        result = w8a16_mixed_precision_overhead(
            [{'name': 'a', 'out_features': 1, 'in_features': 128}],
            2.0, config, W8A16CostConfig(mac_slowdown=1.5))
        self.assertAlmostEqual(result['mac_slowdown_ms_per_token'], 1.0)
        self.assertAlmostEqual(result['pim_linear_with_overhead_ms_per_token'],
                               3.0 + result['arithmetic_ms_per_token']
                               + result['activation_broadcast_ms_per_token'])

    def test_invalid_w8a16_contract_is_rejected(self):
        with self.assertRaises(ValueError):
            W8A16CostConfig(group_size=0)
        with self.assertRaises(ValueError):
            W8A16CostConfig(mac_slowdown=0.9)
        with self.assertRaises(ValueError):
            W8A16CostConfig(broadcast_bandwidth_bytes_per_second=0)

    def test_phase_projection_caps_target_at_measured_cost(self):
        result = phase_latency_projection(
            measured_seconds=10.0, prefill_seconds=1.0,
            host_forward_ms_per_token=100.0, tokens=5,
            target_ms_per_token=20.0)
        self.assertAlmostEqual(result['host_control_seconds'], 1.5)
        self.assertAlmostEqual(result['target_seconds'], 1.1)
        self.assertGreater(result['target_over_host_control'], 1)

    def test_grammar_residency_fails_closed(self):
        config = StructuredControlConfig(capacity_bytes=1024)
        self.assertTrue(grammar_residency(100, 1, 2, 3, config)['feasible'])
        self.assertFalse(grammar_residency(1024, 1, 0, 0, config)['feasible'])
        with self.assertRaises(ValueError):
            grammar_residency(-1, 0, 0, 0, config)

    def test_invalid_cost_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            DecodeCostConfig(1, 1, 1, 1, 1, safety_factor=0.9)
        with self.assertRaises(ValueError):
            DecodeCostConfig(1, -1, 1, 1, 1)


if __name__ == '__main__':
    unittest.main()
