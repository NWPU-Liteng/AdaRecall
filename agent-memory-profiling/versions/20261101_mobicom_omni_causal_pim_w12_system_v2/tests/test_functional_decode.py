from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pim_backend.decode_gemv import bank_tiled_gemv
from pim_backend.functional_decode import (BankOwnedLinear, DecodeBankDispatcher,
    Int8WeightOnlyContract, full_matrix_numerics, materialize_w8a16_primary,
    quantize_int8_groups, quantize_int8_residual_groups)
from pim_backend.online_residency import OnlineResidency


class FunctionalDecodeTests(unittest.TestCase):
    def test_groupwise_weight_only_contract_uses_dequantized_bf16_values(self):
        generator = torch.Generator().manual_seed(20261028)
        weight = torch.randn(19, 512, generator=generator).bfloat16()
        bias = torch.randn(19, generator=generator).bfloat16()
        quantized, scales, dequantized = quantize_int8_groups(weight, group_size=128)
        self.assertEqual(tuple(quantized.shape), (19, 512))
        self.assertEqual(tuple(scales.shape), (19, 4))
        self.assertLessEqual(float(quantized.abs().max()), 127)
        contract = Int8WeightOnlyContract(weight, bias, group_size=128)
        self.assertTrue(torch.equal(contract.weight, dequantized.bfloat16()))
        activation = torch.randn(1, 1, 512, generator=generator).bfloat16()
        expected = torch.nn.functional.linear(activation, contract.weight, contract.bias)
        self.assertTrue(torch.equal(contract(activation), expected))
        self.assertEqual(contract.logical_int8_weight_bytes, 19 * 512)
        self.assertEqual(contract.scale_storage_bytes, 19 * 4 * 4)
        with self.assertRaises(ValueError):
            quantize_int8_groups(weight, group_size=100)

    def test_w8a16_materialization_replaces_only_language_linears(self):
        class Reader(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = torch.nn.Module()
                self.model.language_model = torch.nn.Module()
                self.model.language_model.layers = torch.nn.Sequential(torch.nn.Linear(256, 32))
                self.vision = torch.nn.Linear(8, 8)

        reader = Reader().bfloat16().eval()
        original_vision = reader.vision
        summary = materialize_w8a16_primary(reader, group_size=128)
        self.assertEqual(summary['modules'], 1)
        self.assertIs(reader.vision, original_vision)
        self.assertIsInstance(reader.model.language_model.layers[0], Int8WeightOnlyContract)
        self.assertEqual(summary['logical_int8_weight_and_scale_bytes'],
                         32 * 256 + 32 * 2 * 4 + 32 * 2)

    def test_residual_correction_reduces_groupwise_error_and_is_charged(self):
        generator = torch.Generator().manual_seed(20261029)
        weight = torch.randn(17, 512, generator=generator).bfloat16()
        _, _, base, _, _ = quantize_int8_residual_groups(weight, 128, 0)
        _, _, refined2, residual2, residual_scales2 = quantize_int8_residual_groups(
            weight, 128, 2)
        _, _, refined4, _, _ = quantize_int8_residual_groups(weight, 128, 4)
        error = lambda value: float((value.float() - weight.float()).square().mean())
        self.assertLess(error(refined2), error(base))
        self.assertLess(error(refined4), error(refined2))
        self.assertEqual(tuple(residual2.shape), tuple(weight.shape))
        self.assertEqual(tuple(residual_scales2.shape), (17, 4))
        self.assertLessEqual(int(residual2.abs().max()), 1)
        contract = Int8WeightOnlyContract(weight, group_size=128, residual_bits=2)
        self.assertTrue(torch.equal(contract.weight, refined2))
        with self.assertRaises(ValueError):
            quantize_int8_residual_groups(weight, 128, 1)

    def test_materialized_residual_storage_accounting(self):
        class Reader(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = torch.nn.Module()
                self.model.language_model = torch.nn.Module()
                self.model.language_model.layers = torch.nn.Sequential(
                    torch.nn.Linear(256, 32, bias=False))

        reader = Reader().bfloat16().eval()
        summary = materialize_w8a16_primary(
            reader, group_size=128, residual_bits=2)
        self.assertEqual(summary['residual_bits'], 2)
        self.assertEqual(summary['residual_payload_bytes'], 32 * 2 * (128 * 2 // 8))
        self.assertEqual(summary['residual_scale_bytes'], 32 * 2 * 4)

    def test_w8a16_lm_head_fallback_remains_bf16(self):
        class Reader(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = torch.nn.Module()
                self.model.language_model = torch.nn.Module()
                self.model.language_model.layers = torch.nn.ModuleDict({
                    str(index): torch.nn.Linear(256, 32) for index in range(4)})
                self.lm_head = torch.nn.Linear(32, 64, bias=False)

        reader = Reader().bfloat16().eval()
        original_lm_head = reader.lm_head
        summary = materialize_w8a16_primary(
            reader, group_size=128, fallback_scope='lm_head')
        self.assertIs(reader.lm_head, original_lm_head)
        self.assertIsInstance(reader.model.language_model.layers['0'], Int8WeightOnlyContract)
        self.assertEqual(summary['modules'], 4)
        self.assertEqual(summary['fallback_modules'], ['lm_head'])
        self.assertEqual(summary['retained_bf16_fallback_bytes'], 64 * 32 * 2)

    def test_output_sensitive_fallback_keeps_output_and_boundary_layers(self):
        class Reader(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = torch.nn.Module()
                self.model.language_model = torch.nn.Module()
                self.model.language_model.layers = torch.nn.ModuleDict({
                    str(index): torch.nn.Linear(256, 32) for index in range(6)})
                self.lm_head = torch.nn.Linear(32, 64, bias=False)

        reader = Reader().bfloat16().eval()
        summary = materialize_w8a16_primary(
            reader, group_size=128, fallback_scope='output_sensitive')
        self.assertEqual(summary['fallback_module_count'], 5)
        self.assertIsInstance(reader.model.language_model.layers['0'], torch.nn.Linear)
        self.assertIsInstance(reader.model.language_model.layers['1'], torch.nn.Linear)
        self.assertIsInstance(reader.model.language_model.layers['4'], torch.nn.Linear)
        self.assertIsInstance(reader.model.language_model.layers['5'], torch.nn.Linear)
        self.assertIsInstance(reader.model.language_model.layers['2'], Int8WeightOnlyContract)
        self.assertIsInstance(reader.lm_head, torch.nn.Linear)

    def test_full_ragged_layout_matches_numpy_and_dense(self):
        generator = torch.Generator().manual_seed(20261021)
        weight = torch.randn(67, 517, generator=generator).bfloat16()
        activation = torch.randn(1, 1, 517, generator=generator).bfloat16()
        bias = torch.randn(67, generator=generator).bfloat16()
        layout = BankOwnedLinear(weight)
        actual = layout(activation, bias)
        expected = bank_tiled_gemv(weight.float().numpy(), activation.flatten().float().numpy(),
                                   bias=bias.float().numpy(), operand_dtype='bf16')
        np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-4, atol=3e-5)
        host = torch.nn.functional.linear(activation, weight, bias)
        numerics = full_matrix_numerics(weight, activation, actual, host, bias=bias, row_chunk=13)
        self.assertTrue(numerics['forward_error_bound_pass'])
        self.assertTrue(numerics['host_bf16_forward_error_bound_pass'])
        self.assertEqual(layout.resident_bytes, 67 * 768 * 2)

    def test_stale_weight_and_invalid_activation_fail_closed(self):
        weight = torch.ones(7, 13, dtype=torch.bfloat16)
        layout = BankOwnedLinear(weight)
        with self.assertRaises(ValueError):
            layout(torch.ones(2, 13, dtype=torch.bfloat16))
        with self.assertRaises(ValueError):
            layout(torch.ones(13))
        weight[0, 0] = 2
        with self.assertRaises(ValueError):
            layout(torch.ones(13, dtype=torch.bfloat16))

    def fixture(self):
        reader = torch.nn.Sequential(torch.nn.Linear(13, 7, bias=False)).bfloat16()
        identity = 'parameter:0.weight'
        cost = {'resident_bytes': 7 * 256 * 2, 'host_external_bytes': 300,
                'pim_external_bytes': 100,
                'phase_costs': {'host': {'min_cycles': 12, 'max_cycles': 12},
                               'pim': {'min_cycles': 2, 'max_cycles': 2},
                               'placement': {'min_cycles': 15, 'max_cycles': 15}}}
        residency = OnlineResidency({identity: cost}, {identity: {'resident_bytes': cost['resident_bytes']}},
                                     capacity_bytes=8192)
        return reader, residency

    def test_actual_dispatch_cold_host_then_replace_and_restore(self):
        reader, residency = self.fixture()
        activation = torch.ones(1, 1, 13, dtype=torch.bfloat16)
        original = reader[0].forward
        with DecodeBankDispatcher(reader, residency, mode='replace') as dispatcher:
            reader(torch.ones(4, 13, dtype=torch.bfloat16))
            self.assertEqual(dispatcher.calls, 0)
            with dispatcher.decode(observe=True):
                reader(activation)
            self.assertEqual(dispatcher.host_calls, 1)
            with dispatcher.decode(observe=True):
                actual = reader(activation)
            self.assertEqual(dispatcher.replaced_host_calls, 1)
            self.assertEqual(len(dispatcher.records), 1)
            self.assertEqual(actual.shape, (1, 1, 7))
            self.assertTrue(dispatcher.records[0]['numerics']['forward_error_bound_pass'])
            with self.assertRaises(ValueError):
                with dispatcher.decode():
                    reader(torch.ones(2, 13, dtype=torch.bfloat16))
            self.assertFalse(dispatcher.active)
        self.assertEqual(reader[0].forward, original)

    def test_shadow_never_changes_reader_output(self):
        reader, residency = self.fixture()
        activation = torch.randn(1, 1, 13).bfloat16()
        expected = reader(activation)
        with DecodeBankDispatcher(reader, residency, mode='shadow') as dispatcher:
            for position in range(3):
                with dispatcher.decode(observe=position == 1):
                    self.assertTrue(torch.equal(reader(activation), expected))
            self.assertEqual(dispatcher.replaced_host_calls, 0)
            self.assertEqual(len(dispatcher.records), 1)
            self.assertEqual(dispatcher.pim_calls, 2)

    def test_nonfinite_matrix_and_partial_output_rejected(self):
        with self.assertRaises(ValueError):
            BankOwnedLinear(torch.full((3, 5), float('nan'), dtype=torch.bfloat16))
        with self.assertRaises(ValueError):
            full_matrix_numerics(torch.ones(3, 5).bfloat16(), torch.ones(5),
                                 torch.ones(2), torch.ones(3))


if __name__ == '__main__':
    unittest.main()
