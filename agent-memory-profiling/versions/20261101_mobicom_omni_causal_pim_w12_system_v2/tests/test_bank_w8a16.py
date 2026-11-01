import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pim_backend.bank_w8a16 import W8A16BankLayout
from pim_backend.functional_decode import (Int8WeightOnlyContract,
                                           Int12WeightOnlyContract,
                                           quantize_signed_groups)


class BankW8A16Tests(unittest.TestCase):
    def test_bank_layout_matches_weight_only_contract(self):
        torch.manual_seed(7)
        weight = torch.randn(37, 256, dtype=torch.bfloat16)
        activation = torch.randn(256, dtype=torch.bfloat16)
        layout = W8A16BankLayout(weight, group_size=128, bank_count=8)
        contract = Int8WeightOnlyContract(weight, group_size=128)
        expected = contract(activation.unsqueeze(0)).squeeze(0)
        actual = layout.gemv(activation)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(layout.inventory()['weight_copies'], 1)
        self.assertGreaterEqual(layout.resident_bytes, layout.logical_bytes)

    def test_bank_layout_rejects_bad_activation(self):
        weight = torch.randn(4, 128, dtype=torch.bfloat16)
        layout = W8A16BankLayout(weight, group_size=128, bank_count=2)
        with self.assertRaises(ValueError):
            layout.gemv(torch.randn(128, dtype=torch.float32))

    def test_uniform_twelve_bit_contract_uses_one_scale_plane(self):
        torch.manual_seed(13)
        weight = torch.randn(37, 256, dtype=torch.bfloat16)
        activation = torch.randn(256, dtype=torch.bfloat16)
        quantized, scales, dequantized = quantize_signed_groups(
            weight, group_size=128, weight_bits=12)
        self.assertEqual(quantized.dtype, torch.int16)
        self.assertLessEqual(quantized.abs().max().item(), 2047)
        expected = (dequantized.float() @ activation.float()).to(torch.bfloat16)
        contract = Int12WeightOnlyContract(weight, group_size=128)
        actual = contract(activation.unsqueeze(0)).squeeze(0)
        torch.testing.assert_close(actual, expected, rtol=0, atol=2e-2)
        self.assertEqual(
            contract.logical_storage_bytes,
            37 * ((256 * 12 + 7) // 8) + scales.numel() * scales.element_size())
        self.assertFalse(hasattr(contract, 'residual_payload_bytes'))

    def test_twelve_bit_quantization_error_is_bounded_by_half_a_group_step(self):
        torch.manual_seed(17)
        weight = torch.randn(9, 256, dtype=torch.bfloat16)
        _, scales, dequantized = quantize_signed_groups(
            weight, group_size=128, weight_bits=12)
        error = (weight.float() - dequantized.float()).abs().reshape(9, 2, 128)
        bound = (scales[:, :, None] / 2).expand_as(error) + 1e-3
        self.assertTrue(torch.all(error <= bound))

    def test_residual_bank_layout_matches_contract(self):
        torch.manual_seed(11)
        weight = torch.randn(37, 256, dtype=torch.bfloat16)
        activation = torch.randn(256, dtype=torch.bfloat16)
        layout = W8A16BankLayout(
            weight, group_size=128, bank_count=8, residual_bits=2)
        contract = Int8WeightOnlyContract(weight, group_size=128, residual_bits=2)
        torch.testing.assert_close(
            layout.gemv(activation), contract(activation.unsqueeze(0)).squeeze(0),
            rtol=0, atol=0)
        inventory = layout.inventory()
        self.assertEqual(inventory['residual_bits'], 2)
        self.assertGreater(inventory['logical_bytes'],
                           weight.numel() + 37 * 2 * 4)


if __name__ == '__main__':
    unittest.main()
