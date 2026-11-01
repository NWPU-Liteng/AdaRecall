import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pim_backend.functional_decode import Int12WeightOnlyContract
from pim_backend.w12_bank_layout import W12BankLayout


class W12BankLayoutTests(unittest.TestCase):
    def test_bank_bf16_reference_matches_contract_exactly(self):
        torch.manual_seed(7)
        weight = torch.randn(37, 256, dtype=torch.bfloat16)
        bias = torch.randn(37, dtype=torch.bfloat16)
        activation = torch.randn(256, dtype=torch.bfloat16)
        layout = W12BankLayout(
            weight, group_size=128, bank_count=8, tile_rows=16, bias=bias)
        contract = Int12WeightOnlyContract(
            weight, bias, group_size=128, row_chunk=256)
        expected = contract(activation.unsqueeze(0)).squeeze(0)
        actual = layout.gemv(activation, numeric_mode="bf16_reference")
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_bank_fp32_accumulation_stays_inside_declared_envelope(self):
        torch.manual_seed(11)
        weight = torch.randn(300, 256, dtype=torch.bfloat16)
        activation = torch.randn(256, dtype=torch.bfloat16)
        layout = W12BankLayout(weight, group_size=128, bank_count=8)
        contract = Int12WeightOnlyContract(
            weight, group_size=128, row_chunk=128)
        expected = contract(activation.unsqueeze(0)).squeeze(0).float()
        actual = layout.gemv(activation, numeric_mode="fp32_bank").float()
        relative_l2 = ((actual - expected).norm()
                       / expected.norm().clamp_min(1e-30)).item()
        self.assertLess(relative_l2, 5e-3)
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=0.25)

    def test_row_packing_uses_two_physical_rows_for_2560_columns(self):
        torch.manual_seed(13)
        layout = W12BankLayout(
            torch.randn(17, 2560, dtype=torch.bfloat16),
            group_size=128, bank_count=4)
        inventory = layout.inventory()
        self.assertEqual(inventory["logical_row_bytes"], 3840)
        self.assertEqual(inventory["physical_row_bytes"], 4096)
        self.assertEqual(inventory["physical_rows_per_output_row"], 2)
        self.assertEqual(inventory["weight_copies"], 1)
        self.assertGreater(inventory["resident_bytes"],
                           inventory["logical_bytes"])

    def test_tail_rows_and_multi_round_bank_placement(self):
        torch.manual_seed(17)
        weight = torch.randn(301, 256, dtype=torch.bfloat16)
        activation = torch.randn(256, dtype=torch.bfloat16)
        layout = W12BankLayout(weight, group_size=128, bank_count=4,
                               tile_rows=16)
        inventory = layout.inventory()
        self.assertEqual(inventory["active_banks"], 4)
        self.assertEqual(inventory["rounds"], 5)
        self.assertGreater(inventory["padded_rows"], 301)
        contract = Int12WeightOnlyContract(
            weight, group_size=128, row_chunk=64)
        expected = contract(activation.unsqueeze(0)).squeeze(0)
        actual = layout.gemv(activation, numeric_mode="bf16_reference")
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_linear_fallback_is_explicit_and_exact(self):
        torch.manual_seed(19)
        weight = torch.randn(37, 256, dtype=torch.bfloat16)
        bias = torch.randn(37, dtype=torch.bfloat16)
        activation = torch.randn(5, 256, dtype=torch.bfloat16)
        layout = W12BankLayout(weight, group_size=128, bank_count=8, bias=bias)
        contract = Int12WeightOnlyContract(
            weight, bias, group_size=128, row_chunk=64)
        torch.testing.assert_close(
            layout.linear(activation), contract(activation), rtol=0, atol=0)

    def test_no_bias_path_matches_contract_bitwise(self):
        torch.manual_seed(23)
        weight = torch.randn(2560, 4096, dtype=torch.bfloat16)
        activation = torch.randn(4096, dtype=torch.bfloat16)
        layout = W12BankLayout(
            weight, group_size=128, bank_count=256, tile_rows=16)
        contract = Int12WeightOnlyContract(
            weight, group_size=128, row_chunk=4096)
        torch.testing.assert_close(
            layout.gemv(activation, numeric_mode="bf16_reference"),
            contract(activation.unsqueeze(0)).squeeze(0), rtol=0, atol=0)
        rows = torch.randn(2, 4096, dtype=torch.bfloat16)
        torch.testing.assert_close(
            layout.linear(rows), contract(rows), rtol=0, atol=0)

    def test_invalid_activation_fails_closed(self):
        layout = W12BankLayout(
            torch.randn(4, 128, dtype=torch.bfloat16),
            group_size=128, bank_count=2)
        with self.assertRaises(ValueError):
            layout.gemv(torch.randn(128, dtype=torch.float32))
        with self.assertRaises(ValueError):
            layout.linear(torch.randn(3, 64, dtype=torch.bfloat16))


if __name__ == '__main__':
    unittest.main()
