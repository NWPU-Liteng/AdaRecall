from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pim_backend.functional_decode import Int8LinearContract, materialize_int8_primary, quantize_int8_rows


class Int8ReaderContractTests(unittest.TestCase):
    def fixture(self):
        reader = torch.nn.Module()
        reader.model = torch.nn.Module()
        reader.model.language_model = torch.nn.Sequential(torch.nn.Linear(32, 32))
        reader.lm_head = torch.nn.Linear(32, 16, bias=False)
        return reader.eval()

    def test_primary_replaces_bf16_not_a_second_resident_copy(self):
        reader = self.fixture()
        summary = materialize_int8_primary(reader)
        self.assertEqual(summary['modules'], 2)
        self.assertFalse(summary['source_weight_copies_retained_by_converter'])
        self.assertIsInstance(reader.model.language_model[0], Int8LinearContract)
        self.assertEqual(reader.lm_head.weight.dtype, torch.int8)
        self.assertLess(summary['integer_primary_bytes'], summary['replaced_source_parameter_bytes'])
        self.assertEqual(reader.lm_head(reader.model.language_model(torch.ones(1, 32))).shape, (1, 16))

    def test_embedding_alias_is_retained_and_charged(self):
        reader = self.fixture()
        reader.embedding = torch.nn.Embedding(16, 32)
        reader.lm_head.weight = reader.embedding.weight
        original = reader.embedding.weight
        summary = materialize_int8_primary(reader)
        self.assertIs(reader.embedding.weight, original)
        self.assertIsInstance(reader.lm_head, Int8LinearContract)
        self.assertEqual(summary['retained_bf16_embedding_alias_bytes'], 16 * 32 * 4)
        self.assertEqual(summary['released_bf16_parameter_bytes'],
                         summary['replaced_source_parameter_bytes'] - 16 * 32 * 4)

    def test_other_parameter_alias_fails_before_replacement(self):
        reader = self.fixture()
        reader.extra = torch.nn.Linear(32, 16)
        reader.extra.weight = reader.lm_head.weight
        with self.assertRaises(ValueError):
            materialize_int8_primary(reader)
        self.assertIsInstance(reader.lm_head, torch.nn.Linear)

    def test_zero_rows_and_full_range_have_finite_scales(self):
        value = torch.tensor([[0., 0.], [-2., 1.]])
        quantized, scales = quantize_int8_rows(value)
        self.assertTrue(torch.equal(quantized[0], torch.zeros(2, dtype=torch.int8)))
        self.assertEqual(int(quantized[1, 0]), -127)
        self.assertTrue(torch.isfinite(scales).all())

    def test_block_quantization_is_bitwise_equal_to_whole_matrix(self):
        generator = torch.Generator().manual_seed(20261023)
        weight = torch.randn(2053, 32, generator=generator).bfloat16()
        expected_weight, expected_scales = quantize_int8_rows(weight)
        layout = Int8LinearContract(weight)
        self.assertTrue(torch.equal(layout.weight, expected_weight))
        self.assertTrue(torch.equal(layout.scales, expected_scales))

    def test_shared_selected_weight_keeps_one_integer_layout(self):
        reader = self.fixture()
        reader.model.language_model[0] = torch.nn.Linear(32, 16, bias=False)
        reader.lm_head.weight = reader.model.language_model[0].weight
        summary = materialize_int8_primary(reader)
        self.assertEqual(summary['unique_integer_layouts'], 1)
        self.assertIs(reader.lm_head, reader.model.language_model[0])

    def test_integer_product_matches_independent_int64_and_preserves_shape(self):
        generator = torch.Generator().manual_seed(42)
        weight = torch.randn(17, 32, generator=generator)
        layout = Int8LinearContract(weight, torch.ones(17), output_partitions=2)
        activation = torch.randn(2, 3, 32, generator=generator)
        quantized, scales = quantize_int8_rows(activation.reshape(-1, 32))
        reference = quantized.long() @ layout.weight.t().long()
        self.assertTrue(torch.equal(layout.integer_product(quantized).long(), reference))
        output = reference.float() * (scales * layout.scales.t()) + layout.bias
        self.assertTrue(torch.equal(layout(activation), output.reshape(2, 3, 17)))
        self.assertEqual(layout.storage_bytes, 17 * 32 + 17 * 4 * 2)

    def test_invalid_inputs_fail_closed(self):
        for weight in (torch.ones(3, 7), torch.ones(1, 131072)):
            with self.assertRaises(ValueError):
                Int8LinearContract(weight)
        with self.assertRaises(ValueError):
            quantize_int8_rows(torch.tensor([[float('nan')]]))
        with self.assertRaises(ValueError):
            Int8LinearContract(torch.ones(2, 8), torch.ones(3))


if __name__ == '__main__':
    unittest.main()
