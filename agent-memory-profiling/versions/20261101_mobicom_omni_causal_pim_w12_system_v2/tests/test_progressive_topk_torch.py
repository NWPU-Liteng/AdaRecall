import sys
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pim_backend.progressive_lm_head import ProgressiveLmHeadShadow
from pim_backend.progressive_topk import (progressive_topk,
                                          progressive_topk_torch,
                                          quantize_plane_major,
                                          reference_scores, to_torch_matrix)
from pim_backend.w12_bank_layout import W12BankLayout
from pim_backend.w12_functional_dispatch import W12FunctionalBankLinear


class ProgressiveTopkTorchTests(unittest.TestCase):
    def test_torch_port_matches_numpy_bitwise(self):
        schedules = ((1,) * 12, (4, 2, 2, 2, 2), (12,))
        cases = (
            (7, 128, 1, 3),
            (7, 1024, 5, 11),
            (300, 128, 48, 19),
            (300, 1024, 80, 23),
            (1, 2560, 5, 29),
        )
        for rows, cols, k, seed in cases:
            for schedule in schedules:
                with self.subTest(rows=rows, cols=cols, k=k, schedule=schedule):
                    rng = np.random.default_rng(seed)
                    weight = rng.standard_t(3, size=(rows, cols)).astype(np.float32)
                    activation = rng.standard_normal(cols)
                    matrix = quantize_plane_major(weight, bits=12, group_size=128)
                    torch_matrix = to_torch_matrix(matrix)

                    expected = progressive_topk(
                        matrix, activation, k, plane_schedule=schedule)
                    actual = progressive_topk_torch(
                        torch_matrix, torch.from_numpy(activation), k,
                        plane_schedule=schedule)

                    self.assertTrue(np.array_equal(expected[0], actual[0].numpy()))
                    self.assertTrue(np.array_equal(expected[1], actual[1].numpy()))
                    self.assertEqual(expected[2], actual[2])

    def test_torch_row_masks_and_reference_scores_are_bitwise_equal(self):
        rng = np.random.default_rng(37)
        weight = rng.standard_normal((83, 256)).astype(np.float32)
        activation = rng.standard_normal(256)
        matrix = quantize_plane_major(weight, bits=12, group_size=128)
        torch_matrix = to_torch_matrix(matrix)

        expected_reference = reference_scores(matrix, activation)
        actual_reference = reference_scores(
            torch_matrix, torch.from_numpy(activation))
        self.assertTrue(np.array_equal(
            expected_reference, actual_reference.numpy()))

        masks = (
            np.arange(83) == 17,
            np.asarray([index % 5 for index in range(83)]) != 0,
            np.arange(83) >= 80,
        )
        for mask in masks:
            for k in (1, 5):
                with self.subTest(k=k, allowed=int(mask.sum())):
                    expected = progressive_topk(matrix, activation, k, row_mask=mask)
                    actual = progressive_topk_torch(
                        torch_matrix, torch.from_numpy(activation), k,
                        row_mask=torch.from_numpy(mask))
                    self.assertTrue(np.array_equal(expected[0], actual[0].numpy()))
                    self.assertTrue(np.array_equal(expected[1], actual[1].numpy()))
                    self.assertEqual(expected[2], actual[2])

    def test_torch_integer_rows_keep_index_tie_order(self):
        activation = np.linspace(-1.5, 2.0, 128)
        duplicate = np.tile(np.arange(128, dtype=np.float32), (6, 1))
        matrix = quantize_plane_major(duplicate)
        torch_matrix = to_torch_matrix(matrix)
        rows = np.asarray([5, 3, 1])

        expected = progressive_topk(matrix, activation, 3, row_mask=rows)
        actual = progressive_topk_torch(
            torch_matrix, torch.from_numpy(activation), 3,
            row_mask=torch.from_numpy(rows))

        self.assertEqual(expected[0].tolist(), [1, 3, 5])
        self.assertTrue(np.array_equal(expected[0], actual[0].numpy()))
        self.assertTrue(np.array_equal(expected[1], actual[1].numpy()))
        self.assertEqual(expected[2], actual[2])


class ProgressiveLmHeadShadowTests(unittest.TestCase):
    def _wrapper(self):
        torch.manual_seed(41)
        weight = torch.randn(8, 128, dtype=torch.bfloat16)
        layout = W12BankLayout(
            weight, group_size=128, bank_count=2, tile_rows=4)
        wrapper = W12FunctionalBankLinear(
            'lm_head', layout, mode='replace', numeric_mode='bf16_reference',
            fail_on_decode_fallback=True)
        wrapper.decode = True
        return wrapper

    def test_shadow_counts_scheduled_and_forced_tokens(self):
        wrapper = self._wrapper()
        activation = torch.randn(1, 1, 128, dtype=torch.bfloat16)
        with ProgressiveLmHeadShadow(
                wrapper, vocab_rows=8, valid_vocab_size=8,
                shadow_every=2) as shadow:
            output = wrapper(activation)
            actual = int(output.argmax().item())
            allowed = torch.ones(8, dtype=torch.bool)
            self.assertTrue(shadow.should_validate(0, False))
            self.assertFalse(shadow.should_validate(1, False))
            self.assertTrue(shadow.should_validate(1, True))

            shadow.observe(
                phase='answer', actual_token=actual,
                allowed=allowed, forced=False)
            forced = torch.zeros(8, dtype=torch.bool)
            forced[actual] = True
            shadow.observe(
                phase='answer', actual_token=actual,
                allowed=forced, forced=True)
            self.assertTrue(shadow.summary()['enabled'])

        summary = shadow.summary()
        phase = summary['phases']['answer']
        self.assertEqual(phase['checked_tokens'], 2)
        self.assertEqual(phase['forced_tokens'], 1)
        self.assertEqual(phase['progressive_calls'], 1)
        self.assertEqual(phase['agree_bf16'] + phase['disagree_bf16'], 1)
        self.assertGreater(phase['weight_bytes_read'], 0)
        self.assertLessEqual(
            phase['weight_bytes_read'], phase['weight_bytes_full'])
        self.assertIsNotNone(phase['average_rounds'])
        self.assertFalse(summary['enabled'])

    def test_bitmask_unpack_excludes_padding_rows(self):
        wrapper = self._wrapper()
        with ProgressiveLmHeadShadow(
                wrapper, vocab_rows=8, valid_vocab_size=5) as shadow:
            mask = np.zeros((1, 1), dtype=np.uint32)
            mask[0, 0] = (1 << 0) | (1 << 3) | (1 << 7)
            allowed = shadow.allowed_from_bitmask(mask)
            self.assertEqual([int(value) for value in allowed.nonzero().flatten()],
                             [0, 3])
            self.assertEqual(
                [int(value) for value in shadow.full_allowed().nonzero().flatten()],
                [0, 1, 2, 3, 4])

    def test_disabled_summary_has_all_phases(self):
        summary = ProgressiveLmHeadShadow.disabled_summary()
        self.assertFalse(summary['enabled'])
        self.assertEqual(summary['mode'], 'off')
        self.assertEqual(
            set(summary['phases']),
            {'contract_parse', 'witness_construct', 'answer'})


if __name__ == '__main__':
    unittest.main()
