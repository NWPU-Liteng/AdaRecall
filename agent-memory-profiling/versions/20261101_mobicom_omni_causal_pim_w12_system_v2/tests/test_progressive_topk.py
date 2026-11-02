from pathlib import Path
import sys
import unittest

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pim_backend.functional_decode import quantize_signed_groups
from pim_backend.progressive_topk import (decode_codes, from_codes,
                                           progressive_topk, quantize_plane_major,
                                           reference_scores)
from pim_backend.progressive_topk import _scores_and_bounds


SCHEDULES = ([1] * 12, [4, 2, 2, 2, 2], [12])


def independent_scores(matrix, activation, rows=None):
    codes = decode_codes(matrix)
    groups = matrix.cols // matrix.group_size
    if rows is None:
        rows = np.arange(len(codes))
    grouped = codes[rows].astype(np.float64).reshape(
        len(rows), groups, matrix.group_size)
    activation = activation.reshape(1, groups, matrix.group_size)
    return ((grouped * activation).sum(axis=2) * matrix.scales[rows]).sum(axis=1)


def full_topk(matrix, activation, k, rows=None):
    if rows is None:
        rows = np.arange(matrix.planes.shape[1])
    else:
        rows = np.flatnonzero(rows) if np.asarray(rows).dtype == np.bool_ else np.asarray(rows)
    scores = independent_scores(matrix, activation, rows)
    order = np.lexsort((rows, -scores))[:min(k, len(rows))]
    return rows[order], scores[order]


def random_shapes():
    shapes = []
    shapes.extend([(1, 128), (7, 128), (300, 128)] * 53)
    shapes.extend([(7, 1024), (300, 1024)] * 10)
    shapes.extend([(7, 2560), (300, 2560)] * 5)
    shapes.extend([(5000, 128)] * 10)
    shapes.extend([(5000, 1024)] * 5)
    shapes.extend([(5000, 2560)] * 4)
    return shapes[:200]


class ProgressiveTopKTests(unittest.TestCase):
    def test_quantization_round_trip_uses_v1_codes(self):
        rng = np.random.default_rng(7)
        weight = rng.normal(size=(37, 256)).astype(np.float32)
        matrix = quantize_plane_major(weight)
        expected, scales, _ = quantize_signed_groups(
            torch.from_numpy(weight), group_size=128, weight_bits=12)
        np.testing.assert_array_equal(decode_codes(matrix), expected.numpy())
        np.testing.assert_array_equal(matrix.scales, scales.numpy())
        self.assertEqual(matrix.planes.shape, (12, 37, 32))
        self.assertEqual(matrix.codes_sha256, from_codes(
            expected.numpy(), scales.numpy(), 12, 128).codes_sha256)

    def test_reference_scores_match_independent_grouped_gemm(self):
        rng = np.random.default_rng(11)
        matrix = quantize_plane_major(rng.normal(size=(300, 2560)).astype(np.float32))
        activation = rng.standard_t(4, size=2560)
        rows = np.asarray([0, 4, 17, 299])
        np.testing.assert_array_equal(
            reference_scores(matrix, activation, rows=rows),
            independent_scores(matrix, activation, rows))

    def test_two_hundred_random_exactness_cases(self):
        schedules = SCHEDULES
        ks = (1, 5, 48, 80)
        for seed, shape in enumerate(random_shapes()):
            rng = np.random.default_rng(10_000 + seed)
            if seed % 2:
                weight = rng.standard_t(3, size=shape)
            else:
                weight = rng.normal(size=shape)
            activation = rng.standard_t(5, size=shape[1])
            if seed % 7 == 0:
                activation[::11] = -np.abs(activation[::11])
            matrix = quantize_plane_major(weight.astype(np.float32))
            k = ks[seed % len(ks)]
            schedule = schedules[seed % len(schedules)]
            indices, scores, stats = progressive_topk(
                matrix, activation, k, plane_schedule=schedule)
            expected_indices, expected_scores = full_topk(matrix, activation, k)
            np.testing.assert_array_equal(indices, expected_indices)
            np.testing.assert_array_equal(scores, expected_scores)
            self.assertLessEqual(stats['plane_row_reads'], stats['full_plane_row_reads'])
            self.assertLessEqual(stats['weight_bytes_read'], stats['weight_bytes_full'])
            self.assertEqual(stats['allowed_rows'], shape[0])

    def test_ties_duplicates_and_zero_rows_use_index_order(self):
        activation = np.linspace(-1.5, 2.0, 128)
        duplicate = np.tile(np.arange(128, dtype=np.float32), (6, 1))
        matrix = quantize_plane_major(duplicate)
        indices, scores, _ = progressive_topk(matrix, activation, 4)
        expected_indices, expected_scores = full_topk(matrix, activation, 4)
        np.testing.assert_array_equal(indices, expected_indices)
        np.testing.assert_array_equal(scores, expected_scores)
        self.assertEqual(indices.tolist(), [0, 1, 2, 3])

        zero = quantize_plane_major(np.zeros((5, 128), dtype=np.float32))
        indices, scores, _ = progressive_topk(zero, activation, 3)
        self.assertEqual(indices.tolist(), [0, 1, 2])
        self.assertTrue(np.array_equal(scores, np.zeros(3)))

    def test_row_mask_zero_rows_and_fewer_rows_than_k(self):
        rng = np.random.default_rng(23)
        matrix = quantize_plane_major(rng.normal(size=(7, 128)).astype(np.float32))
        activation = rng.normal(size=128)
        mask = np.zeros(7, dtype=bool)
        mask[3] = True
        indices, scores, stats = progressive_topk(matrix, activation, 5, row_mask=mask)
        expected_indices, expected_scores = full_topk(matrix, activation, 5, mask)
        np.testing.assert_array_equal(indices, expected_indices)
        np.testing.assert_array_equal(scores, expected_scores)
        self.assertEqual(stats['allowed_rows'], 1)
        self.assertEqual(stats['full_plane_row_reads'], 12)
        with self.assertRaisesRegex(ValueError, 'at least one allowed row'):
            progressive_topk(matrix, activation, 5, row_mask=np.zeros(7, dtype=bool))

        mask[:] = False
        mask[[1, 6]] = True
        indices, _, _ = progressive_topk(matrix, activation, 5, row_mask=mask)
        self.assertEqual(indices.tolist(), full_topk(matrix, activation, 5, mask)[0].tolist())

    def test_all_plane_schedules_and_read_accounting(self):
        rng = np.random.default_rng(29)
        matrix = quantize_plane_major(rng.normal(size=(300, 1024)).astype(np.float32))
        activation = rng.standard_t(4, size=1024)
        expected = full_topk(matrix, activation, 5)
        for schedule in SCHEDULES:
            indices, scores, stats = progressive_topk(
                matrix, activation, 5, plane_schedule=schedule)
            np.testing.assert_array_equal(indices, expected[0])
            np.testing.assert_array_equal(scores, expected[1])
            self.assertEqual(sum(item['planes_in_round'] for item in stats['rounds']), 12)
            self.assertEqual(stats['full_plane_row_reads'], 300 * 12)
            self.assertLessEqual(stats['plane_row_reads'], 300 * 12)
            self.assertEqual(
                stats['weight_bytes_read'],
                stats['plane_row_reads'] * ((1024 + 7) // 8))

    def test_bounds_hold_for_every_plane(self):
        rng = np.random.default_rng(31)
        matrix = quantize_plane_major(rng.standard_t(3, size=(300, 2560)).astype(np.float32))
        activation = rng.standard_t(4, size=2560)
        reference = independent_scores(matrix, activation)
        groups = matrix.cols // matrix.group_size
        grouped = activation.reshape(groups, matrix.group_size)
        positive = matrix.scales.dot(np.maximum(grouped, 0).sum(axis=1))
        negative = matrix.scales.dot(np.minimum(grouped, 0).sum(axis=1))
        rows = np.arange(300)
        for stop_plane in range(1, 13):
            _, lower, upper = _scores_and_bounds(
                matrix, activation, rows, stop_plane, positive, negative)
            self.assertTrue(np.all(lower <= reference + 1e-9))
            self.assertTrue(np.all(reference <= upper + 1e-9))

    def test_invalid_inputs_are_rejected(self):
        matrix = quantize_plane_major(np.ones((2, 128), dtype=np.float32))
        activation = np.ones(128)
        with self.assertRaisesRegex(ValueError, 'positive integer k'):
            progressive_topk(matrix, activation, 0)
        with self.assertRaisesRegex(ValueError, 'sum to bits'):
            progressive_topk(matrix, activation, 1, plane_schedule=[2, 2])
        with self.assertRaisesRegex(ValueError, 'codes must fit'):
            from_codes(np.asarray([[2048] * 128], dtype=np.int32),
                       np.ones((1, 1)), 12, 128)


if __name__ == '__main__':
    unittest.main()
