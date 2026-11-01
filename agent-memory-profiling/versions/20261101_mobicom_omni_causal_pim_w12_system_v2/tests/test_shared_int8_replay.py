from collections import Counter
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from replay_shared_int8_store import count_range, physical_bank_scores
from qualify_lpddr_materialization import TRANSFER_BYTES, rd16_coordinates


class SharedInt8ReplayTests(unittest.TestCase):
    def test_physical_partial_reduction_matches_whole_integer_scores(self):
        rng = np.random.default_rng(20261019)
        for dimension in (32, 1024, 2048):
            ordinals = [0, 1, 17, 69, 1184]
            values = rng.integers(-128, 128, size=(len(ordinals), dimension), dtype=np.int8)
            query = rng.integers(-128, 128, size=dimension, dtype=np.int8)
            scores, native_counts = physical_bank_scores(values, query, ordinals)
            self.assertTrue(np.array_equal(scores, values.astype(np.int32) @ query.astype(np.int32)))
            expected = Counter()
            for ordinal in ordinals:
                count_range(expected, ordinal * dimension, dimension)
            self.assertEqual(native_counts, dict(expected))
            self.assertEqual(sum(native_counts.values()), values.nbytes // TRANSFER_BYTES)

    def test_bank_read_distribution_is_from_address_not_local_vector_index(self):
        values = np.ones((2, 1024), dtype=np.int8)
        query = np.ones(1024, dtype=np.int8)
        _, first = physical_bank_scores(values, query, [0, 1])
        _, second = physical_bank_scores(values, query, [128, 129])
        self.assertEqual(first, second)
        _, adjacent = physical_bank_scores(values, query, [4, 5])
        self.assertNotEqual(first, adjacent)
        self.assertEqual(rd16_coordinates(0)[0], 0)
        self.assertEqual(rd16_coordinates(32)[0], 1)

    def test_invalid_physical_operands_are_rejected(self):
        values = np.ones((2, 1024), dtype=np.int8)
        query = np.ones(1024, dtype=np.int8)
        for ordinals in ([0, 0], [-1, 1], [0, 2**21]):
            with self.assertRaises(ValueError):
                physical_bank_scores(values, query, ordinals)
        with self.assertRaises(ValueError):
            physical_bank_scores(values.astype(np.int32), query, [0, 1])
        with self.assertRaises(ValueError):
            count_range(Counter(), 1, 32)


if __name__ == '__main__':
    unittest.main()
