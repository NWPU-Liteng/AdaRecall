from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_omni_vision_partition_probe import compare_features


class VisionPartitionTests(unittest.TestCase):
    def test_equal_numeric_zero_with_different_bits_is_not_exact(self):
        positive = torch.tensor([[0.]], dtype=torch.bfloat16)
        negative = torch.tensor([[-0.]], dtype=torch.bfloat16)
        self.assertTrue(torch.equal(positive, negative))
        self.assertFalse(compare_features(positive, negative)['bitwise_equal'])

    def test_dtype_and_shape_mismatch_and_nonfinite_values_fail(self):
        value = torch.ones((2, 3), dtype=torch.bfloat16)
        self.assertTrue(compare_features(value, value.clone())['bitwise_equal'])
        self.assertFalse(compare_features(value, value.float())['bitwise_equal'])
        self.assertFalse(compare_features(value, value[:1])['bitwise_equal'])
        bad = torch.tensor([[float('nan')]])
        self.assertFalse(compare_features(bad, bad)['bitwise_equal'])


if __name__ == '__main__':
    unittest.main()
