import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from omni_joint_vision_contract import compare_partition_outputs


class JointVisionContractTests(unittest.TestCase):
    def outputs(self):
        first, second = torch.ones(3, 4), torch.zeros(2, 4)
        reference = SimpleNamespace(pooler_output=(first, second),
            deepstack_features=[torch.cat((first, second)), torch.cat((first + 2, second + 2))])
        partitions = [SimpleNamespace(pooler_output=(value,), deepstack_features=[value, value + 2])
                      for value in (first, second)]
        return reference, partitions

    def test_concatenated_token_count_is_not_image_count(self):
        pooled, deepstack = compare_partition_outputs(*self.outputs())
        self.assertEqual(len(pooled), 2)
        self.assertEqual(len(deepstack), 2)
        self.assertTrue(all(value['bitwise_equal'] for value in pooled + deepstack))

    def test_missing_image_fails_closed(self):
        reference, partitions = self.outputs()
        with self.assertRaisesRegex(ValueError, 'image counts'):
            compare_partition_outputs(reference, partitions[:1])

    def test_missing_stage_fails_closed(self):
        reference, partitions = self.outputs()
        partitions[1].deepstack_features.pop()
        with self.assertRaisesRegex(ValueError, 'stage counts'):
            compare_partition_outputs(reference, partitions)

    def test_deepstack_drift_is_not_hidden_by_identical_pooled_outputs(self):
        reference, partitions = copy.deepcopy(self.outputs())
        partitions[1].deepstack_features[1][0, 0] += 1
        pooled, deepstack = compare_partition_outputs(reference, partitions)
        self.assertTrue(all(value['bitwise_equal'] for value in pooled))
        self.assertFalse(deepstack[1]['bitwise_equal'])

    def test_wrong_token_count_is_not_equal(self):
        reference, partitions = self.outputs()
        partitions[1].deepstack_features[1] = torch.zeros(1, 4)
        _, deepstack = compare_partition_outputs(reference, partitions)
        self.assertFalse(deepstack[1]['same_shape_dtype'])


if __name__ == '__main__':
    unittest.main()
