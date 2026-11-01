from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from omni_execution_input_contract import execution_input_contract
from omni_visual_feature_store import VisualFeatureStore


class ExecutionInputContractTests(unittest.TestCase):
    def fixture(self):
        inputs = {'input_ids': torch.tensor([[1, 2]]), 'pixel_values': torch.zeros((4, 3)),
                  'image_grid_thw': torch.tensor([[1, 2, 2]])}
        original = {'processor_inputs_sha256': {name: VisualFeatureStore.tensor_digest(value)
                    for name, value in inputs.items()}, 'images': [{'sha256': 'image'}],
                    'embedding_producers': [{'producer_sha256': 'producer', 'output_sha256': 'output'}]}
        return inputs, original

    def test_complete_identity_and_provenance(self):
        inputs, original = self.fixture()
        contract = execution_input_contract(inputs, original)
        self.assertEqual(contract['processor_inputs_sha256'], original['processor_inputs_sha256'])
        self.assertEqual(contract['embedding_producer_outputs'][0]['output_sha256'], 'output')
        self.assertFalse(contract['reader_state_cross_request_reuse_executed'])

    def test_same_shape_different_pixels_rejected(self):
        inputs, original = self.fixture()
        inputs['pixel_values'][0, 0] = 1.
        with self.assertRaises(ValueError):
            execution_input_contract(inputs, original)

    def test_missing_operand_rejected(self):
        inputs, original = self.fixture()
        del inputs['image_grid_thw']
        with self.assertRaises(ValueError):
            execution_input_contract(inputs, original)


if __name__ == '__main__':
    unittest.main()
