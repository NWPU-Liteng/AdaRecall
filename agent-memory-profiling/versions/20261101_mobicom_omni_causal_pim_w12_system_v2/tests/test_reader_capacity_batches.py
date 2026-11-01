import sys
from pathlib import Path
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

from run_reader_capacity_batches import cache_tensor_bytes, collate_inputs


class ReaderCapacityBatchTests(unittest.TestCase):
    def test_collates_unequal_text_inputs_with_left_padding(self):
        batch = collate_inputs([
            {'input_ids': torch.tensor([[10, 11]]),
             'attention_mask': torch.tensor([[1, 1]])},
            {'input_ids': torch.tensor([[12, 13, 14]]),
             'attention_mask': torch.tensor([[1, 1, 1]])},
        ], pad_token_id=0)
        self.assertEqual(batch['input_ids'].tolist(), [[0, 10, 11], [12, 13, 14]])
        self.assertEqual(batch['attention_mask'].tolist(), [[0, 1, 1], [1, 1, 1]])

    def test_collates_mixed_text_and_image_inputs(self):
        batch = collate_inputs([
            {'input_ids': torch.tensor([[1, 2]]),
             'attention_mask': torch.tensor([[1, 1]])},
            {'input_ids': torch.tensor([[3, 4, 5]]),
             'attention_mask': torch.tensor([[1, 1, 1]]),
             'pixel_values': torch.ones((2, 3, 4, 4)),
             'image_grid_thw': torch.tensor([[1, 2, 2]])},
        ], pad_token_id=0)
        self.assertEqual(batch['input_ids'].shape, (2, 3))
        self.assertEqual(batch['pixel_values'].shape, (2, 3, 4, 4))
        self.assertEqual(batch['image_grid_thw'].tolist(), [[1, 2, 2]])

    def test_cache_byte_accounting_can_filter_by_device(self):
        cache = {'keys': torch.ones((2, 3)), 'values': (torch.ones((4,)),)}
        self.assertEqual(cache_tensor_bytes(cache), 10 * torch.tensor([], dtype=torch.float32).element_size())
        self.assertEqual(cache_tensor_bytes(cache, device='cpu'), cache_tensor_bytes(cache))
        self.assertEqual(cache_tensor_bytes(cache, device='cuda'), 0)


if __name__ == '__main__':
    unittest.main()
