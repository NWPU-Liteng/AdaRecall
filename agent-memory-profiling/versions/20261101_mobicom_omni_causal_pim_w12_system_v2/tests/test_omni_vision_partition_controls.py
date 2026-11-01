from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from diagnose_omni_vision_partition_controls import capture_stages, stage_comparisons


class PartitionControlTests(unittest.TestCase):
    def test_ordered_stages_and_first_image_slicing(self):
        visual = torch.nn.Module()
        visual.patch_embed = torch.nn.Identity()
        visual.blocks = torch.nn.ModuleList([torch.nn.Identity()])
        visual.merger = torch.nn.Identity()
        captured, hooks = capture_stages(visual, 2, 1)
        with torch.inference_mode():
            value = torch.arange(12, dtype=torch.float32).reshape(4, 3)
            visual.patch_embed(value)
            visual.blocks[0](value)
            visual.merger(value)
        for hook in hooks:
            hook.remove()
        self.assertEqual(list(captured), ['patch_embed', 'block_0', 'merger'])
        self.assertEqual(captured['patch_embed'].shape, (2, 3))
        self.assertEqual(captured['merger'].shape, (1, 3))
        self.assertTrue(all(row['bitwise_equal'] for row in stage_comparisons(captured, captured)))

    def test_missing_or_reordered_stages_fail(self):
        value = torch.ones(1, 2)
        with self.assertRaises(ValueError):
            stage_comparisons({'patch': value}, {})
        with self.assertRaises(ValueError):
            stage_comparisons({'patch': value, 'block': value}, {'block': value, 'patch': value})

    def test_layer_difference_stays_visible(self):
        reference = {'patch': torch.ones(1, 2), 'block': torch.ones(1, 2)}
        candidate = {'patch': torch.ones(1, 2), 'block': torch.zeros(1, 2)}
        rows = stage_comparisons(reference, candidate)
        self.assertTrue(rows[0]['bitwise_equal'])
        self.assertFalse(rows[1]['bitwise_equal'])


if __name__ == '__main__':
    unittest.main()
