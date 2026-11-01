from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from probe_joint_patch_controls import patch_linear
from omni_patch_projection import configure_patch_projection


class JointPatchControlsTests(unittest.TestCase):
    def test_shared_projection_preserves_weights_and_cpu_reference(self):
        visual = torch.nn.Module()
        visual.patch_embed = torch.nn.Module()
        visual.patch_embed.proj = torch.nn.Conv3d(3, 5, kernel_size=2, stride=2).double()
        projection = visual.patch_embed.proj
        pixels = torch.randn(7, 3, 2, 2, 2, dtype=torch.float64)
        expected = projection(pixels).reshape(7, 5)
        weight = projection.weight
        configure_patch_projection(visual, 'linear')
        self.assertIs(projection.weight, weight)
        self.assertTrue(torch.allclose(projection(pixels), expected, atol=1e-12, rtol=1e-12))

    def test_shared_projection_conv_is_unchanged_and_invalid_mode_rejected(self):
        visual = torch.nn.Module()
        visual.patch_embed = torch.nn.Module()
        visual.patch_embed.proj = torch.nn.Conv3d(3, 5, kernel_size=2, stride=2)
        configure_patch_projection(visual, 'conv')
        self.assertNotIn('forward', visual.patch_embed.proj.__dict__)
        with self.assertRaises(ValueError):
            configure_patch_projection(visual, 'invalid')

    def test_full_patch_linear_matches_cpu_conv(self):
        torch.manual_seed(11)
        projection = torch.nn.Conv3d(3, 5, kernel_size=(2, 3, 3), stride=(2, 3, 3)).double()
        pixels = torch.randn(7, 54, dtype=torch.float64)
        reference = projection(pixels.reshape(7, 3, 2, 3, 3)).reshape(7, 5)
        self.assertTrue(torch.allclose(patch_linear(pixels, projection), reference, atol=1e-12, rtol=1e-12))

    def test_grouped_projection_rejected(self):
        projection = torch.nn.Conv3d(2, 4, kernel_size=2, stride=2, groups=2)
        with self.assertRaisesRegex(ValueError, 'nonoverlapping'):
            patch_linear(torch.ones(2, 16), projection)

    def test_padded_projection_rejected(self):
        projection = torch.nn.Conv3d(2, 4, kernel_size=2, stride=2, padding=1)
        with self.assertRaisesRegex(ValueError, 'nonoverlapping'):
            patch_linear(torch.ones(2, 16), projection)

    def test_overlapping_projection_rejected(self):
        projection = torch.nn.Conv3d(2, 4, kernel_size=2, stride=1)
        with self.assertRaisesRegex(ValueError, 'nonoverlapping'):
            patch_linear(torch.ones(2, 16), projection)


if __name__ == '__main__':
    unittest.main()
