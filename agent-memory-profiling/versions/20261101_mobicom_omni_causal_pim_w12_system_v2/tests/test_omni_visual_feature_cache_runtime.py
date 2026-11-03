from pathlib import Path
import sys
import tempfile
import unittest

import torch
from transformers.modeling_outputs import BaseModelOutputWithPooling

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from omni_visual_feature_cache_runtime import (VisualFeatureCache,
                                               create_visual_feature_cache,
                                               visual_feature_identities)


class FakeImageProcessor:
    def __init__(self, longest_edge=262144):
        self.longest_edge = longest_edge

    def to_dict(self):
        return {'size': {'longest_edge': self.longest_edge}}


class FakeProcessor:
    def __init__(self, longest_edge=262144):
        self.image_processor = FakeImageProcessor(longest_edge)

    def to_dict(self):
        return {'image_processor': self.image_processor.to_dict(),
                'chat_template': 'fixture'}


class FakeVisual(torch.nn.Module):
    dtype = torch.float32
    spatial_merge_size = 1


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.training = False
        self.visual = FakeVisual()
        self.calls = 0

    def get_image_features(self, pixel_values, image_grid_thw=None, **kwargs):
        self.calls += 1
        sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        return BaseModelOutputWithPooling(
            last_hidden_state=None,
            pooler_output=tuple(chunk * 2 for chunk in pixel_values.split(sizes)))


class FakeConditionalModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.training = False
        self.model = FakeModel()
        self.outer_calls = 0

    def get_image_features(self, pixel_values, image_grid_thw=None, **kwargs):
        self.outer_calls += 1
        return self.model.get_image_features(
            pixel_values, image_grid_thw, **kwargs)


class VisualFeatureCacheRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.reader = Path(self.temporary.name) / 'reader'
        self.reader.mkdir()
        for name, value in (
                ('config.json', b'{"model":"fixture"}'),
                ('model.safetensors', b'weights-v1'),
                ('tokenizer.json', b'tokenizer-v1'),
                ('preprocessor_config.json', b'processor-v1'),
                ('chat_template.jinja', b'template-v1')):
            (self.reader / name).write_bytes(value)

    def tearDown(self):
        self.temporary.cleanup()

    def test_default_off_is_zero_work_and_invalid_switch_fails(self):
        self.assertIsNone(create_visual_feature_cache(
            'off', Path('/missing-reader'), None, None))
        with self.assertRaisesRegex(ValueError, 'off.*on'):
            create_visual_feature_cache('yes', self.reader, FakeProcessor(), FakeModel())

    def test_identity_binds_model_files_processor_files_and_runtime_config(self):
        processor = FakeProcessor()
        first = visual_feature_identities(
            self.reader, processor, file_hashes={
                'config.json': 'a' * 64,
                'model.safetensors': 'b' * 64,
                'tokenizer.json': 'c' * 64,
                'preprocessor_config.json': 'd' * 64,
            })
        changed_runtime = visual_feature_identities(
            self.reader, FakeProcessor(longest_edge=65536), file_hashes={
                'config.json': 'a' * 64,
                'model.safetensors': 'b' * 64,
                'tokenizer.json': 'c' * 64,
                'preprocessor_config.json': 'd' * 64,
            })
        changed_model = visual_feature_identities(
            self.reader, processor, file_hashes={
                'config.json': 'a' * 64,
                'model.safetensors': 'e' * 64,
                'tokenizer.json': 'c' * 64,
                'preprocessor_config.json': 'd' * 64,
            })
        self.assertNotEqual(first[0], changed_model[0])
        self.assertNotEqual(first[1], changed_runtime[1])
        self.assertEqual(first[0], changed_runtime[0])
        self.assertEqual(first[1], changed_model[1])
        cache = VisualFeatureCache(self.reader, processor, FakeModel())
        self.assertEqual(cache.snapshot()['key_scope'].split(' + ')[0],
                         'ordered image-byte sha256')

    def test_miss_hit_no_image_bypass_and_hit_copy_are_isolated(self):
        model = FakeModel()
        cache = VisualFeatureCache(self.reader, FakeProcessor(), model, capacity_bytes=4096)
        pixels = torch.tensor([[1.], [2.]])
        grid = torch.tensor([[1, 1, 2]])
        with cache.activate(model, ('image-a',)):
            with torch.inference_mode():
                cold = model.get_image_features(
                    pixels, grid, return_dict=True, use_cache=False)
                warm = model.get_image_features(
                    pixels, grid, return_dict=True, use_cache=False)
                warm.pooler_output[0][0, 0] = 99
                owned = model.get_image_features(
                    pixels, grid, return_dict=True, use_cache=False)
        self.assertEqual(model.calls, 1)
        self.assertEqual(cache.store.misses, 1)
        self.assertEqual(cache.store.hits, 2)
        self.assertEqual(owned.pooler_output[0][0, 0].item(), 2.0)
        self.assertEqual(cache.store.entries[list(cache.store.entries)[0]][0][0][0, 0].item(), 2.0)

        with cache.activate(model, ()):
            self.assertNotIn('get_image_features', model.__dict__)
        self.assertEqual(cache.no_image_requests, 1)
        snapshot = cache.snapshot()
        self.assertEqual(snapshot['resident_bytes'], 8)
        self.assertEqual(snapshot['no_image_requests'], 1)
        self.assertIn('whole-batch', snapshot['boundary'])

    def test_conditional_generation_wrapper_resolves_nested_visual_boundary(self):
        wrapper = FakeConditionalModel()
        cache = VisualFeatureCache(
            self.reader, FakeProcessor(), wrapper, capacity_bytes=4096)
        pixels = torch.tensor([[1.], [2.]])
        grid = torch.tensor([[1, 1, 2]])
        with cache.activate(wrapper, ('image-a',)):
            with torch.inference_mode():
                first = wrapper.model.get_image_features(
                    pixels, grid, return_dict=True, use_cache=False)
                second = wrapper.model.get_image_features(
                    pixels, grid, return_dict=True, use_cache=False)
        self.assertTrue(torch.equal(first.pooler_output[0], second.pooler_output[0]))
        self.assertEqual(wrapper.model.calls, 1)
        self.assertEqual(wrapper.outer_calls, 0)
        self.assertNotIn('get_image_features', wrapper.__dict__)


if __name__ == '__main__':
    unittest.main()
