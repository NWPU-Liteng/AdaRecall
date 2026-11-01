from pathlib import Path
import sys
import unittest

import torch
from transformers.modeling_outputs import BaseModelOutputWithPooling

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from omni_visual_feature_store import VisualFeatureStore


class VisualFeatureStoreTests(unittest.TestCase):
    def store(self, capacity=128, model='model-a', processor='processor-a'):
        return VisualFeatureStore(capacity, model, processor, output_dtype=torch.float32, spatial_merge_size=1)

    def test_bfloat16_digest_and_identity_mismatches(self):
        pixels = torch.ones((2, 3), dtype=torch.bfloat16)
        grid = torch.tensor([[1, 1, 2]])
        first = self.store().key(['image-a'], pixels, grid)
        self.assertEqual(first, self.store().key(['image-a'], pixels.clone(), grid.clone()))
        alternatives = [self.store().key(['image-b'], pixels, grid),
                        self.store().key(['image-a'], pixels + 1, grid),
                        self.store().key(['image-a'], pixels, torch.tensor([[1, 2, 1]])),
                        self.store(model='model-b').key(['image-a'], pixels, grid),
                        self.store(processor='processor-b').key(['image-a'], pixels, grid)]
        self.assertTrue(all(first != other for other in alternatives))

    def test_lru_oversize_and_owned_hit(self):
        store = self.store(16)
        store.put(('a',), [torch.tensor([1., 2.])])
        store.put(('b',), [torch.tensor([3., 4.])])
        hit = store.get(('a',), 'cpu')
        hit[0][0] = 9
        self.assertEqual(store.entries[('a',)][0][0][0].item(), 1)
        store.put(('c',), [torch.tensor([5., 6.])])
        self.assertNotIn(('b',), store.entries)
        self.assertEqual(store.evictions, 1)
        self.assertFalse(store.put(('large',), [torch.ones(5)]))
        self.assertEqual(store.resident_bytes, 16)

    def test_integrity_corruption_recomputes(self):
        store = self.store()
        store.put(('a',), [torch.tensor([1., 2.])])
        store.entries[('a',)][0][0][0] = 9
        self.assertIsNone(store.get(('a',), 'cpu'))
        self.assertEqual(store.integrity_failures, 1)
        self.assertEqual(store.resident_bytes, 0)

    def test_ordered_whole_batch_cold_hit_and_partial_overlap_miss(self):
        store = self.store()
        pixels = torch.tensor([[1.], [2.], [3.]])
        grids = torch.tensor([[1, 1, 1], [1, 1, 2]])
        calls = []

        def encode(batch, grid, **kwargs):
            calls.append(batch.clone())
            return BaseModelOutputWithPooling(last_hidden_state=None,
                pooler_output=tuple(chunk * 2 for chunk in batch.split(grid.prod(-1).tolist())))

        with torch.inference_mode():
            cold = store.resolve_batch(pixels, grids, ['a', 'b'], encode)
            warm = store.resolve_batch(pixels, grids, ['a', 'b'], encode)
            store.resolve_batch(pixels, grids, ['a', 'c'], encode)
        self.assertEqual(len(calls), 2)
        self.assertTrue(torch.equal(calls[-1], pixels))
        self.assertEqual(store.hits, 1)
        self.assertEqual(store.image_hits, 2)
        self.assertTrue(all(torch.equal(first, second) for first, second in zip(
            cold.pooler_output, warm.pooler_output)))

    def test_oversize_is_cold_fallback(self):
        store = self.store(1)
        calls = []

        def encode(batch, grid):
            calls.append(True)
            return BaseModelOutputWithPooling(pooler_output=(batch.clone(),))

        with torch.inference_mode():
            for _ in range(2):
                store.resolve_batch(torch.ones((1, 1)), torch.tensor([[1, 1, 1]]), ['a'], encode)
        self.assertEqual(len(calls), 2)
        self.assertEqual(store.rejected, 2)
        self.assertEqual(store.hits, 0)

    def test_unsupported_calls_and_bad_shapes_fail_closed(self):
        store = self.store()
        with self.assertRaises(ValueError):
            store.resolve_batch(torch.ones((1, 1)), torch.tensor([[1, 1, 1]]), ['a'], lambda *_: None)
        with self.assertRaises(ValueError):
            store.key(['a'], torch.ones((2, 1)), torch.tensor([[1, 1, 1]]))
        with self.assertRaises(ValueError):
            store.key(['a'], torch.ones((1, 1)), torch.tensor([[1., 1., 1.]]))
        with torch.inference_mode():
            marker = object()
            result = store.resolve_batch(torch.ones((1, 1)), torch.tensor([[1, 1, 1]]), ['a'],
                lambda *args, **kwargs: marker, output_attentions=True)
            self.assertIs(result, marker)
            self.assertEqual(store.bypasses, 1)

    def test_boolean_use_cache_is_preserved_and_keyed(self):
        store = self.store()
        calls = []

        def encode(pixels, grids, **kwargs):
            calls.append(kwargs)
            return BaseModelOutputWithPooling(pooler_output=(pixels.clone(),))

        with torch.inference_mode():
            for option in (True, True, False):
                store.resolve_batch(torch.ones((1, 1)), torch.tensor([[1, 1, 1]]), ['a'],
                                    encode, use_cache=option, return_dict=True)
        self.assertEqual(calls, [{'use_cache': True, 'return_dict': True},
                                 {'use_cache': False, 'return_dict': True}])
        self.assertEqual(store.hits, 1)
        self.assertEqual(store.bypasses, 0)

    def test_reference_bypass_and_restore_on_exception(self):
        class Model(torch.nn.Module):
            def get_image_features(self, pixels, grid, **kwargs):
                return BaseModelOutputWithPooling(pooler_output=(pixels.clone(),))

        model = Model().eval()
        store = self.store()
        pixels, grid = torch.ones((1, 1)), torch.tensor([[1, 1, 1]])
        with torch.inference_mode():
            with store.activate(model, ['a']):
                model.get_image_features(pixels, grid)
                model.get_image_features(pixels, grid)
            model.get_image_features(pixels, grid)
        self.assertEqual(store.hits, 1)
        self.assertEqual(store.misses, 1)
        self.assertNotIn('get_image_features', model.__dict__)
        with self.assertRaises(RuntimeError):
            with store.activate(model, ['a']):
                raise RuntimeError('restore')
        self.assertNotIn('get_image_features', model.__dict__)
        model.train()
        with self.assertRaises(ValueError):
            with store.activate(model, ['a']):
                pass


if __name__ == '__main__':
    unittest.main()
