from pathlib import Path
import sys
import unittest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from omni_embedding_store import EdgeEmbeddingStore


class EdgeEmbeddingStoreTests(unittest.TestCase):
    def test_exact_content_identity_and_owning_storage(self):
        store = EdgeEmbeddingStore(32)
        edge = {'edge_id': 'a', 'embedding_text': 'original'}
        source = torch.ones((8, 4))
        self.assertIsNone(store.get(edge))
        self.assertTrue(store.put(edge, source[0]))
        source.zero_()
        self.assertTrue(torch.equal(store.get(edge), torch.ones(4)))
        self.assertIsNone(store.get({**edge, 'embedding_text': 'changed'}))
        self.assertEqual(store.summary()['resident_bytes'], 16)

    def test_capacity_lru_eviction_and_oversize_fallback(self):
        store = EdgeEmbeddingStore(32)
        edges = [{'edge_id': identity, 'embedding_text': identity} for identity in ('a', 'b', 'c')]
        for edge in edges[:2]:
            store.put(edge, torch.ones(4))
        store.get(edges[0])
        store.put(edges[2], torch.ones(4))
        self.assertIsNone(store.get(edges[1]))
        self.assertIsNotNone(store.get(edges[0]))
        self.assertEqual(store.evictions, 1)
        self.assertFalse(store.put(edges[1], torch.ones(9)))
        self.assertLessEqual(store.peak_resident_bytes, store.capacity_bytes)
        with self.assertRaises(ValueError):
            EdgeEmbeddingStore(0)


if __name__ == '__main__':
    unittest.main()
