from copy import deepcopy
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from omni_embedding_producer import producer_identity, resolve_producers, stable_producer_batches
from omni_embedding_store import EdgeEmbeddingStore


def inputs():
    atoms = {str(index): {'session_id': str(index // 3), 'ordinal': index} for index in range(6)}
    edges = [{'edge_id': str(index), 'atom_ids': [str(index)], 'embedding_text': str(index)}
             for index in range(6)]
    return edges, atoms


class EmbeddingProducerTests(unittest.TestCase):
    def test_canonical_batches_do_not_depend_on_retrieval_rank_or_other_sessions(self):
        edges, atoms = inputs()
        batches = stable_producer_batches(edges, atoms, 2)
        reversed_batches = stable_producer_batches(list(reversed(edges)), atoms, 2)
        self.assertEqual(batches, reversed_batches)
        self.assertEqual(batches[:2], stable_producer_batches(edges[:3], atoms, 2))

    def test_cache_hit_preserves_exact_batch_sensitive_reference(self):
        edges, atoms = inputs()
        calls = []

        def encode(texts):
            calls.append(texts)
            return [torch.tensor([float(text), float(len(texts)), float(position)])
                    for position, text in enumerate(texts)]

        reference, _ = resolve_producers(edges, atoms, None, encode, 'model/tokenizer/dtype', 2)
        store = EdgeEmbeddingStore(1024)
        first, records = resolve_producers(list(reversed(edges)), atoms, store, encode, 'model/tokenizer/dtype', 2)
        hit, hit_records = resolve_producers(edges[:3], atoms, store, encode, 'model/tokenizer/dtype', 2)
        self.assertEqual(len(calls), 8)
        self.assertTrue(all(not record['reused'] for record in records))
        self.assertTrue(all(record['reused'] for record in hit_records))
        self.assertTrue(all(torch.equal(reference[key], value) for key, value in first.items()))
        self.assertTrue(all(torch.equal(reference[key], value) for key, value in hit.items()))

    def test_partial_hit_recomputes_whole_producer_and_model_change_is_cold(self):
        edges, atoms = inputs()
        store = EdgeEmbeddingStore(1024)
        encode = lambda texts: [torch.tensor([float(text)]) for text in texts]
        resolve_producers(edges, atoms, store, encode, 'model', 2)
        store.entries.pop(next(iter(store.entries)))
        _, records = resolve_producers(edges, atoms, store, encode, 'model', 2)
        self.assertEqual(records[0]['computed_vectors'], 2)
        self.assertEqual(records[0]['partial_hits_recomputed'], 1)
        _, changed = resolve_producers(edges, atoms, store, encode, 'new-model', 2)
        self.assertTrue(all(not record['reused'] for record in changed))

    def test_operand_or_producer_identity_mutation_invalidates_entry(self):
        edges, atoms = inputs()
        original = producer_identity(edges[:2], 'model')
        changed = deepcopy(edges[:2])
        changed[0]['embedding_text'] = 'changed'
        self.assertNotEqual(original, producer_identity(changed, 'model'))
        self.assertNotEqual(original, producer_identity(list(reversed(edges[:2])), 'model'))
        self.assertNotEqual(original, producer_identity(edges[:2], 'new-model'))
        with self.assertRaises(ValueError):
            stable_producer_batches(edges + edges, atoms)


if __name__ == '__main__':
    unittest.main()
