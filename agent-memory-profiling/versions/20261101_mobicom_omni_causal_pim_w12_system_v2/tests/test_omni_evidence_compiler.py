import unittest
from itertools import combinations
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from omni_evidence_compiler import compile_evidence


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()


class EvidenceCompilerTest(unittest.TestCase):
    def test_exact_budgeted_semantic_coverage(self):
        atoms = {
            'a': {'atom_id': 'a', 'ordinal': 0, 'session_id': 's', 'role': 'user',
                  'content': 'alpha', 'content_type': 'text', 'image_refs': []},
            'b': {'atom_id': 'b', 'ordinal': 1, 'session_id': 's', 'role': 'assistant',
                  'content': 'beta', 'content_type': 'text', 'image_refs': []},
        }
        edges = [{'edge_id': 'e0', 'atom_ids': ['a'], 'embedding_text': 'alpha'},
                 {'edge_id': 'e1', 'atom_ids': ['b'], 'embedding_text': 'beta'}]
        result = compile_evidence(
            edges, np.asarray([[1., 0.], [0., 1.]]), np.asarray([1., 0.]), atoms,
            Tokenizer(), {}, text_budget=3, max_images=0, input_budget=8,
            fixed_prompt_tokens=1, time_limit=1.)
        self.assertEqual(result['accepted_atom_ids'], ['a'])
        self.assertEqual(result['policy'], 'evidence_compiler_v1')

    def test_matches_exhaustive_coverage_without_atom_count_prizes(self):
        atoms = {str(index): {'atom_id': str(index), 'ordinal': index, 'session_id': 's',
                             'role': 'user', 'content': 'fact', 'content_type': 'text', 'image_refs': []}
                 for index in range(3)}
        edges = [{'edge_id': 'e' + identity, 'atom_ids': [identity]} for identity in atoms]
        vectors = np.asarray([[1., 0.], [1., 0.], [.6, .8]])
        query = np.asarray([1., 0.])
        result = compile_evidence(edges, vectors, query, atoms, Tokenizer(), {}, text_budget=4,
                                  max_images=0, input_budget=20, fixed_prompt_tokens=1)
        weights = (vectors @ query) ** 2
        affinity = np.clip(vectors @ vectors.T, 0., 1.)
        exhaustive = max(float(weights @ affinity[:, list(chosen)].max(axis=1))
                         for count in (1, 2) for chosen in combinations(range(3), count))
        self.assertAlmostEqual(result['objective'], exhaustive)
        self.assertIn('2', result['accepted_atom_ids'])

    def test_actual_visual_tokens_constrain_input_not_fixed_image_charge(self):
        atoms = {'a': {'atom_id': 'a', 'ordinal': 0, 'session_id': 's', 'role': 'user',
                       'content': 'fact', 'content_type': 'text', 'image_refs': [
                           {'status': 'exact', 'normalized': 's/image.png'}]}}
        edges = [{'edge_id': 'image', 'atom_ids': ['a']}]
        result = compile_evidence(edges, [[1., 0.]], [1., 0.], atoms, Tokenizer(),
                                  {'s/image.png': 1024}, text_budget=10, max_images=1,
                                  input_budget=512, fixed_prompt_tokens=1)
        self.assertEqual(result['accepted_atom_ids'], [])
        self.assertEqual(result['images'], [])

    def test_shared_atom_union_is_charged_once(self):
        atoms = {'a': {'atom_id': 'a', 'ordinal': 0, 'session_id': 's', 'role': 'user',
                       'content': 'fact', 'content_type': 'text', 'image_refs': []}}
        edges = [{'edge_id': 'first', 'atom_ids': ['a']}, {'edge_id': 'second', 'atom_ids': ['a']}]
        result = compile_evidence(edges, [[1., 0.], [0., 1.]], [2 ** -.5, 2 ** -.5],
                                  atoms, Tokenizer(), {}, text_budget=2, max_images=0,
                                  input_budget=10, fixed_prompt_tokens=1)
        self.assertEqual(result['accepted_atom_ids'], ['a'])
        self.assertEqual(len(result['selected_hyperedges']), 2)

    def test_rejects_mixed_or_nonfinite_embedding_space(self):
        atoms = {'a': {'atom_id': 'a', 'ordinal': 0, 'session_id': 's', 'role': 'user',
                       'content': 'fact', 'content_type': 'text', 'image_refs': []}}
        for vectors in ([[2., 0.]], [[float('nan'), 0.]]):
            with self.subTest(vectors=vectors), self.assertRaises(ValueError):
                compile_evidence([{'edge_id': 'e', 'atom_ids': ['a']}], vectors, [1., 0.],
                                 atoms, Tokenizer(), {}, text_budget=10, max_images=0,
                                 input_budget=20, fixed_prompt_tokens=1)


if __name__ == '__main__':
    unittest.main()
