import itertools
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from omni_provenance_optimizer import build_provenance_hyperedges, optimize_provenance
from omni_evidence_planner import render_atoms


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(text)


def atom(identity, ordinal, role='user', image=None):
    return {'atom_id': identity, 'session_id': 's', 'event_id': 'e',
            'ordinal': ordinal, 'role': role, 'content_type': 'text', 'content': identity,
            'image_refs': [] if image is None else [{'normalized': image, 'status': 'exact'}]}


class ProvenanceOptimizerTests(unittest.TestCase):
    def test_exchange_closure_uses_structure_not_text(self):
        atoms = [atom('a', 0), atom('b', 1), atom('c', 2, 'assistant'), atom('d', 3)]
        edges = build_provenance_hyperedges(['s'], {'s': atoms})
        self.assertEqual([edge['atom_ids'] for edge in edges], [['a', 'b', 'c'], ['d']])

    def test_optimum_matches_exhaustive_union_budget(self):
        atoms = {identity: atom(identity, index) for index, identity in enumerate(('a', 'b', 'c'))}
        edges = [{'edge_id': 'x', 'atom_ids': ['a', 'b']},
                 {'edge_id': 'y', 'atom_ids': ['b', 'c']},
                 {'edge_id': 'z', 'atom_ids': ['a']}]
        values = [2., 3., 1.]
        result = optimize_provenance(edges, values, atoms, Tokenizer(), {}, text_budget=40,
                                     max_images=0, input_budget=50, fixed_prompt_tokens=0)
        expected = max(sum(values[index] for index in subset) for count in range(4)
                       for subset in itertools.combinations(range(3), count)
                       if sum(len(render_atoms([atoms[identity]]) + '\n') for identity in
                              {identity for index in subset for identity in edges[index]['atom_ids']}) <= 40)
        self.assertAlmostEqual(result['objective'], expected)
        self.assertEqual(result['solver_status'], 0)

    def test_shared_images_charged_once(self):
        atoms = {'a': atom('a', 0, image='pic'), 'b': atom('b', 1, image='pic')}
        edges = [{'edge_id': identity, 'atom_ids': [identity]} for identity in atoms]
        result = optimize_provenance(edges, [2., 3.], atoms, Tokenizer(), {'pic': 17},
                                     text_budget=200, max_images=1, input_budget=200, fixed_prompt_tokens=0)
        self.assertEqual(len(result['selected_hyperedges']), 2)
        self.assertEqual(result['visual_tokens_from_processor_grid'], 17)

    def test_no_image_is_forced_and_negative_evidence_rejected(self):
        atoms = {'a': atom('a', 0)}
        result = optimize_provenance([{'edge_id': 'a', 'atom_ids': ['a']}], [-3.], atoms,
                                     Tokenizer(), {}, text_budget=100, max_images=0,
                                     input_budget=100, fixed_prompt_tokens=0)
        self.assertEqual(result['accepted_atom_ids'], [])
        self.assertEqual(result['images'], [])

    def test_missing_grid_and_unknown_atoms_fail_closed(self):
        atoms = {'a': atom('a', 0, image='pic')}
        with self.assertRaises(ValueError):
            optimize_provenance([{'edge_id': 'a', 'atom_ids': ['a']}], [1.], atoms,
                                 Tokenizer(), {}, text_budget=100, max_images=1,
                                 input_budget=100, fixed_prompt_tokens=0)


if __name__ == '__main__':
    unittest.main()
