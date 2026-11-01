from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from omni_evidence_planner import bound_hyperedges, build_local_hyperedges, render_atoms, select_hyperedges


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(text)


class OmniEvidencePlannerTests(unittest.TestCase):
    def atom(self, identity, ordinal, image=False):
        return {'atom_id': identity, 'session_id': 's', 'ordinal': ordinal,
                'role': 'user', 'content_type': 'image' if image else 'text',
                'content': '' if image else identity * 3,
                'image_refs': [{'normalized': identity + '.png', 'status': 'exact'}] if image else []}

    def test_complete_context_and_raw_image_are_indivisible(self):
        atoms = [self.atom('before', 0), self.atom('image', 1, True), self.atom('after', 2)]
        edges = build_local_hyperedges(['s'], {'s': atoms}, radius=1)
        image_edge = next(edge for edge in edges if len(edge['atom_ids']) == 3)
        result = select_hyperedges([image_edge], [0.8], {atom['atom_id']: atom for atom in atoms},
                                  CharacterTokenizer(), text_budget=1000, max_images=1)
        self.assertEqual(result['accepted_atom_ids'], ['before', 'image', 'after'])
        self.assertEqual(result['images'][0]['source_atom_id'], 'image')
        self.assertEqual(result['text_tokens'], len(result['text']))
        with self.assertRaises(ValueError):
            select_hyperedges([image_edge], [0.8], {atom['atom_id']: atom for atom in atoms},
                              CharacterTokenizer(), text_budget=1, max_images=1)

    def test_overlap_charged_once_and_order_is_observable_chronology(self):
        atoms = [self.atom('a', 0), self.atom('b', 1), self.atom('c', 2)]
        edges = [{'edge_id': 'later', 'atom_ids': ['b', 'c']},
                 {'edge_id': 'earlier', 'atom_ids': ['a', 'b']}]
        budget = len(render_atoms(atoms))
        result = select_hyperedges(edges, [0.9, 0.8], {atom['atom_id']: atom for atom in atoms},
                                  CharacterTokenizer(), text_budget=budget, max_images=1, require_image=False)
        self.assertEqual(result['accepted_atom_ids'], ['a', 'b', 'c'])
        self.assertEqual(result['text_tokens'], budget)
        self.assertAlmostEqual(result['objective'], 0.8**2 + 2 * 0.9**2)

    def test_image_capacity_and_unresolved_refs_reject_whole_edge(self):
        atoms = [self.atom('text', 0), self.atom('one', 1, True), self.atom('two', 2, True)]
        edges = [{'edge_id': 'too_many', 'atom_ids': ['text', 'one', 'two']}]
        for max_images in [1, 2]:
            if max_images == 2:
                atoms[1]['image_refs'][0]['status'] = 'missing'
            with self.assertRaises(ValueError):
                select_hyperedges(edges, [1.0], {atom['atom_id']: atom for atom in atoms},
                                  CharacterTokenizer(), text_budget=1000, max_images=max_images)

    def test_bad_membership_and_scores_fail_closed(self):
        atom = self.atom('a', 0)
        for ids, score in [(['missing'], 0.5), (['a', 'a'], 0.5), (['a'], float('nan'))]:
            with self.assertRaises(ValueError):
                select_hyperedges([{'edge_id': 'bad', 'atom_ids': ids}], [score], {'a': atom},
                                  CharacterTokenizer(), text_budget=100, max_images=1, require_image=False)
        with self.assertRaises(ValueError):
            build_local_hyperedges(['s', 's'], {'s': [atom]})

    def test_bounded_candidates_are_stable_and_keep_observable_image_seed(self):
        atoms = {identity: self.atom(identity, rank, identity == 'image')
                 for rank, identity in enumerate(('a', 'b', 'c', 'image'))}
        edges = [{'edge_id': identity, 'atom_ids': [identity]} for identity in atoms]
        kept, scores, pruned = bound_hyperedges(edges, [0.8, 0.8, 0.7, 0.1], atoms, 2)
        self.assertEqual([edge['edge_id'] for edge in kept], ['a', 'image'])
        self.assertEqual(scores, [0.8, 0.1])
        self.assertEqual(pruned, ['b', 'c'])
        for limit, values in [(0, [0.8] * 4), (2, [float('nan')] * 4), (2, [0.8])]:
            with self.assertRaises(ValueError):
                bound_hyperedges(edges, values, atoms, limit)

    def test_zero_gain_and_image_overflow_do_not_trigger_tokenization(self):
        class CountingTokenizer(CharacterTokenizer):
            def __init__(self):
                self.calls = 0

            def encode(self, text, add_special_tokens=False):
                self.calls += 1
                return super().encode(text, add_special_tokens)

        atoms = {identity: self.atom(identity, rank, identity in {'one', 'two'})
                 for rank, identity in enumerate(('text', 'one', 'two'))}
        edges = [{'edge_id': 'good', 'atom_ids': ['text', 'one']},
                 {'edge_id': 'zero', 'atom_ids': ['text']},
                 {'edge_id': 'overflow', 'atom_ids': ['one', 'two']}]
        tokenizer = CountingTokenizer()
        result = select_hyperedges(edges, [0.9, 0.0, 0.8], atoms, tokenizer,
                                  text_budget=1000, max_images=1)
        self.assertEqual([edge['edge_id'] for edge in result['selected_hyperedges']], ['good'])
        self.assertEqual(tokenizer.calls, 1)

    def test_optional_session_image_frontier_is_diverse_bounded_and_label_free(self):
        atoms = {identity: self.atom(identity, rank, identity in {'one', 'two', 'other'})
                 for rank, identity in enumerate(('text', 'one', 'two', 'other'))}
        atoms['other']['session_id'] = 'second'
        edges = [{'edge_id': 'text', 'atom_ids': ['text']},
                 {'edge_id': 'one', 'atom_ids': ['one']},
                 {'edge_id': 'one_duplicate', 'atom_ids': ['one']},
                 {'edge_id': 'two', 'atom_ids': ['two']},
                 {'edge_id': 'other', 'atom_ids': ['other']}]
        kept, _, pruned = bound_hyperedges(edges, [0.9, 0.8, 0.7, 0.6, 0.1], atoms, 3,
                                         image_edges_per_session=2)
        self.assertEqual([edge['edge_id'] for edge in kept], ['one', 'two', 'other'])
        self.assertEqual(pruned, ['text', 'one_duplicate'])
        legacy, _, _ = bound_hyperedges(edges, [0.9, 0.8, 0.7, 0.6, 0.1], atoms, 3)
        self.assertEqual([edge['edge_id'] for edge in legacy], ['text', 'one', 'one_duplicate'])
        with self.assertRaises(ValueError):
            bound_hyperedges(edges, [0.9] * len(edges), atoms, 3, image_edges_per_session=-1)

    def test_image_novelty_weight_is_label_free_and_bounded(self):
        atoms = {
            'a': {'atom_id': 'a', 'session_id': 's', 'ordinal': 0, 'role': 'user',
                  'content_type': 'text', 'content': 'alpha',
                  'image_refs': [{'status': 'exact', 'normalized': 'img-a'}]},
            'b': {'atom_id': 'b', 'session_id': 's', 'ordinal': 1, 'role': 'user',
                  'content_type': 'text', 'content': 'beta',
                  'image_refs': [{'status': 'exact', 'normalized': 'img-b'}]},
        }
        edges = [{'edge_id': 'e-a', 'atom_ids': ['a']}, {'edge_id': 'e-b', 'atom_ids': ['b']}]
        result = select_hyperedges(edges, [0.5, 0.49], atoms, CharacterTokenizer(),
                                   text_budget=100, max_images=2, require_image=True,
                                   image_novelty_weight=2.0)
        self.assertEqual({ref['normalized'] for ref in result['images']}, {'img-a', 'img-b'})
        self.assertEqual(result['image_novelty_weight'], 2.0)
        self.assertAlmostEqual(result['objective'], sum(edge['marginal_gain']
                                                       for edge in result['selected_hyperedges']))
        self.assertAlmostEqual(result['objective'], 3 * (0.5**2 + 0.49**2))
        for weight in (-1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                select_hyperedges(edges, [0.5, 0.49], atoms, CharacterTokenizer(),
                                  text_budget=100, max_images=2, image_novelty_weight=weight)

    def test_image_identity_has_one_max_reward_across_overlapping_edges(self):
        atoms = {identity: self.atom(identity, rank) for rank, identity in enumerate(('a', 'b'))}
        for atom in atoms.values():
            atom['image_refs'] = [{'normalized': 'same.png', 'status': 'exact'}]
        edges = [{'edge_id': identity, 'atom_ids': [identity]} for identity in atoms]
        result = select_hyperedges(edges, [0.8, 0.9], atoms, CharacterTokenizer(),
                                   text_budget=1000, max_images=1, image_novelty_weight=2)
        self.assertEqual(len(result['images']), 1)
        self.assertAlmostEqual(result['image_coverage_objective'], 0.9**2)
        self.assertAlmostEqual(result['objective'], 0.8**2 + 3 * 0.9**2)
        self.assertAlmostEqual(result['objective'], sum(edge['marginal_gain']
                                                       for edge in result['selected_hyperedges']))

    def test_unique_image_weight_changes_choice_without_changing_capacity(self):
        atoms = {identity: self.atom(identity, rank) for rank, identity in enumerate(('a', 'b', 'c'))}
        for identity in ('a', 'b'):
            atoms[identity]['image_refs'] = [{'normalized': identity + '.png', 'status': 'exact'}]
        edges = [{'edge_id': 'text_rich', 'atom_ids': ['a', 'c']},
                 {'edge_id': 'image_relevant', 'atom_ids': ['b']}]
        results = [select_hyperedges(edges, [0.6, 0.7], atoms, CharacterTokenizer(),
                                     text_budget=1000, max_images=1, image_novelty_weight=weight)
                   for weight in (0, 2)]
        self.assertEqual([result['selected_hyperedges'][0]['edge_id'] for result in results],
                         ['text_rich', 'image_relevant'])
        self.assertTrue(all(len(result['images']) == 1 for result in results))


if __name__ == '__main__':
    unittest.main()
