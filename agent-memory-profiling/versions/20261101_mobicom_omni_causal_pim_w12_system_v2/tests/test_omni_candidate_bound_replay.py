from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from omni_evidence_planner import bound_hyperedges, select_hyperedges
from replay_omni_candidate_bound import candidate_summary, replay_bounds


class Tokenizer:
    def encode(self, text, **kwargs):
        return text.split()


class CandidateBoundReplayTests(unittest.TestCase):
    def fixture(self):
        atoms = {}
        edges = []
        for index, (session, image) in enumerate((('a', False), ('a', True), ('b', True))):
            identity = f'atom-{index}'
            atoms[identity] = {'atom_id': identity, 'ordinal': index, 'role': 'user',
                'content_type': 'text', 'content': 'observed text', 'session_id': session,
                'image_refs': [{'normalized': f'image-{index}', 'status': 'exact'}] if image else []}
            edges.append({'edge_id': f'edge-{index}', 'atom_ids': [identity]})
        scores = [.9, .8, .1]
        bounded, bounded_scores, pruned = bound_hyperedges(edges, scores, atoms, 2)
        expected = select_hyperedges(bounded, bounded_scores, atoms, Tokenizer(),
                                    text_budget=100, max_images=2)
        expected.update(candidate_hyperedge_limit=2, image_candidate_quota_per_session=0,
                        candidate_pruned_edge_ids=pruned,
                        candidate_replay=[dict(edge, similarity=score) for edge, score in zip(bounded, bounded_scores)])
        return atoms, edges, scores, expected

    def test_existing_quota_intervenes_before_selection(self):
        atoms, edges, scores, expected = self.fixture()
        result = replay_bounds(edges, scores, atoms, expected, Tokenizer())
        self.assertTrue(result['counterfactual_executed'])
        self.assertEqual(result['control_mismatches'], [])
        self.assertEqual(result['control_candidates']['image_ids'], ['image-1'])
        self.assertEqual(result['quota_candidates']['image_ids'], ['image-1', 'image-2'])
        self.assertEqual(result['quota_plan']['max_images'], 2)
        self.assertNotIn('image_edges_per_session', result['quota_plan'])

    def test_unmatched_control_blocks_intervention(self):
        atoms, edges, scores, expected = self.fixture()
        expected['candidate_replay'][0]['similarity'] = .99
        result = replay_bounds(edges, scores, atoms, expected, Tokenizer())
        self.assertFalse(result['counterfactual_executed'])
        self.assertIn('control_similarity_values', result['control_mismatches'])

    def test_observable_summary_has_no_evaluation_input(self):
        atoms, edges, _, _ = self.fixture()
        self.assertEqual(candidate_summary(edges, atoms)['session_ids'], ['a', 'b'])


if __name__ == '__main__':
    unittest.main()
