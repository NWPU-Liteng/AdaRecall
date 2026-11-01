from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from copy import deepcopy
from evaluate_omni_registered_cohort import combine_requests, verify_candidate_protocol


class CohortEvaluationTests(unittest.TestCase):
    def test_candidate_protocol_binds_quota_output_and_bounded_frontier(self):
        plan = {'image_edges_per_session': 2, 'max_new_tokens': 512}
        registration = {'arguments': {'image_edges_per_session': 2,
                                     'max_new_tokens': 512, 'max_hyperedges': 64}}
        smoke = {'max_new_tokens': 512, 'rows': [{'hypergraph_evidence_plan': {
            'image_candidate_quota_per_session': 2, 'candidate_hyperedge_limit': 64,
            'candidate_hyperedge_count': 1, 'candidate_replay': [{'edge_id': 'observable'}]}}]}
        verify_candidate_protocol(plan, registration, smoke)
        changed = deepcopy(smoke)
        changed['rows'][0]['hypergraph_evidence_plan']['image_candidate_quota_per_session'] = 0
        with self.assertRaises(ValueError):
            verify_candidate_protocol(plan, registration, changed)
        for name, value in [('candidate_hyperedge_count', 65), ('candidate_hyperedge_limit', 65)]:
            changed = deepcopy(smoke)
            changed['rows'][0]['hypergraph_evidence_plan'][name] = value
            with self.assertRaises(ValueError):
                verify_candidate_protocol(plan, registration, changed)
        for name, value in [('image_edges_per_session', 0), ('max_new_tokens', 256)]:
            changed = deepcopy(registration)
            changed['arguments'][name] = value
            with self.assertRaises(ValueError):
                verify_candidate_protocol(plan, changed, smoke)
        changed = deepcopy(registration)
        changed['arguments']['image_novelty_weight'] = 2.0
        with self.assertRaises(ValueError):
            verify_candidate_protocol(plan, changed, smoke)
        changed = deepcopy(smoke)
        changed['rows'][0]['hypergraph_evidence_plan']['image_novelty_weight'] = 2.0
        with self.assertRaises(ValueError):
            verify_candidate_protocol(plan, registration, changed)
        for weight in (-1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                verify_candidate_protocol({**plan, 'image_novelty_weight': weight}, registration, smoke)

    def test_shared_judge_batch_preserves_per_user_requests_and_rejects_duplicate(self):
        parts = [{'judge_requests': [{'question_id': '0_q', 'prompt': 'unchanged'}]},
                 {'judge_requests': [{'question_id': '1_q', 'prompt': 'original'}]}]
        self.assertEqual([row['question_id'] for row in combine_requests(parts)], ['0_q', '1_q'])
        self.assertEqual(combine_requests(parts)[0]['prompt'], 'unchanged')
        with self.assertRaises(ValueError):
            combine_requests(parts + parts)
        with self.assertRaises(ValueError):
            combine_requests([])

    def test_visual_budget_and_reference_mode_cannot_be_silently_changed(self):
        plan = {'max_new_tokens': 512, 'visual_cache_bytes': 64}
        registration = {'arguments': {'max_new_tokens': 512, 'visual_cache_bytes': 64,
                                     'verify_generation': True}}
        smoke = {'max_new_tokens': 512, 'visual_state_reuse': {'capacity_bytes': 64},
                 'verify_generation': True, 'rows': []}
        verify_candidate_protocol(plan, registration, smoke)
        for changed in ({**smoke, 'verify_generation': False},
                        {**smoke, 'visual_state_reuse': {'capacity_bytes': 128}},
                        {**smoke, 'visual_state_reuse': None}):
            with self.assertRaises(ValueError):
                verify_candidate_protocol(plan, registration, changed)
        with self.assertRaises(ValueError):
            verify_candidate_protocol({**plan, 'visual_cache_bytes': 0}, registration, smoke)


if __name__ == '__main__':
    unittest.main()
