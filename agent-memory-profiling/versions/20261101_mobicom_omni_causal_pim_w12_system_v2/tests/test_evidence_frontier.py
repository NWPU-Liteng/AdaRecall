from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pim_backend.evidence_frontier import (build_frontier_levels, candidates_for_frontier_level,
                                           compile_candidate_frontier)
from pim_backend.evidence_state import plan_evidence_route


def candidate(demand_id, base_record_id, score, text='candidate'):
    return {'demand_id': demand_id, 'base_record_id': base_record_id,
            'record_id': base_record_id + '#' + demand_id,
            'object_ids': ['text:' + base_record_id], 'score': score,
            'retrieval_cosine': score, 'retrieval_text': text}


class EvidenceFrontierTests(unittest.TestCase):
    def test_frontier_deduplicates_only_same_demand_and_source(self):
        frontier = compile_candidate_frontier([
            candidate('d0', 'r0', .4),
            candidate('d0', 'r0', .8),
            candidate('d1', 'r0', .3),
        ], question_id='q0')
        self.assertEqual(frontier['schema'], 'evidence-candidate-frontier-v1')
        self.assertEqual(frontier['state_count'], 2)
        self.assertEqual(frontier['deduplicated_candidate_count'], 1)
        self.assertEqual(frontier['candidate_count_by_demand'], {'d0': 1, 'd1': 1})
        self.assertEqual(frontier['records'][0]['score'], .8)
        self.assertFalse(frontier['semantic_pruning_applied'])

    def test_frontier_route_is_explicitly_host_without_profile(self):
        frontier = compile_candidate_frontier([candidate('d0', 'r0', .8)])
        route = plan_evidence_route(frontier)
        self.assertEqual(route['chosen_backend'], 'host')
        self.assertIn('missing_backend_profile', route['reason_codes'])

    def test_invalid_candidate_identity_is_rejected(self):
        invalid = candidate('d0', 'r0', .8)
        invalid['object_ids'] = ['text:r0', 'text:r0']
        with self.assertRaises(ValueError):
            compile_candidate_frontier([invalid])

    def test_frontier_keeps_distinct_same_source_witness_contracts(self):
        first = candidate('d0', 'r0', .8)
        second = candidate('d0', 'r0', .7)
        second['object_ids'] = ['text:r0:alternate']
        second['bindings'] = {'person': 'Alice'}
        frontier = compile_candidate_frontier([first, second])
        self.assertEqual(frontier['state_count'], 2)
        self.assertEqual(frontier['deduplicated_candidate_count'], 0)
        self.assertEqual(frontier['source_provenance_count'], 1)

    def test_frontier_levels_expand_to_complete_candidate_set(self):
        candidates = [candidate('d0', f'r{index}', 1.0 - index / 10)
                      for index in range(4)]
        candidates += [candidate('d1', f's{index}', 1.0 - index / 10)
                       for index in range(3)]
        frontier = compile_candidate_frontier(candidates)
        levels = build_frontier_levels(frontier, seed_per_demand=1)
        self.assertEqual([level['candidate_count'] for level in levels], [2, 4, 7])
        self.assertTrue(all(level['covered_demand_count'] == 2 for level in levels))
        self.assertTrue(levels[-1]['is_complete'])
        selected = candidates_for_frontier_level(candidates, levels[0])
        self.assertEqual({item['demand_id'] for item in selected}, {'d0', 'd1'})

if __name__ == '__main__':
    unittest.main()
