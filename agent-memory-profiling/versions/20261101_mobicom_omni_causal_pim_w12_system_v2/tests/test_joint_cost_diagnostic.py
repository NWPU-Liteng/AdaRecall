import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from audit_joint_cost_diagnostic import compare_rows


class JointCostDiagnosticTests(unittest.TestCase):
    def row(self):
        return {'question_id': 'q', 'person_id': '0', 'hypergraph_evidence_plan': {
            'candidate_replay': [{'edge_id': 'edge', 'relevance_log_odds': 1.}],
            'accepted_atom_ids': ['a'], 'images': []},
            'processor_inputs_sha256': {'input_ids': 'hash'},
            'answer_token_ids': [12], 'stop_reason': 'max_new_tokens'}

    def test_exact_result_is_not_performance_qualification(self):
        result = compare_rows(self.row(), self.row())
        self.assertTrue(result['scores_bitwise_equal'])
        self.assertTrue(result['generated_tokens_equal'])
        self.assertFalse(result['performance_qualified'])

    def test_numerical_drift_and_selection_change_are_visible(self):
        before = self.row()
        after = copy.deepcopy(before)
        after['hypergraph_evidence_plan']['candidate_replay'][0]['relevance_log_odds'] = 1.25
        after['hypergraph_evidence_plan']['accepted_atom_ids'] = ['b']
        result = compare_rows(before, after)
        self.assertFalse(result['scores_bitwise_equal'])
        self.assertEqual(result['max_score_absolute_difference'], .25)
        self.assertFalse(result['accepted_atoms_equal'])

    def test_mismatched_request_or_candidates_fail_closed(self):
        before = self.row()
        after = self.row()
        after['question_id'] = 'other'
        with self.assertRaises(ValueError):
            compare_rows(before, after)
        after = self.row()
        after['hypergraph_evidence_plan']['candidate_replay'][0]['edge_id'] = 'other'
        with self.assertRaises(ValueError):
            compare_rows(before, after)


if __name__ == '__main__':
    unittest.main()
