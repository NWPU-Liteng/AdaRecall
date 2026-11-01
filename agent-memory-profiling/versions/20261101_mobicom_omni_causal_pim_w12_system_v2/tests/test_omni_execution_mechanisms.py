from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from screen_omni_execution_mechanisms import summarize


class ExecutionMechanismTests(unittest.TestCase):
    def audit(self):
        return {'complete': True, 'requests': 1, 'complete_greedy_token_matches': 1, 'users': [
            {'rows': [{'question_id': 'first', 'input_tokens': 100,
                'prefix_opportunity': {'matched_prefix_tokens': 0}, 'graph_analysis': {
                    'shared_linear_input_groups_with_unchanged_write_epoch': [[
                        {'input_logical_bytes': 16, 'out_features': 32},
                        {'input_logical_bytes': 16, 'out_features': 64}]],
                    'linear_weight_logical_read_bytes_per_step': 16000,
                    'operator_counts': {}, 'visible_mutated_cache_storage_bytes': 32}}]}]}

    def test_bank_input_dedup_does_not_remove_weight_payload(self):
        summary = summarize(self.audit(), 32)
        row = summary['all_request_mechanism_screen'][0]
        self.assertEqual(row['conditional_repeated_bank_input_payload_bytes'], 512)
        self.assertEqual(row['linear_weight_logical_read_bytes'], 16000)
        self.assertFalse(summary['shared_input_pim_fusion_executed'])
        self.assertIsNone(summary['end_to_end_speedup'])

    def test_all_requests_and_identical_actual_inputs_required(self):
        audit = self.audit()
        audit['requests'] = 2
        with self.assertRaises(ValueError):
            summarize(audit, 32)
        audit = self.audit()
        group = audit['users'][0]['rows'][0]['graph_analysis']['shared_linear_input_groups_with_unchanged_write_epoch'][0]
        group[0]['input_logical_bytes'] = 8
        with self.assertRaises(ValueError):
            summarize(audit, 32)


if __name__ == '__main__':
    unittest.main()
