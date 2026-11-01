from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from omni_decode_lifecycle import same_platform_linear_lifecycle


class GeniePimRequestLifecycleTests(unittest.TestCase):
    def fixture(self):
        run = {'complete': True, 'evaluation_label_files_read': [], 'rows': [
            {'question_id': 'first', 'person_id': '0', 'answer_token_ids': [1, 2, 3], 'stop_reason': 'eos'},
            {'question_id': 'second', 'person_id': '0', 'answer_token_ids': [4, 5], 'stop_reason': 'max_new_tokens'}]}
        costs = {'resident': {'calls': 3, 'host_per_call_ns': 100., 'pim_per_call_ns': 20.},
                 'host': {'calls': 3, 'host_per_call_ns': 30., 'pim_per_call_ns': 40.}}
        return run, costs

    def test_cold_writes_charged_without_double_counting_residency(self):
        run, costs = self.fixture()
        ledger = same_platform_linear_lifecycle(run, costs, {'resident'}, 12.)
        self.assertEqual(ledger['known_linear_host_ns'], 390.)
        self.assertEqual(ledger['known_linear_warm_hybrid_ns'], 150.)
        self.assertEqual(ledger['persistent_stream_minimum_layout_write_ns'], 12.)
        self.assertEqual(ledger['cold_per_request_minimum_layout_write_ns'], 24.)
        self.assertEqual([row['known_linear_persistent_stream_ns_with_minimum_placement']
                          for row in ledger['requests']], [112., 50.])
        self.assertTrue(all(row['full_request_hybrid_ns'] is None for row in ledger['requests']))
        self.assertIsNone(ledger['unmapped_work_ns'])

    def test_lost_or_duplicated_decode_and_cross_user_pooling_rejected(self):
        for modification in (
                lambda run, costs: costs['resident'].update(calls=2),
                lambda run, costs: run['rows'][1].update(question_id='first'),
                lambda run, costs: run['rows'][1].update(person_id='1'),
                lambda run, costs: run['rows'][1].update(answer_token_ids=[])):
            run, costs = self.fixture()
            modification(run, costs)
            with self.assertRaises(ValueError):
                same_platform_linear_lifecycle(run, costs, {'resident'}, 12.)

    def test_unmapped_placement_and_nonfinite_cost_rejected(self):
        run, costs = self.fixture()
        with self.assertRaises(ValueError):
            same_platform_linear_lifecycle(run, costs, {'unknown'}, 12.)
        for field in ('host_per_call_ns', 'pim_per_call_ns'):
            invalid = deepcopy(costs)
            invalid['resident'][field] = float('nan')
            with self.assertRaises(ValueError):
                same_platform_linear_lifecycle(run, invalid, {'resident'}, 12.)


if __name__ == '__main__':
    unittest.main()
