from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from omni_decode_lifecycle import decode_calls, lifecycle_for_user, normalized_service_sensitivity


def fixture():
    run = {'complete': True, 'evaluation_label_files_read': [],
        'rows': [{'person_id': '0', 'question_id': 'first', 'answer_token_ids': [2, 3, 4],
                  'stop_reason': 'eos'},
                 {'person_id': '0', 'question_id': 'second', 'answer_token_ids': [5, 6],
                  'stop_reason': 'max_new_tokens'}],
        'operator_trace': {'operators': [{'phase': 'decode', 'name': 'projection', 'rows': 1,
            'in_features': 8, 'out_features': 16, 'element_bytes': 2, 'weight_bytes': 256, 'calls': 3}]}}
    totals = {'host_min_cycles': 100, 'host_max_cycles': 110, 'hybrid_min_cycles': 45,
        'hybrid_max_cycles': 50, 'host_external_bytes': 300, 'hybrid_external_bytes': 20, 'pim_calls': 1}
    schedules = [{'question_id': row['question_id'], 'person_id': '0',
                  'known_linear_subgraph': totals,
                  'events': [{'kind': 'modeled_linear', 'parameter_identity': 'parameter:projection.weight',
                              'route': 'pim'}]} for row in run['rows']]
    costs = {'parameter:projection.weight': {'inputs': 8, 'outputs': 16, 'element_bytes': 2,
                                             'resident_bytes': 256}}
    return run, schedules, costs, {'parameter:projection.weight': {}}, {'min_cycles': 29, 'max_cycles': 30}


class DecodeLifecycleTests(unittest.TestCase):
    def test_prefill_emits_first_token_and_every_stop_preserves_denominator(self):
        run, *rest = fixture()
        self.assertEqual([decode_calls(row) for row in run['rows']], [2, 1])
        result = lifecycle_for_user(run, *rest)
        self.assertEqual(result['requests'], 2)
        self.assertEqual(result['decode_invocations'], 3)
        self.assertEqual(result['linear_calls'], 3)
        self.assertEqual(result['persistent_stream_hybrid_max_cycles'], 180)
        self.assertEqual(result['cold_per_request_hybrid_max_cycles'], 210)
        self.assertEqual(result['persistent_stream_known_linear_speedup_lower_scenario'], 300 / 180)
        self.assertIn('opaque inclusive durations', result['unmodeled'])

    def test_aggregate_trace_must_cover_all_calls_identities_and_shapes(self):
        for mutation in ('count', 'shape', 'name', 'weight', 'rows', 'missing', 'order', 'placement'):
            run, schedules, costs, placement, copy_range = deepcopy(fixture())
            operator = run['operator_trace']['operators'][0]
            if mutation == 'count':
                operator['calls'] -= 1
            elif mutation == 'shape':
                operator['in_features'] += 1
            elif mutation == 'name':
                operator['name'] = 'other'
            elif mutation == 'weight':
                operator['weight_bytes'] += 1
            elif mutation == 'rows':
                operator['rows'] = 2
            elif mutation == 'missing':
                run['operator_trace']['operators'] = []
            elif mutation == 'order':
                schedules.reverse()
            else:
                placement = {}
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                lifecycle_for_user(run, schedules, costs, placement, copy_range)

    def test_sensitivity_keeps_unknown_work_explicit_and_dimensionless(self):
        self.assertEqual(normalized_service_sensitivity(.3, 2., 0.)['conditional_service_speedup'], 1.)
        self.assertAlmostEqual(normalized_service_sensitivity(.3, 2., 1.)['conditional_service_speedup'], 1 / .85)
        self.assertLess(normalized_service_sensitivity(.3, .9, 1.)['conditional_service_speedup'], 1.)
        for args in ((-1, 2, .5), (.5, 0, .5), (.5, 2, 2), (.5, float('nan'), .5)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                normalized_service_sensitivity(*args)


if __name__ == '__main__':
    unittest.main()
