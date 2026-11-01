from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from replay_omni_online_residency import replay_user
from tests.test_omni_decode_lifecycle import fixture


class OnlineResidencyReplayTests(unittest.TestCase):
    def fixture(self):
        run, schedules, costs, placement, copy_range = fixture()
        identity = 'parameter:projection.weight'
        costs[identity].update({'host_external_bytes': 300, 'pim_external_bytes': 20,
            'phase_costs': {name: {'min_cycles': low, 'max_cycles': high}
                for name, low, high in (('host', 100, 110), ('pim', 45, 50), ('placement', 119, 120))}})
        placement[identity]['resident_bytes'] = 256
        for schedule in schedules:
            schedule['context_identity'] = schedule['question_id']
            schedule['events'][0]['event_id'] = 9
        routes = {'costs': costs, 'placement': placement, 'placement_cycle_range': copy_range,
            'capacity_bytes': 320, 'serving_state_reservations': {'reserved_bytes': 64}}
        return run, schedules, routes

    def test_cross_request_residency_without_reading_future_horizon(self):
        run, schedules, routes = self.fixture()
        result = replay_user(run, schedules, routes)
        self.assertEqual(result['requests'], 2)
        self.assertEqual(result['totals']['placement_operations'], 1)
        self.assertEqual(result['admissions'][0]['observed_linear_call_index'], 2)
        self.assertEqual(result['totals']['host_min_cycles'], 300)
        self.assertEqual(result['known_linear_online_max_cycles_including_placement'], 330)
        self.assertEqual(result['resident_weight_bytes'], 256)
        self.assertTrue(result['fixed_pool_certificate']['weights'][0]['twice_oracle_bound_pass'])
        self.assertLess(result['known_linear_host_best_over_online_worst_scenario'], 1.)

    def test_new_user_is_cold_and_aggregate_tampering_rejected(self):
        run, schedules, routes = self.fixture()
        self.assertEqual(replay_user(run, schedules, routes), replay_user(run, schedules, routes))
        changed = deepcopy(run)
        changed['operator_trace']['operators'][0]['calls'] -= 1
        with self.assertRaises(ValueError):
            replay_user(changed, schedules, routes)


if __name__ == '__main__':
    unittest.main()
