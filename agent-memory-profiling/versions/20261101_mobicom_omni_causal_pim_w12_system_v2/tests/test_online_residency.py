from copy import deepcopy
import unittest

from pim_backend.online_residency import OnlineResidency


def fixture(host=13, pim=3, purchase=25, size=8):
    cost = {'resident_bytes': size, 'host_external_bytes': 20, 'pim_external_bytes': 4,
        'phase_costs': {name: {'min_cycles': value, 'max_cycles': value}
                      for name, value in (('host', host), ('pim', pim), ('placement', purchase))}}
    return {'weight': cost}, {'weight': {'resident_bytes': size}}


class OnlineResidencyTests(unittest.TestCase):
    def test_current_call_threshold_charges_purchase_once(self):
        costs, pool = fixture()
        gate = OnlineResidency(costs, pool, capacity_bytes=8)
        records = [gate.dispatch('weight') for _ in range(5)]
        self.assertEqual([record['route'] for record in records], ['host', 'host', 'pim', 'pim', 'pim'])
        self.assertEqual([record['hybrid_max_cycles'] for record in records], [13, 13, 3, 3, 3])
        self.assertEqual([record['placement_max_cycles'] for record in records], [0, 0, 25, 0, 0])
        self.assertEqual(sum(record['hybrid_external_bytes'] for record in records), 60)
        self.assertEqual(gate.certificate()['resident_bytes'], 8)

    def test_exhaustive_integer_cost_certificate_and_short_horizon_risk(self):
        for delta in (1, 3, 10, 50):
            for purchase in range(0, 51):
                costs, pool = fixture(host=delta + 3, purchase=purchase)
                gate = OnlineResidency(costs, pool, capacity_bytes=8)
                for _ in range(55):
                    gate.dispatch('weight')
                    self.assertTrue(gate.certificate()['weights'][0]['twice_oracle_bound_pass'])
        costs, pool = fixture()
        gate = OnlineResidency(costs, pool, capacity_bytes=8)
        records = [gate.dispatch('weight') for _ in range(3)]
        total_with_purchase = sum(record['hybrid_max_cycles'] + record['placement_max_cycles'] for record in records)
        self.assertGreater(total_with_purchase, sum(record['host_max_cycles'] for record in records))
        self.assertFalse(gate.certificate()['no_loss_against_all_host_guaranteed'])

    def test_future_suffix_cannot_change_current_decision(self):
        costs, pool = fixture()
        short = OnlineResidency(costs, pool, capacity_bytes=8)
        long = OnlineResidency(costs, pool, capacity_bytes=8)
        prefix = [short.dispatch('weight') for _ in range(2)]
        extended = [long.dispatch('weight') for _ in range(20)]
        self.assertEqual(prefix, extended[:2])

    def test_stream_isolation_and_nonresident_host_fallback(self):
        costs, pool = fixture()
        first = OnlineResidency(costs, pool, capacity_bytes=8)
        second = OnlineResidency(costs, pool, capacity_bytes=8)
        for _ in range(3):
            first.dispatch('weight')
        self.assertTrue(first.resident)
        self.assertFalse(second.dispatch('weight')['placement_now'])
        fallback = OnlineResidency(costs, {}, capacity_bytes=0)
        for _ in range(5):
            self.assertEqual(fallback.dispatch('weight')['route'], 'host')
        with self.assertRaises(ValueError):
            fallback.dispatch('unknown')

    def test_capacity_reservations_and_nonpositive_gain_rejected(self):
        costs, pool = fixture()
        for capacity, reservation in ((7, 0), (8, 1), (8, -1), (True, 0)):
            with self.assertRaises(ValueError):
                OnlineResidency(costs, pool, capacity_bytes=capacity, reserved_bytes=reservation)
        unprofitable = deepcopy(costs)
        unprofitable['weight']['phase_costs']['pim']['max_cycles'] = 13
        with self.assertRaises(ValueError):
            OnlineResidency(unprofitable, pool, capacity_bytes=8)
        corrupt = deepcopy(costs)
        corrupt['weight']['phase_costs']['placement']['min_cycles'] = -1
        with self.assertRaises(ValueError):
            OnlineResidency(corrupt, pool, capacity_bytes=8)


if __name__ == '__main__':
    unittest.main()
