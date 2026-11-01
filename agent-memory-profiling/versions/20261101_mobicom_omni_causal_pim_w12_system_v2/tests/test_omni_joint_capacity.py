from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from project_omni_joint_capacity import joint_projection
from pim_backend.lpddr_bank_pim import BankPimConfig


class JointCapacityTests(unittest.TestCase):
    def test_index_and_state_reserve_capacity_before_weight_admission(self):
        smoke = {'embedding_state_reuse': {'peak_resident_bytes': 8 * 1024**2,
                                           'capacity_bytes': 8 * 1024**2},
                 'embedding_windows': 1024,
                 'operator_trace': {'operators': [
                     {'name': 'weight', 'phase': 'decode', 'rows': 1, 'in_features': 2048,
                      'out_features': 2048, 'element_bytes': 2, 'weight_bytes': 8 * 1024**2, 'calls': 5}]}}
        result = joint_projection(smoke, BankPimConfig(pim_capacity_bytes=16 * 1024**2),
                                   reader_tensor_bytes=9 * 1024**3, embedding_dimension=1024,
                                   device_capacity_bytes=4 * 1024**3)
        self.assertEqual(result['weight_placement_budget_bytes'], 4 * 1024**2)
        self.assertEqual(result['placed_weight_bytes'], 0)
        self.assertFalse(result['reader_alone_fits_device'])
        self.assertFalse(result['full_device_residency_qualified'])
        self.assertIsNone(result['end_to_end_speedup'])
        self.assertTrue(result['known_joint_pool_fits'])
        smoke['embedding_state_reuse']['peak_resident_bytes'] = 1
        same_budget = joint_projection(smoke, BankPimConfig(pim_capacity_bytes=16 * 1024**2),
                                       reader_tensor_bytes=9 * 1024**3, embedding_dimension=1024,
                                       device_capacity_bytes=4 * 1024**3)
        self.assertEqual(same_budget['weight_placement_budget_bytes'], result['weight_placement_budget_bytes'])

    def test_exhausted_reservations_do_not_silently_overcommit(self):
        smoke = {'embedding_state_reuse': {'peak_resident_bytes': 8, 'capacity_bytes': 16},
                 'embedding_windows': 1, 'operator_trace': {'operators': []}}
        with self.assertRaises(ValueError):
            joint_projection(smoke, BankPimConfig(pim_capacity_bytes=16), reader_tensor_bytes=4,
                              embedding_dimension=4, device_capacity_bytes=64)

    def test_visual_budget_reserved_before_weight_placement(self):
        smoke = {'embedding_state_reuse': {'peak_resident_bytes': 8, 'capacity_bytes': 16},
                 'embedding_windows': 1, 'operator_trace': {'operators': []},
                 'visual_state_reuse': {'capacity_bytes': 32, 'peak_resident_bytes': 4}}
        result = joint_projection(smoke, BankPimConfig(pim_capacity_bytes=1024),
                                  reader_tensor_bytes=100, embedding_dimension=4,
                                  device_capacity_bytes=2048)
        self.assertEqual(result['known_reservations_total_bytes'], 64)
        self.assertEqual(result['weight_placement_budget_bytes'], 960)
        self.assertEqual(result['minimum_unique_device_bytes'], 128)
        smoke['visual_state_reuse']['peak_resident_bytes'] = 33
        with self.assertRaises(ValueError):
            joint_projection(smoke, BankPimConfig(pim_capacity_bytes=1024),
                             reader_tensor_bytes=100, embedding_dimension=4,
                             device_capacity_bytes=2048)


if __name__ == '__main__':
    unittest.main()
