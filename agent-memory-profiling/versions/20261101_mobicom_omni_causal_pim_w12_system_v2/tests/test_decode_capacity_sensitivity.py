import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from project_decode_capacity_sensitivity import project_capacity_sensitivity


class DecodeCapacitySensitivityTests(unittest.TestCase):
    def test_capacity_is_explicit_and_changes_placement(self):
        run = {
            'complete': True,
            'evaluation_label_files_read': [],
            'pre_reader_operator_profile': {'enabled': True},
            'rows': [{
                'question_id': 'q0',
                'service_seconds': 10.0,
                'proof_plan': {
                    'retrieval': {'linear_operator_trace': {'operators': [
                        {'name': 'a', 'phase': 'decode', 'rows': 1,
                         'in_features': 256, 'out_features': 256,
                         'element_bytes': 2, 'weight_bytes': 131072, 'calls': 2},
                        {'name': 'b', 'phase': 'decode', 'rows': 1,
                         'in_features': 256, 'out_features': 256,
                         'element_bytes': 2, 'weight_bytes': 131072, 'calls': 2},
                    ]}},
                    'grounding': {'frontier_attempts': []},
                },
            }],
        }
        profile = {
            'name': 'test',
            'calibration_status': 'test_only',
            'physical_capacity_bytes': 2 * 1024**2,
            'config': {
                'pim_capacity_bytes': 131072,
                'local_buffer_bytes_per_bank': 65536,
            },
        }
        result = project_capacity_sensitivity(run, profile, [0.0001, 0.001])
        self.assertEqual(result['capacity_scenarios'][0]['resident_matrices'], 0)
        self.assertEqual(result['capacity_scenarios'][1]['resident_matrices'], 2)
        self.assertTrue(result['interpretation']['capacity_can_be_overridden'])
        self.assertTrue(result['capacity_scenarios'][0]['physically_admissible_under_profile'])
        self.assertFalse(result['capacity_scenarios'][1]['physically_admissible_under_profile'])
        self.assertTrue(result['capacity_scenarios'][1]['within_profile_physical_capacity'])
        self.assertEqual(
            result['capacity_scenarios'][1]['physical_capacity_status'],
            'requires_pool_reallocation_and_host_reserve_validation')

    def test_capacity_above_physical_limit_is_hypothetical(self):
        run = {
            'complete': True,
            'evaluation_label_files_read': [],
            'pre_reader_operator_profile': {'enabled': True},
            'rows': [{'question_id': 'q0', 'proof_plan': {
                'retrieval': {'linear_operator_trace': {'operators': [
                    {'name': 'a', 'phase': 'decode', 'rows': 1,
                     'in_features': 256, 'out_features': 256,
                     'element_bytes': 2, 'weight_bytes': 131072, 'calls': 1},
                ]}}, 'grounding': {'frontier_attempts': []},
            }}],
        }
        profile = {
            'name': 'test',
            'calibration_status': 'test_only',
            'physical_capacity_bytes': 2 * 1024**2,
            'config': {'pim_capacity_bytes': 131072,
                       'local_buffer_bytes_per_bank': 65536},
        }
        scenario = project_capacity_sensitivity(run, profile, [0.003])[
            'capacity_scenarios'][0]
        self.assertFalse(scenario['within_profile_physical_capacity'])
        self.assertTrue(scenario['requires_new_physical_profile'])
        self.assertEqual(scenario['physical_capacity_status'],
                         'exceeds_profile_physical_capacity')


if __name__ == '__main__':
    unittest.main()
