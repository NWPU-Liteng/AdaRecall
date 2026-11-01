from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from summarize_omni_service_repeats import summarize_repeats


def pair():
    return {'complete': True, 'execution_integrity': 'pass', 'requests': 4,
        'pair_mismatches': [], 'evaluation_label_files_read': [],
        'per_user': [{'person_id': '0', 'requests': 4, 'measurement_contract': {'gpu': 'same'},
                      'reference_service_seconds': {'sum_seconds': 20.},
                      'candidate_service_seconds': {'sum_seconds': 10.}}],
        'measured_server_end_to_end_speedup': 2.,
        'reference': {'service_seconds': {'sum_seconds': 20.},
            'fine_grained_service_stages': {'disjoint_seconds': {'prefill_seconds': {'sum_seconds': 12.}}}},
        'candidate': {'service_seconds': {'sum_seconds': 10.},
            'fine_grained_service_stages': {'disjoint_seconds': {'prefill_seconds': {'sum_seconds': 3.}}}}}


class ServiceRepeatTests(unittest.TestCase):
    def test_repeats_do_not_inflate_unique_cohort_or_causal_claim(self):
        result = summarize_repeats([pair(), pair()])
        self.assertEqual(result['unique_requests'], 4)
        self.assertEqual(result['request_observations_per_path'], 8)
        self.assertEqual(result['pooled_observed_server_latency_ratio'], 2.)
        self.assertEqual(result['disjoint_stage_totals_seconds']['prefill_seconds']['observed_difference_seconds'], 18.)
        self.assertIsNone(result['causally_attributable_reuse_speedup'])
        self.assertIn('prefill_seconds', result['untargeted_stages_not_reuse_credit'])

    def test_changed_gpu_cohort_or_failed_equivalence_cannot_be_pooled(self):
        for mutation in ('gpu', 'cohort', 'equivalence'):
            changed = deepcopy(pair())
            if mutation == 'gpu':
                changed['per_user'][0]['measurement_contract']['gpu'] = 'different'
            elif mutation == 'cohort':
                changed['requests'] = 3
            else:
                changed['execution_integrity'] = 'failed_pair_equivalence_gate'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                summarize_repeats([pair(), changed])


if __name__ == '__main__':
    unittest.main()
