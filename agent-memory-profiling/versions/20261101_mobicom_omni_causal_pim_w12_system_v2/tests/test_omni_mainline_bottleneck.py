from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from diagnose_omni_mainline_bottleneck import account_stages


class MainlineBottleneckTests(unittest.TestCase):
    def setUp(self):
        self.smoke = {'complete': True, 'evaluation_label_files_read': [],
            'numerical_probe_overhead_in_service': False, 'rows': [{
                'question_id': 'request', 'service_seconds': 20., 'query_retrieval_seconds': 1.,
                'edge_embedding_seconds': 2., 'evidence_planning_seconds': 3.,
                'prefill_seconds': 8., 'decode_seconds': 4.,
                'visual_cache_before': {'resolve_seconds': 2., 'miss_compute_seconds': 1., 'hits': 0, 'misses': 1},
                'visual_cache_after': {'resolve_seconds': 8., 'miss_compute_seconds': 6., 'hits': 0, 'misses': 2},
                'reference_vision_seconds': [1000.], 'generation_validation_seconds': 2000.}]}

    def test_disjoint_accounting_excludes_validation_and_subtracts_vision_once(self):
        result = account_stages(self.smoke)
        stages = result['disjoint_stage_seconds']
        self.assertEqual(stages['vision_cache_resolution_including_encoder_on_miss'], 6.)
        self.assertEqual(stages['remaining_reader_prefill'], 2.)
        self.assertEqual(stages['other_serving_work'], 2.)
        self.assertEqual(sum(stages.values()), 20.)
        self.assertAlmostEqual(sum(result['disjoint_stage_service_fractions'].values()), 1.)
        self.assertIsNone(result['end_to_end_pim_speedup'])

    def test_missing_actual_counters_cannot_use_reference_timing(self):
        del self.smoke['rows'][0]['visual_cache_before']
        with self.assertRaises(ValueError):
            account_stages(self.smoke)

    def test_invalid_stage_nesting_is_rejected(self):
        self.smoke['rows'][0]['visual_cache_after']['resolve_seconds'] = 100.
        with self.assertRaises(ValueError):
            account_stages(self.smoke)

    def test_nonmonotone_or_multiple_lookups_are_rejected(self):
        for field, value in (('resolve_seconds', 0.), ('misses', 3)):
            smoke = deepcopy(self.smoke)
            smoke['rows'][0]['visual_cache_after'][field] = value
            with self.assertRaises(ValueError):
                account_stages(smoke)

    def test_invalid_totals_duplicates_and_nonfinite_timings_are_rejected(self):
        for change in ('short-total', 'duplicate', 'nan', 'probe'):
            smoke = deepcopy(self.smoke)
            if change == 'short-total':
                smoke['rows'][0]['service_seconds'] = 1.
            elif change == 'duplicate':
                smoke['rows'].append(deepcopy(smoke['rows'][0]))
            elif change == 'nan':
                smoke['rows'][0]['decode_seconds'] = float('nan')
            else:
                smoke['numerical_probe_overhead_in_service'] = True
            with self.assertRaises(ValueError):
                account_stages(smoke)


if __name__ == '__main__':
    unittest.main()
