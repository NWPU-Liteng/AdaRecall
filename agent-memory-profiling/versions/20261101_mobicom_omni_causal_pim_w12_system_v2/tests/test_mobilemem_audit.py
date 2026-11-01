from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from audit_mobilemem_development import distribution, numeric_coverage
from freeze_mobilemem_development import promotion_gates
from system_audit import SCHEMA, system_checks


class MobileMemAuditTests(unittest.TestCase):
    def test_distribution_keeps_full_values(self):
        result = distribution([1, None, 3, 20])
        self.assertEqual(result['count'], 3)
        self.assertEqual(result['p50'], 3)
        self.assertEqual(result['mean'], 8)
        self.assertEqual(result['max'], 20)

    def test_empty_distribution_is_not_zero_latency(self):
        self.assertIsNone(distribution([]))
        self.assertIsNone(distribution([None]))

    def test_promotion_uses_inclusive_latency_without_rounding(self):
        audit = {'engineering_pass': True, 'source_event_hit_recall': 0.97,
                 'questions': 32,
                 'execution_selection': {'limit': 768, 'sample_count': 32, 'shard_count': 1, 'shard_index': 0},
                 'schema': SCHEMA, 'system_v2_pass': True, 'protocol': 'audit',
                 'local_judge': {'local_semantic_answer_accuracy': 0.90},
                 'service_including_validation_seconds': {'p95': 20.0407},
                 'service_excluding_validation_seconds': {'p95': 19.417},
                 'generation_limit_counts': {'first': 0, 'proof': 0}}
        gates = promotion_gates(audit)
        self.assertFalse(gates['host_service_p95_le_20_seconds'])
        self.assertFalse(all(gates.values()))
        audit['service_including_validation_seconds']['p95'] = 20.0
        self.assertTrue(all(promotion_gates(audit).values()))

    def test_small_or_cold_sharded_stream_cannot_promote_to_development128(self):
        for questions, selection in (
                (16, {'limit': 16, 'sample_count': 0, 'shard_count': 1, 'shard_index': 0}),
                (32, {'limit': 32, 'sample_count': 32, 'shard_count': 1, 'shard_index': 0}),
                (32, {'limit': 768, 'sample_count': 32, 'shard_count': 2, 'shard_index': 0}),
                (32, {})):
            audit = {'engineering_pass': True, 'source_event_hit_recall': 1.0,
                     'schema': SCHEMA, 'system_v2_pass': True, 'protocol': 'audit',
                     'questions': questions, 'execution_selection': selection,
                     'local_judge': {'local_semantic_answer_accuracy': 1.0},
                     'service_including_validation_seconds': {'p95': 1.0},
                     'generation_limit_counts': {'first': 0, 'proof': 0}}
            self.assertFalse(promotion_gates(audit)['development32_temporal_single_stream'])

    def test_unscored_quality_never_promotes(self):
        audit = {'engineering_pass': True, 'source_event_hit_recall': 1.0,
                 'local_judge': None, 'service_including_validation_seconds': {'p95': 1},
                 'generation_limit_counts': {'first': 0, 'proof': 0}}
        self.assertFalse(promotion_gates(audit)['local_development_semantic_accuracy_ge_0_75'])

    def test_disabled_numeric_validation_is_not_coverage(self):
        self.assertFalse(numeric_coverage({'prefix_validation': {'status': 'not_requested'},
                                           'prefix_state_reuse': {'status': 'exact_token_prefix_restored'}}))
        self.assertFalse(numeric_coverage({'prefix_validation': {'status': 'pass'}}))
        complete = {'status': 'pass', 'full_greedy_token_ids_equal': True, 'stop_reason_equal': True}
        self.assertTrue(numeric_coverage({'prefix_validation': complete}))
        self.assertTrue(numeric_coverage({'prefix_validation': {**complete, 'status': 'fail'},
                                          'prefix_state_reuse': {'status': 'numeric_contract_full_recompute'}}))
        self.assertTrue(numeric_coverage({'numeric_audit_scheduled': False,
                                          'prefix_validation': {'status': 'not_requested'}}))
        self.assertFalse(numeric_coverage({'numeric_audit_scheduled': True,
                                           'prefix_validation': {'status': 'not_requested'}}))

    def test_legacy_and_empty_system_never_promote(self):
        checks = system_checks([], {'reader_protocol': 'single'})
        self.assertFalse(all(checks.values()))
        legacy = {'engineering_pass': True, 'source_event_hit_recall': 1.0,
                  'local_judge': {'local_semantic_answer_accuracy': 1.0},
                  'service_including_validation_seconds': {'p95': 1.0},
                  'generation_limit_counts': {}}
        self.assertFalse(promotion_gates(legacy)['system_v2_audit'])


if __name__ == '__main__':
    unittest.main()
