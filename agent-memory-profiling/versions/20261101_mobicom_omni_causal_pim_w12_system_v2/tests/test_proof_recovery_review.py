import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

from review_proof_recovery_run import bind_judgments, mechanism_summary, quality_summary, summarize_locators
from omni_proof_join import solve_witness_join


class ProofRecoveryReviewTests(unittest.TestCase):
    def test_duplicate_judgments_are_not_silently_collapsed(self):
        judgment = {'run': '/tmp/run.json', 'question_id': 'query', 'parse_valid': True,
                    'judge_label': 'CORRECT'}
        with self.assertRaisesRegex(ValueError, 'duplicate Judge'):
            bind_judgments([judgment, dict(judgment)])

    def test_validity_must_match_judge_label(self):
        with self.assertRaisesRegex(ValueError, 'validity contract'):
            bind_judgments([{'run': '/tmp/run.json', 'question_id': 'query',
                             'parse_valid': True, 'judge_label': None}])

    def test_soft_source_location_is_not_a_hard_complementary_proof(self):
        row = {'images': [], 'proof_plan': {
            'query_parse': {'status': 'ok', 'parsed': {'demands': [{'hard': False}], 'relations': []}},
            'grounding': {'status': 'complete'},
            'selected_witnesses': [{'base_record_id': 'source'}],
            'optimal': True, 'states': 2, 'structure': {'separator_dp_used': False}}}
        summary = mechanism_summary([row])
        self.assertEqual(summary['declared_hard_demand_count_histogram'], {0: 1})
        self.assertEqual(summary['multi_hard_demand_source_located_requests'], 0)
        self.assertEqual(summary['selected_record_count_histogram'], {1: 1})
        self.assertFalse(summary['semantic_entailment_proven'])

    def test_one_witness_can_contain_multiple_source_records(self):
        row = {'images': [], 'proof_plan': {
            'query_parse': {'status': 'ok', 'parsed': {'demands': [{'hard': False}], 'relations': []}},
            'grounding': {'status': 'complete'},
            'selected_witnesses': [{'base_record_id': 'first|second', 'retrieval_record_ids': ['first', 'second']}],
            'optimal': True, 'states': 2, 'structure': {'separator_dp_used': False}}}
        summary = mechanism_summary([row])
        self.assertEqual(summary['selected_witness_count_histogram'], {1: 1})
        self.assertEqual(summary['selected_record_count_histogram'], {2: 1})
        self.assertEqual(summary['multi_record_source_located_nonfallback_requests'], 1)
        self.assertEqual(summary['multi_hard_demand_source_located_requests'], 0)

    def test_one_soft_domain_cannot_express_complementary_evidence(self):
        witnesses = [{'witness_id': 'old', 'demand_id': 'soft', 'object_ids': ['old_source'],
                      'score': .9, 'bindings': {}},
                     {'witness_id': 'new', 'demand_id': 'soft', 'object_ids': ['new_source'],
                      'score': .8, 'bindings': {}}]
        result = solve_witness_join([{'demand_id': 'soft'}], witnesses, [],
            lambda identities: {'feasible': len(identities) <= 2, 'cost': len(identities)}, time_limit=1.)
        self.assertEqual(result['accepted_object_ids'], ['old_source'])
        self.assertTrue(result['optimal'])

    def test_placeholder_filter_is_not_counted_as_raw_json_failure(self):
        records = [
            {'image_sha256': 'raw_valid', 'original_parse_valid': True, 'parse_valid': True, 'eos': True},
            {'image_sha256': 'placeholder', 'original_parse_valid': True, 'parse_valid': False, 'eos': True},
            {'image_sha256': 'truncated', 'original_parse_valid': False, 'parse_valid': False, 'eos': False}]
        summary = summarize_locators(records, {'images': 3, 'invalid_records': 2})
        self.assertEqual(summary['raw_full_parse_invalid_before_placeholder_filter'], 1)
        self.assertEqual(summary['no_complete_region_record_after_placeholder_filter'], 2)
        self.assertEqual(summary['non_eos_records'], 1)

    def test_visual_extraction_denominator_must_remain_unique(self):
        record = {'image_sha256': 'same', 'original_parse_valid': True, 'parse_valid': True, 'eos': True}
        with self.assertRaisesRegex(ValueError, 'unique identity'):
            summarize_locators([record, record], {'images': 2, 'invalid_records': 0})

    def test_invalid_judge_keeps_the_full_denominator(self):
        summary = quality_summary([{'correct': True, 'judge_parse_valid': True},
                                   {'correct': False, 'judge_parse_valid': False}])
        self.assertEqual(summary['requests'], 2)
        self.assertEqual(summary['accuracy'], .5)
        self.assertEqual(summary['invalid_counted_wrong'], 1)
        self.assertIsNone(quality_summary([])['accuracy'])


if __name__ == '__main__':
    unittest.main()
