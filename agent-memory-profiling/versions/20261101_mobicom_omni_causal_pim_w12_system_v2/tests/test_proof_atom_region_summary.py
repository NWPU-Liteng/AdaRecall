import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

from summarize_proof_atom_region import (bind_quality, summarize_budgets,
                                         summarize_locator_costs, summarize_retained_domains)


class ProofAtomRegionSummaryTests(unittest.TestCase):
    def test_judge_is_bound_to_exact_answer_and_full_denominator(self):
        root = Path('/tmp/candidate')
        rows = [{'person_id': '0', 'question_id': 'first', 'answer_sha256': 'first-answer'},
                {'person_id': '0', 'question_id': 'second', 'answer_sha256': 'second-answer'}]
        judgments = [{'run': str(root / 'proof_user_0.json'), 'question_id': row['question_id'],
                      'prediction_sha256': row['answer_sha256'], 'parse_valid': index == 0,
                      'judge_label': 'CORRECT' if index == 0 else None}
                     for index, row in enumerate(rows)]
        bound = bind_quality(rows, judgments, root)
        self.assertEqual(len(bound), 2)
        self.assertEqual(sum(judgment['parse_valid'] for _, judgment in bound), 1)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            bind_quality(rows, judgments + [judgments[0]], root)
        with self.assertRaisesRegex(ValueError, 'denominator'):
            bind_quality(rows, judgments[:1], root)
        changed = copy.deepcopy(judgments)
        changed[0]['prediction_sha256'] = 'different-answer'
        with self.assertRaisesRegex(ValueError, 'binding'):
            bind_quality(rows, changed, root)

    def test_judge_validity_cannot_disagree_with_label(self):
        with self.assertRaisesRegex(ValueError, 'validity'):
            bind_quality([], [{'run': 'run', 'question_id': 'query', 'parse_valid': False,
                               'judge_label': 'CORRECT'}], Path('/tmp'))

    def test_offline_locator_cost_is_a_separate_historical_dependency(self):
        manifest = {'locator_extraction_manifest': {'images': 2, 'shards': [
            {'directory': 'first', 'status': {'complete': True, 'total': 1, 'elapsed_seconds': 10.,
                'load_and_identity_seconds': 2., 'batches': [{'seconds': 7.}]}},
            {'directory': 'second', 'status': {'complete': True, 'total': 1, 'elapsed_seconds': 12.,
                'load_and_identity_seconds': 3., 'batches': [{'seconds': 8.}]}}]}}
        result = summarize_locator_costs(manifest)
        self.assertEqual(result['shard_elapsed_seconds_sum'], 22.)
        self.assertEqual(result['shard_elapsed_seconds_max'], 12.)
        self.assertEqual(result['batch_seconds_sum'], 15.)
        self.assertFalse(result['rerun_for_this_candidate'])

    def test_final_text_and_image_budgets_are_checked_independently(self):
        row = {'question_id': 'query', 'input_tokens': 500, 'images': [],
               'proof_plan': {'final_render': {'input_tokens': 500, 'text_tokens': 200}}}
        protocol = {'input_tokens': 8192, 'text_tokens': 3600, 'max_images': 4}
        self.assertTrue(summarize_budgets([row], protocol)['passed'])
        excessive = copy.deepcopy(row)
        excessive['proof_plan']['final_render']['text_tokens'] = 3601
        self.assertEqual(summarize_budgets([excessive], protocol)['violating_question_ids'], ['query'])
        excessive = copy.deepcopy(row)
        excessive['images'] = [None] * 5
        self.assertFalse(summarize_budgets([excessive], protocol)['passed'])

    def test_omitted_demand_record_pair_is_not_a_missing_retained_base_record(self):
        row = {'question_id': 'query', 'proof_plan': {
            'grounding': {'alias_to_record_id': {'r0': 'same-source'},
                          'omitted_whole_record_ids_due_to_context': ['same-source#d0']},
            'selected_witnesses': [{'retrieval_record_ids': ['same-source']}]}}
        result = summarize_retained_domains([row])
        self.assertFalse(result['selected_witnesses_with_unpresented_sources'])
        self.assertEqual(result['omitted_demand_record_pairs']['total'], 1)
        row['proof_plan']['selected_witnesses'][0]['retrieval_record_ids'] = ['unpresented']
        self.assertEqual(summarize_retained_domains([row])['selected_witnesses_with_unpresented_sources'],
                         [{'question_id': 'query', 'unpresented_records': ['unpresented']}])


if __name__ == '__main__':
    unittest.main()
