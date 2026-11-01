from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from inherit_omni_quality_if_identical import digest, inherit_quality


class QualityInheritanceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.reference = Path(self.directory.name) / 'reference'
        self.candidate = Path(self.directory.name) / 'candidate'
        self.reference.mkdir()
        self.candidate.mkdir()
        answer = 'saved answer'
        self.row = dict(question_id='0_q_0', person_id='0', question_sha256='question',
                        images=[], answer_token_ids=[1, 2], answer=answer,
                        stop_reason='max_new_tokens', input_tokens=3, visual_patches=0,
                        answer_sha256=hashlib.sha256(answer.encode()).hexdigest(),
                        processor_inputs_sha256={'input_ids': 'tokens'})
        self.run = dict(complete=True, rows=[self.row], evaluation_label_files_read=[],
                        prepared_audit_sha256='prepared')
        for root in (self.reference, self.candidate):
            (root / 'reference_user_0.json').write_text(json.dumps(self.run))
        self.reference_path = self.reference / 'reference_user_0.json'
        self.judge = dict(complete=True, provenance=dict(model_files_sha256={'model': 'model'},
            omni_prompt_source_sha256='prompt', prepared_audit_sha256='prepared', protocol='pinned',
            run_sha256={str(self.reference_path): digest(self.reference_path)}),
            rows=[dict(run=str(self.reference_path), question_id='0_q_0',
                prediction_sha256=self.row['answer_sha256'], stop_reason='max_new_tokens',
                response='{"label":"CORRECT"}', judge_label='CORRECT', parse_valid=True)],
            summaries={str(self.reference_path): dict(questions=1, correct=1, invalid=0,
                                                     full_denominator_accuracy=1.)})
        self.cohort = {'question_ids_by_user': {'0': ['0_q_0']}}

    def inherit(self, judge=None):
        return inherit_quality(self.reference, self.candidate, self.cohort,
                               self.judge if judge is None else judge)

    def test_exact_answers_inherit_without_dropping_truncation(self):
        result = self.inherit()
        self.assertEqual(result['requests'], 1)
        self.assertEqual(result['correct'], 1)
        self.assertEqual(result['strict_eos_correct'], 0)
        self.assertEqual(result['truncated_question_ids'], ['0_q_0'])

    def test_changed_tokens_processor_user_payload_and_incomplete_run_rejected(self):
        for mutate in (
            lambda run: run['rows'][0].update(answer_token_ids=[1, 3]),
            lambda run: run['rows'][0].update(processor_inputs_sha256={'input_ids': 'changed'}),
            lambda run: run['rows'][0].update(person_id='1'),
            lambda run: run['rows'][0].update(answer='tampered'),
            lambda run: run.update(complete=False),
        ):
            with self.subTest(mutate=mutate):
                run = deepcopy(self.run)
                mutate(run)
                (self.candidate / 'reference_user_0.json').write_text(json.dumps(run))
                with self.assertRaises(ValueError):
                    self.inherit()

    def test_tampered_judge_rows_summary_and_provenance_rejected(self):
        for mutate in (
            lambda judge: judge['provenance']['run_sha256'].update({str(self.reference_path): 'bad'}),
            lambda judge: judge['rows'].clear(),
            lambda judge: judge['rows'][0].update(judge_label='WRONG'),
            lambda judge: judge['rows'][0].update(prediction_sha256='bad'),
            lambda judge: judge['summaries'][str(self.reference_path)].update(correct=0),
        ):
            judge = deepcopy(self.judge)
            mutate(judge)
            with self.assertRaises(ValueError):
                self.inherit(judge)

    def test_duplicate_cohort_rejected(self):
        self.cohort['question_ids_by_user']['0'].append('0_q_0')
        with self.assertRaises(ValueError):
            self.inherit()

    def test_sealed_users_rejected_before_files_are_opened(self):
        self.cohort = {'question_ids_by_user': {'3': ['3_q_0']}}
        with self.assertRaisesRegex(ValueError, 'authorized development'):
            self.inherit()


if __name__ == '__main__':
    unittest.main()
