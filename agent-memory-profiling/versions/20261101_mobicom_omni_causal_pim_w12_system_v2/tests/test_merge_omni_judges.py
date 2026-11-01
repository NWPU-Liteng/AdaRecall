import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from merge_omni_judges import digest, merge


class MergeOmniJudgesTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.output = self.root / 'merged.json'

    def component(self, person='0', labels=('CORRECT', None)):
        run_path = self.root / f'run_{person}.json'
        judge_path = self.root / f'judge_{person}.json'
        answer = 'saved answer'
        answer_hash = hashlib.sha256(answer.encode()).hexdigest()
        run_rows = [{'question_id': f'{person}_q_{index}', 'answer': answer,
                     'answer_sha256': answer_hash} for index in range(len(labels))]
        run = {'complete': True, 'evaluation_label_files_read': [], 'rows': run_rows}
        run_path.write_text(json.dumps(run))
        rows = [{'question_id': row['question_id'], 'run': str(run_path),
                 'prediction_sha256': answer_hash, 'judge_label': label,
                 'parse_valid': label is not None} for row, label in zip(run_rows, labels)]
        correct = labels.count('CORRECT')
        valid = sum(label is not None for label in labels)
        judge = {'complete': True, 'rows': rows, 'provenance': {
            'model_files_sha256': {'weights': 'model-hash'}, 'protocol': 'strict-json',
            'temperature': 0., 'max_new_tokens': 512, 'omni_prompt_source_sha256': 'prompt-hash',
            'prepared_audit_sha256': 'audit-hash', 'run_sha256': {str(run_path): digest(run_path)}},
            'summaries': {str(run_path): {'questions': len(rows), 'correct': correct,
                'invalid': len(rows) - valid, 'full_denominator_accuracy': correct / len(rows),
                'valid_judge_accuracy': correct / valid if valid else None,
                'official_judge_accuracy': None}}}
        judge_path.write_text(json.dumps(judge))
        return run_path, judge_path

    def mutate(self, path, change):
        content = json.loads(path.read_text())
        change(content)
        path.write_text(json.dumps(content))

    def test_merge_keeps_invalid_requests_order_and_source_hashes(self):
        first_run, first_judge = self.component()
        second_run, second_judge = self.component('1', ('WRONG',))
        result = merge([first_run, second_run], [first_judge, second_judge], self.output)
        self.assertEqual([row['question_id'] for row in result['rows']], ['0_q_0', '0_q_1', '1_q_0'])
        self.assertEqual(result['summaries'][str(first_run)]['full_denominator_accuracy'], .5)
        self.assertEqual(result['summaries'][str(first_run)]['invalid'], 1)
        self.assertEqual(result['provenance']['component_judge_sha256'][str(first_judge)], digest(first_judge))
        self.assertEqual(json.loads(self.output.read_text()), result)
        with self.assertRaises(ValueError):
            merge([first_run], [first_judge], self.output)

    def test_all_invalid_judgments_remain_in_denominator(self):
        run, judge = self.component(labels=(None, None))
        result = merge([run], [judge], self.output)
        self.assertEqual(result['summaries'][str(run)]['full_denominator_accuracy'], 0.)
        self.assertIsNone(result['summaries'][str(run)]['valid_judge_accuracy'])

    def test_changed_run_hash_rejected(self):
        run, judge = self.component()
        self.mutate(run, lambda content: content.update(extra='changed'))
        with self.assertRaisesRegex(ValueError, 'unchanged'):
            merge([run], [judge], self.output)

    def test_missing_or_reordered_judge_request_rejected(self):
        for modification in (lambda content: content['rows'].pop(),
                             lambda content: content['rows'].reverse()):
            run, judge = self.component()
            self.mutate(judge, modification)
            with self.assertRaisesRegex(ValueError, 'ordered requests'):
                merge([run], [judge], self.output)

    def test_duplicate_requests_within_or_across_runs_rejected(self):
        run, judge = self.component()
        with self.assertRaisesRegex(ValueError, 'across users'):
            merge([run, run], [judge, judge], self.output)
        self.mutate(run, lambda content: content['rows'].__setitem__(1, content['rows'][0]))
        self.mutate(judge, lambda content: content['provenance'].update(run_sha256={str(run): digest(run)}))
        with self.assertRaisesRegex(ValueError, 'within serving run'):
            merge([run], [judge], self.output)

    def test_changed_prediction_and_invalid_summary_rejected(self):
        for modification, message in (
                (lambda content: content['rows'][0].update(prediction_sha256='changed'), 'answer differs'),
                (lambda content: content['summaries'][next(iter(content['summaries']))].update(
                    full_denominator_accuracy=1.), 'denominator'),
                (lambda content: content['rows'][1].update(parse_valid=True), 'validity disagree')):
            run, judge = self.component()
            self.mutate(judge, modification)
            with self.assertRaisesRegex(ValueError, message):
                merge([run], [judge], self.output)

    def test_protocol_mismatch_missing_identity_and_incomplete_rejected(self):
        first_run, first_judge = self.component()
        second_run, second_judge = self.component('1')
        self.mutate(second_judge, lambda content: content['provenance'].update(temperature=1.))
        with self.assertRaisesRegex(ValueError, 'protocols or models differ'):
            merge([first_run, second_run], [first_judge, second_judge], self.output)
        self.mutate(first_judge, lambda content: content['provenance'].pop('model_files_sha256'))
        with self.assertRaisesRegex(ValueError, 'identity required'):
            merge([first_run], [first_judge], self.output)
        self.mutate(first_judge, lambda content: content.update(complete=False))
        with self.assertRaisesRegex(ValueError, 'must be complete'):
            merge([first_run], [first_judge], self.output)


if __name__ == '__main__':
    unittest.main()
