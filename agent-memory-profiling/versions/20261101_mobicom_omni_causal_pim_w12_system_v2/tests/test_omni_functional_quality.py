from pathlib import Path
import contextlib
import io
import json
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from audit_omni_functional_quality import matched_quality
import evaluate_omni_matched_bank as evaluator


class FunctionalQualityTests(unittest.TestCase):
    def write_evaluation_fixture(self, output_root, reverse_identities=False):
        identities = [
            {'evaluation_id': 'matched_host/q1', 'mode': 'matched_host', 'question_id': 'q1'},
            {'evaluation_id': 'matched_replace/q1', 'mode': 'matched_replace', 'question_id': 'q1'},
        ]
        predictions, requests, judgments = [], [], []
        for identity, success in zip(identities, (True, False)):
            evaluation_id = identity['evaluation_id']
            predictions.append({'question_id': evaluation_id, 'person_id': '0',
                                'answer_sha256': 'answer', 'stop_reason': 'eos'})
            requests.append({'question_id': evaluation_id, 'prediction_sha256': 'answer',
                             'prompt_sha256': 'prompt', 'reader_stop_reason': 'eos'})
            judgments.append({'question_id': evaluation_id, 'prediction_sha256': 'answer',
                              'prompt_sha256': 'prompt', 'parse_valid': True,
                              'value': float(success)})
        requests_path = output_root / 'judge_requests.json'
        requests_path.write_text(json.dumps({'judge_requests': requests}))
        (output_root / 'judge.json').write_text(json.dumps({'complete': True, 'rows': judgments,
            'provenance': {'scores_sha256': evaluator.digest(requests_path)}}))
        (output_root / 'evaluation_plan.json').write_text(json.dumps({
            'predictions': predictions, 'identities': identities[::-1] if reverse_identities else identities,
            'evaluator_sha256': evaluator.digest(Path(evaluator.__file__)), 'inputs_sha256': {},
            'judge_requests_sha256': evaluator.digest(requests_path), 'functional': {}}))

    def test_restores_mode_prefixed_ids_before_matched_comparison(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            output_root = Path(directory)
            self.write_evaluation_fixture(output_root)
            evaluator.close(output_root)
            result = json.loads((output_root / 'summary.json').read_text())
        comparison = result['matched_quality']
        self.assertEqual(comparison['requests'], 1)
        self.assertEqual(comparison['lost_successes'], 1)
        self.assertEqual(result['quality_by_mode']['matched_host'][0]['question_id'], 'q1')
        self.assertFalse(result['replacement_expansion_authorized'])
        self.assertIsNone(result['end_to_end_speedup'])

    def test_rejects_mismatched_registered_evaluation_id(self):
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            self.write_evaluation_fixture(output_root, reverse_identities=True)
            with self.assertRaisesRegex(ValueError, 'registered evaluation identity'):
                evaluator.close(output_root)
            self.assertFalse((output_root / 'summary.json').exists())

    def test_all_requests_and_lost_successes_retained(self):
        original = [{'question_id': str(index), 'strict_local_proxy_success': index < 5}
                    for index in range(6)]
        replacement = [{'question_id': str(index), 'strict_local_proxy_success': index < 3}
                       for index in range(6)]
        result = matched_quality(original, replacement)
        self.assertEqual(result['requests'], 6)
        self.assertEqual(result['original_successes'], 5)
        self.assertEqual(result['replacement_successes'], 3)
        self.assertEqual(result['lost_successes'], 2)
        self.assertEqual(result['gained_successes'], 0)
        self.assertFalse(result['observed_count_nonregression'])
        self.assertFalse(result['statistical_noninferiority_established'])

    def test_rejects_empty_duplicate_subset_and_order_change(self):
        rows = [{'question_id': 'first', 'strict_local_proxy_success': True},
                {'question_id': 'second', 'strict_local_proxy_success': False}]
        for original, replacement in (([], []), (rows, [rows[0], rows[0]]),
                                      (rows, rows[:1]), (rows, rows[::-1])):
            with self.subTest(replacement=replacement), self.assertRaises(ValueError):
                matched_quality(original, replacement)


if __name__ == '__main__':
    unittest.main()
