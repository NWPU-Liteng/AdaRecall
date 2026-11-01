import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import evaluate_reader_quantization_pair as evaluation


class ReaderQuantizationEvaluationTests(unittest.TestCase):
    def test_identical_pairs_pass_noninferiority(self):
        rows = [{'person_id': str(user), 'reference_correct': correct,
                 'candidate_correct': correct}
                for user in range(3) for correct in (False, True)]
        result = evaluation.paired_summary(rows, 5.0, 200, 7)
        self.assertEqual(result['paired_delta_pp'], 0.0)
        self.assertTrue(result['noninferiority_passed'])
        self.assertEqual(result['exact_mcnemar_two_sided_p'], 1.0)

    def test_candidate_regression_fails_margin(self):
        rows = [{'person_id': '0', 'reference_correct': True,
                 'candidate_correct': False} for _ in range(20)]
        rows += [{'person_id': '0', 'reference_correct': True,
                  'candidate_correct': True} for _ in range(80)]
        result = evaluation.paired_summary(rows, 5.0, 200, 7)
        self.assertEqual(result['paired_delta_pp'], -20.0)
        self.assertFalse(result['noninferiority_passed'])

    def test_rate_retains_entire_denominator(self):
        rows = [{'flag': True}, {'flag': False}, {'flag': False}]
        self.assertAlmostEqual(evaluation.rate(rows, 'flag'), 1 / 3)

    def test_bootstrap_scope_is_not_hardcoded_to_development_users(self):
        rows = [{'person_id': str(user), 'reference_correct': True,
                 'candidate_correct': True} for user in range(4)]
        result = evaluation.paired_summary(rows, 5.0, 200, 7)
        self.assertNotIn('three development users', result['bootstrap']['scope'])
        self.assertIn('observed users', result['bootstrap']['scope'])

    def test_candidate_arm_label_declares_residual_and_scope(self):
        self.assertEqual(
            evaluation.candidate_arm_label('w8a16', 'none', 2), 'w8a16_r2')
        self.assertEqual(
            evaluation.candidate_arm_label('w8a16', 'output_sensitive', 0),
            'w8a16_output_sensitive')


if __name__ == '__main__':
    unittest.main()
