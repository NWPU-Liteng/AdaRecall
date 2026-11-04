from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

from compare_v2_arms import compare, exact_mcnemar_p


def judge_row(question_type, correct):
    return {
        'question_type': question_type,
        'parse_valid': True,
        'judge_label': 'CORRECT' if correct else 'WRONG',
    }


class CompareV2ArmsTests(unittest.TestCase):
    def test_paired_counts_and_type_deltas(self):
        baseline = {
            'q1': judge_row('single_hop', True),
            'q2': judge_row('single_hop', True),
            'q3': judge_row('temporal_reasoning', False),
        }
        candidate = {
            'q1': judge_row('single_hop', True),
            'q2': judge_row('single_hop', False),
            'q3': judge_row('temporal_reasoning', True),
        }
        report = compare(baseline, candidate)
        self.assertEqual(report['delta_correct'], 0)
        self.assertEqual(report['paired']['both_correct'], 1)
        self.assertEqual(report['paired']['baseline_only_correct'], 1)
        self.assertEqual(report['paired']['candidate_only_correct'], 1)
        self.assertEqual(report['by_type']['single_hop']['delta_correct'], -1)
        self.assertEqual(report['by_type']['temporal_reasoning']['delta_correct'], 1)

    def test_exact_mcnemar_uses_discordant_pairs(self):
        self.assertEqual(exact_mcnemar_p(0, 0), 1.0)
        self.assertAlmostEqual(exact_mcnemar_p(8, 5), 0.5810546875)


if __name__ == '__main__':
    unittest.main()
