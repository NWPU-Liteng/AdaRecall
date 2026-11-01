from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from summarize_omni_protocol_alignment import compare_judgments


class ProtocolAlignmentTests(unittest.TestCase):
    def test_saved_predictions_are_compared_without_regeneration(self):
        previous = [{'question_id': 'q', 'prediction_sha256': 'answer',
                     'parse_valid': True, 'value': 1.0}]
        current = [{'question_id': 'q', 'prediction_sha256': 'answer',
                    'parse_valid': True, 'judge_label': 'WRONG'}]
        result = compare_judgments(previous, current)
        self.assertEqual(result['previous_only_correct'], 1)
        self.assertEqual(result['current_only_correct'], 0)
        self.assertEqual(result['requests'], 1)

    def test_changed_answers_missing_requests_and_duplicate_ids_are_rejected(self):
        previous = [{'question_id': 'q', 'prediction_sha256': 'before',
                     'parse_valid': True, 'value': 1.0}]
        current = [{'question_id': 'q', 'prediction_sha256': 'after',
                    'parse_valid': True, 'judge_label': 'CORRECT'}]
        for bad_old, bad_new in [(previous, current), (previous, []), (previous * 2, current)]:
            with self.assertRaises(ValueError):
                compare_judgments(bad_old, bad_new)

    def test_invalid_judgment_is_not_silently_removed(self):
        previous = [{'question_id': 'q', 'prediction_sha256': 'same',
                     'parse_valid': True, 'value': 1.0}]
        current = [{'question_id': 'q', 'prediction_sha256': 'same',
                    'parse_valid': False, 'judge_label': None}]
        with self.assertRaises(ValueError):
            compare_judgments(previous, current)


if __name__ == '__main__':
    unittest.main()
