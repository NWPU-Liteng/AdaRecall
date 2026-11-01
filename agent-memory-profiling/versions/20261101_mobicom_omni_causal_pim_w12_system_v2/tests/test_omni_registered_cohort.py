from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_omni_registered_cohort import freeze_cohort


class RegisteredCohortTests(unittest.TestCase):
    def test_serving_order_and_user_identity_frozen_without_labels(self):
        rows = [{'person_id': user, 'split': split, 'question_id': question,
                 'visibility_protocol': 'history_end'} for user, split, question in (
                     ('0', 'development', 'first'), ('9', 'holdout', 'sealed'),
                     ('1', 'development', 'other'), ('0', 'development', 'second'))]
        result = freeze_cohort(rows, ['0', '1'], 1)
        self.assertEqual(result, [{'person_id': '0', 'question_ids': ['first']},
                                  {'person_id': '1', 'question_ids': ['other']}])
        for users, limit in [(['0'], 0), (['0', '0'], 1), (['9'], 1), (['1'], 2)]:
            with self.assertRaises(ValueError):
                freeze_cohort(rows, users, limit)


if __name__ == '__main__':
    unittest.main()
