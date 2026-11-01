from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from freeze_closure_cohort import freeze


class ClosureCohortTests(unittest.TestCase):
    def test_no_answer_category_or_content_changes_selection(self):
        rows = [{'person_id': '0', 'question_id': str(index), 'split': 'development'}
                for index in range(100)]
        selection = freeze(rows, ['0'], 16, 'frozen')
        changed = [{**row, 'question': 'changed', 'question_type': 'changed', 'answer': 'changed'} for row in rows]
        self.assertEqual(selection, freeze(changed, ['0'], 16, 'frozen'))
        self.assertEqual(len(selection['0']), 16)
        self.assertEqual(selection['0'], sorted(selection['0'], key=int))

    def test_sealed_users_are_not_eligible(self):
        with self.assertRaises(ValueError):
            freeze([{'person_id': '6', 'question_id': 'x', 'split': 'holdout'}], ['6'], 1, 'frozen')


if __name__ == '__main__':
    unittest.main()
