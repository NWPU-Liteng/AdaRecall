from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from audit_omni_recall_width import image_omission_stage, paired_quality, selected_atom_ids


class RecallWidthTests(unittest.TestCase):
    def test_omissions_are_not_all_called_reader_failures(self):
        expected = {'picture'}
        self.assertEqual(image_omission_stage(expected, set(), set(), set(), False),
                         'source_session_pool_image_missing')
        self.assertEqual(image_omission_stage(expected, expected, set(), set(), False),
                         'bounded_candidate_or_embedding_budget_image_missing')
        self.assertEqual(image_omission_stage(expected, expected, expected, set(), False),
                         'selection_budget_image_missing')
        self.assertEqual(image_omission_stage(expected, expected, expected, expected, False),
                         'label_images_retained_answer_wrong')

    def test_paired_partition_and_conditional_interval(self):
        rows = [{'person_id': '0', 'reference_correct': before, 'expanded_correct': after}
                for before, after in ((True, True), (True, False), (False, True), (False, False))]
        summary = paired_quality(rows, bootstrap_samples=100)
        self.assertEqual(summary['paired_accuracy_delta'], 0)
        self.assertEqual(summary['reference_correct'], 2)
        self.assertEqual(summary['expanded_correct'], 2)
        self.assertEqual(summary['exact_mcnemar_two_sided_p'], 1)
        self.assertEqual(sum(summary['partition'].values()), 4)

    def test_unanimous_discordant_outcomes(self):
        rows = [{'person_id': '0', 'reference_correct': False, 'expanded_correct': True}] * 10
        summary = paired_quality(rows, bootstrap_samples=20)
        self.assertEqual(summary['paired_delta_bootstrap_ci95'], [1, 1])
        self.assertEqual(summary['exact_mcnemar_two_sided_p'], 2 / 1024)

    def test_empty_outcomes_or_missing_selection_rejected(self):
        with self.assertRaises(ValueError):
            paired_quality([])
        with self.assertRaises(ValueError):
            selected_atom_ids({'question_id': 'missing'})


if __name__ == '__main__':
    unittest.main()
