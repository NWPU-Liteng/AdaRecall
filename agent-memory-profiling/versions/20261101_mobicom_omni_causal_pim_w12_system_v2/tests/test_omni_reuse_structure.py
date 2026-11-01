from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from diagnose_omni_reuse_structure import analyze_user


class ReuseStructureTests(unittest.TestCase):
    def test_ordered_batch_and_individual_image_opportunities_are_distinct(self):
        smoke = {'rows': [{'question_id': str(index), 'images': [{'sha256': value} for value in hashes]}
                         for index, hashes in enumerate((('a','b'), ('a','c'), ('a','b'), ('b','a')))]}
        result = analyze_user(smoke)
        self.assertEqual(result['ordered_batch_repeat_opportunities'], 1)
        self.assertEqual(result['previously_seen_image_occurrences'], 5)
        self.assertEqual(result['new_batch_requests_with_prior_image_overlap'], 2)
        self.assertEqual(result['rows'][2]['ordered_batch_distance'], 2)
        self.assertEqual(result['rows'][1]['image_reuse_distances'][0]['last_seen_request_distance'], 1)

    def test_users_do_not_share_seen_image_state(self):
        smoke = {'rows': [{'question_id': 'q', 'images': [{'sha256': 'a'}]}]}
        self.assertEqual(analyze_user(smoke)['previously_seen_image_occurrences'], 0)
        self.assertEqual(analyze_user(smoke)['previously_seen_image_occurrences'], 0)

    def test_text_only_request_is_not_multimodal_opportunity(self):
        with self.assertRaises(ValueError):
            analyze_user({'rows': [{'question_id': 'q', 'images': []}]})


if __name__ == '__main__':
    unittest.main()
