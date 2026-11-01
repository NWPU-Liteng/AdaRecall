from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from omni_prefix_opportunities import PrefixOpportunityIndex, multimodal_prefix_symbols


class PrefixOpportunityTests(unittest.TestCase):
    def test_longest_past_only_prefix_and_no_future_knowledge(self):
        index = PrefixOpportunityIndex()
        self.assertEqual(index.observe('first', [1, 2, 3])['matched_prefix_tokens'], 0)
        self.assertEqual(index.observe('second', [1, 2, 9])['matched_prefix_tokens'], 2)
        match = index.observe('third', [1, 2, 3, 4])
        self.assertEqual(match['matched_prefix_tokens'], 3)
        self.assertEqual(match['prior_request_id'], 'first')
        self.assertFalse(match['prefix_state_reuse_authorized'])

    def test_same_image_placeholder_different_batch_is_not_common_state(self):
        first = multimodal_prefix_symbols([1, 7, 7, 2], 7, [2], 'batch-a')
        second = multimodal_prefix_symbols([1, 7, 7, 2], 7, [2], 'batch-b')
        index = PrefixOpportunityIndex()
        index.observe('first', first)
        self.assertEqual(index.observe('second', second)['matched_prefix_tokens'], 1)

    def test_separate_user_index_starts_cold(self):
        first, second = PrefixOpportunityIndex(), PrefixOpportunityIndex()
        first.observe('user-a', [1, 2])
        self.assertEqual(second.observe('user-b', [1, 2])['matched_prefix_tokens'], 0)

    def test_invalid_placeholder_and_duplicate_request_rejected(self):
        with self.assertRaises(ValueError):
            multimodal_prefix_symbols([1, 7], 7, [2], 'batch')
        with self.assertRaises(ValueError):
            multimodal_prefix_symbols([7, 7], 7, [1], 'batch')
        index = PrefixOpportunityIndex()
        index.observe('first', [1])
        with self.assertRaises(ValueError):
            index.observe('first', [1])


if __name__ == '__main__':
    unittest.main()
