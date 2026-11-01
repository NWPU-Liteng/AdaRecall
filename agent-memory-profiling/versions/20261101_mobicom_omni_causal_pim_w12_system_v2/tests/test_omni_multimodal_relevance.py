from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from omni_multimodal_relevance import binary_log_odds, relevance_messages
from omni_provenance_optimizer import build_provenance_hyperedges


class MultimodalRelevanceTests(unittest.TestCase):
    def test_same_binary_log_odds_without_score_fusion(self):
        logits = torch.tensor([[1., 3., 4.], [-2., 5., 1.]])
        self.assertEqual(binary_log_odds(logits, 1, 2), [-1., 4.])
        with self.assertRaises(ValueError):
            binary_log_odds(torch.tensor([[float('nan'), 0.]]), 0, 1)

    def test_original_image_is_attributed_without_answer_or_category(self):
        ref = {'source_session_id': 's', 'source_atom_id': 'a'}
        image = object()
        messages = relevance_messages('question', 'text', [ref], [image])
        self.assertIs(messages[1]['content'][-1]['image'], image)
        self.assertIn('s, turn a', messages[1]['content'][-2]['text'])
        self.assertNotIn('gold', str(messages))
        with self.assertRaises(ValueError):
            relevance_messages('question', 'text', [ref], [])

    def test_image_only_exchange_is_not_silently_dropped(self):
        atom = {'atom_id': 'a', 'session_id': 's', 'event_id': 'e', 'ordinal': 0,
                'role': 'user', 'content_type': 'image', 'content': '',
                'image_refs': [{'normalized': 'p', 'status': 'exact'}]}
        edges = build_provenance_hyperedges(['s'], {'s': [atom]})
        self.assertEqual(edges[0]['atom_ids'], ['a'])


if __name__ == '__main__':
    unittest.main()
