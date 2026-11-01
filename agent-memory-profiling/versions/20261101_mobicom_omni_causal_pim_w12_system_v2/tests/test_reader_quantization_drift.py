import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import analyze_reader_quantization_drift as drift


class ReaderQuantizationDriftTests(unittest.TestCase):
    def test_answer_extension_preserves_prompt_boundary(self):
        inputs = {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.ones(1, 3, dtype=torch.long),
            'mm_token_type_ids': torch.zeros(1, 3, dtype=torch.long),
        }
        extended, prompt_tokens = drift.extend_with_answer(inputs, [7, 8])
        self.assertEqual(prompt_tokens, 3)
        self.assertEqual(extended['input_ids'].tolist(), [[1, 2, 3, 7, 8]])
        self.assertEqual(extended['attention_mask'].tolist(), [[1, 1, 1, 1, 1]])
        self.assertEqual(extended['mm_token_type_ids'].tolist(), [[0, 0, 0, 0, 0]])

    def test_teacher_forced_compare_reports_layer_and_logit_drift(self):
        hidden = torch.zeros(1, 6, 4, dtype=torch.bfloat16)
        logits = torch.zeros(1, 3, 8, dtype=torch.bfloat16)
        logits[0, 0, 2] = 4.0
        logits[0, 1, 3] = 4.0
        logits[0, 2, 4] = 4.0
        reference = {
            'prompt_tokens': 3,
            'answer_tokens': [2, 3, 4],
            'hidden_states': (hidden, hidden.clone()),
            'logits': logits.clone(),
            'prediction_positions': torch.arange(2, 5),
        }
        changed_logits = logits.clone()
        changed_logits[0, 1, 0] = 9.0
        changed_hidden = hidden.clone()
        changed_hidden[0, 3:5] = 0.01
        candidate = {
            'prompt_tokens': 3,
            'answer_tokens': [2, 3, 4],
            'hidden_states': (hidden.clone(), changed_hidden),
            'logits': changed_logits,
            'prediction_positions': torch.arange(2, 5),
        }
        result = drift.compare_arms(reference, candidate)
        self.assertGreater(result['layers'][1]['relative_l2_max'], 0)
        self.assertEqual(
            result['first_layer_relative_l2_mean_at_least']['0.01'], 1)
        self.assertEqual(result['logits']['top1_agreement'], 2)
        self.assertEqual(result['logits']['first_top1_flip_position'], 1)
        self.assertLess(result['logits']['logit_cosine_mean'], 1.0)

    def test_reference_tokens_are_label_isolated(self):
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'run.json'
            path.write_text(json.dumps({
                'complete': True, 'evaluation_label_files_read': [],
                'rows': [{'question_id': 'q0', 'answer_token_ids': [1, 2]}],
            }))
            tokens, provenance = drift.load_reference_tokens([path])
            self.assertEqual(tokens, {'q0': [1, 2]})
            self.assertIn(str(path), provenance)


if __name__ == '__main__':
    unittest.main()
