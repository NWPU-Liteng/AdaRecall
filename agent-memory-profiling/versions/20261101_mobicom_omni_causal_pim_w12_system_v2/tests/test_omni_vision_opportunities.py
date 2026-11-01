from copy import deepcopy
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from audit_omni_vision_opportunities import audit_stream, image_operands


def fixture(batches):
    rows, observations = [], []
    for index, batch in enumerate(batches):
        rows.append({'question_id': str(index), 'images': [{} for _ in batch],
            'service_seconds': 10., 'service_stage_ledger': {
                'nested_seconds': {'vision_in_prefill_seconds': 4.}},
            'embedding_producers': [{'producer_sha256': 'producer'}],
            'hypergraph_evidence_plan': {'accepted_atom_ids': ['atom']}})
        observations.append({'question_id': str(index), 'images': [
            {'operand_sha256': name, 'raw_sha256': name, 'pooled_payload_bytes': 8}
            for name in batch]})
    return rows, observations


class VisionOpportunityTests(unittest.TestCase):
    def test_past_only_ordered_whole_batch_and_partial_bound(self):
        rows, observations = fixture([['a', 'b'], ['a', 'c'], ['a', 'b'], ['b', 'a']])
        result = audit_stream(rows, observations, 64)
        self.assertEqual(result['bounded_whole_batch_hit_opportunities'], 1)
        self.assertEqual(result['previous_request_image_operand_occurrences'], 5)
        self.assertEqual(result['zero_cost_whole_batch_time_removal_bound_seconds'], 4.)
        self.assertEqual(result['zero_cost_independent_image_loose_time_removal_bound_seconds'], 12.)
        self.assertEqual(result['rows'][0]['previous_image_operands'], [False, False])

    def test_no_within_request_or_cross_user_reuse(self):
        rows, observations = fixture([['a', 'a']])
        for _ in range(2):
            self.assertEqual(audit_stream(rows, observations, 64)['previous_request_image_operand_occurrences'], 0)

    def test_eviction_and_oversize(self):
        rows, observations = fixture([['a'], ['b'], ['a']])
        result = audit_stream(rows, observations, 8)
        self.assertEqual(result['ordered_batch_repeats'], 1)
        self.assertEqual(result['bounded_whole_batch_hit_opportunities'], 0)
        self.assertEqual(result['evictions'], 2)
        self.assertEqual(audit_stream(rows, observations, 7)['oversize_rejections'], 3)

    def test_denominator_and_invalid_timing(self):
        rows, observations = fixture([['a'], ['b']])
        with self.assertRaises(ValueError):
            audit_stream(rows, observations[:1], 64)
        observations[1]['question_id'] = '0'
        with self.assertRaises(ValueError):
            audit_stream(rows, observations, 64)
        rows, observations = fixture([['a']])
        rows[0]['service_seconds'] = float('nan')
        with self.assertRaises(ValueError):
            audit_stream(rows, observations, 64)

    def test_grid_split_identity_and_layout_validation(self):
        inputs = {'pixel_values': torch.arange(32, dtype=torch.float32).view(8, 4),
                  'image_grid_thw': torch.tensor([[1, 2, 2], [1, 2, 2]])}
        images = [{'sha256': 'raw', 'source_session_id': 'session', 'source_atom_id': str(index)}
                  for index in range(2)]
        operands = image_operands(inputs, images, 'execution', 2, 4)
        self.assertNotEqual(operands[0]['operand_sha256'], operands[1]['operand_sha256'])
        self.assertEqual(operands[0]['pooled_payload_bytes'], 8)
        single = {name: value[:4] if name == 'pixel_values' else value[:1] for name, value in inputs.items()}
        self.assertEqual(image_operands(single, images[:1], 'execution', 2, 4)[0]['operand_sha256'],
                         operands[0]['operand_sha256'])
        changed = deepcopy(inputs)
        changed['image_grid_thw'][0, 2] = 4
        with self.assertRaises(ValueError):
            image_operands(changed, images, 'execution', 2, 4)


if __name__ == '__main__':
    unittest.main()
