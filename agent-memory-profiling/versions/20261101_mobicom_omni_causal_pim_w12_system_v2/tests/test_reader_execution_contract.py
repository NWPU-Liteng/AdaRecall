from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from probe_reader_execution_contract import compare_outputs, synthetic_cases
from reader_single_pass import read_single_answer
from system_audit import reference_execution_valid
from audit_mobilemem_development import numeric_coverage


class TensorInputs(dict):
    @property
    def input_ids(self):
        return self['input_ids']

    def to(self, device):
        return TensorInputs({name: value.to(device) for name, value in self.items()})


class FakeProcessor:
    eos_token_id = 9

    def __init__(self):
        self.tokenizer = self

    def apply_chat_template(self, messages, **kwargs):
        return ''.join(message['content'] for message in messages)

    def encode(self, text, **kwargs):
        return [ord(character) for character in text]

    def decode(self, tokens, **kwargs):
        return str(tokens)

    def __call__(self, text, **kwargs):
        tokens = torch.tensor([self.encode(text[0])], dtype=torch.long)
        return TensorInputs(input_ids=tokens, attention_mask=torch.ones_like(tokens))


class FakeReader:
    def __init__(self):
        self.config = SimpleNamespace(eos_token_id=9)
        self.model = SimpleNamespace(rope_deltas=None)
        self.calls = []

    def __call__(self, input_ids, past_key_values=None, **kwargs):
        count = int(input_ids.shape[1])
        self.calls.append((count, past_key_values is not None))
        if past_key_values is None:
            cache = SimpleNamespace(tokens=count)
            next_token = 7
        else:
            cache = SimpleNamespace(tokens=past_key_values.tokens + count)
            next_token = 9 if count == 1 else 8
        logits = torch.full((1, 1, 10), -10.0)
        logits[0, 0, next_token] = 10.0
        return SimpleNamespace(logits=logits, past_key_values=cache)


class FakeSnapshot:
    bytes = 16
    prefix_fingerprint = 'immutable-test-prefix'

    def __init__(self, cache, input_ids, rope, **kwargs):
        self.tokens = int(input_ids.shape[1])

    def restore(self, device):
        return SimpleNamespace(tokens=self.tokens), None


class FakeStore:
    capacity_bytes = 1024
    minimum_tokens = 1

    def __init__(self):
        self.snapshot = None
        self.hits = 0
        self.invalidated = False

    def lookup(self, *args, **kwargs):
        if self.snapshot is not None:
            self.hits += 1
        return self.snapshot

    def admit(self, snapshot, **kwargs):
        self.snapshot = snapshot
        return True

    def invalidate(self):
        self.snapshot = None
        self.invalidated = True

    def metrics(self):
        return {'tensor_bytes': 16 if self.snapshot else 0, 'capacity_bytes': self.capacity_bytes}


class ReaderExecutionContractTests(unittest.TestCase):
    def run_fake_request(self, reader, store):
        processor = FakeProcessor()
        atoms = [{'event_id': 'first', 'content': 'history ' * 30},
                 {'event_id': 'second', 'content': 'suffix ' * 20}]
        args = SimpleNamespace(device='cpu', budget_tokens=6000, validate_prefix=True,
                               request_index=0, prefix_audit_every=1, gen_tokens=8)
        with patch('reader_single_pass.PrefixStateSnapshot', FakeSnapshot), \
                patch('reader_single_pass.ReaderOperatorTrace') as trace:
            trace.return_value.as_dict.return_value = {'operators': [], 'calls_by_phase': {}}
            return read_single_answer(reader, processor,
                                      {'effective_timestamp': '2026-10-19', 'question': 'Which record?'},
                                      atoms, args, store, 'fixed-test-contract', 'system',
                                      lambda atom: atom['content'])

    def test_cold_checkpoint_never_serves_partitioned_state(self):
        reader, store = FakeReader(), FakeStore()
        row = self.run_fake_request(reader, store)
        self.assertEqual(row['answer'], '[7, 9]')
        self.assertEqual(len(reader.calls), 3)
        self.assertFalse(reader.calls[0][1])
        self.assertFalse(reader.calls[1][1])
        self.assertTrue(reader.calls[2][1])
        self.assertEqual(reader.calls[2][0], 1)
        self.assertGreater(reader.calls[1][0], reader.calls[0][0])
        self.assertEqual(row['prefix_validation']['status'], 'reference_execution')
        self.assertFalse(row['numeric_audit_scheduled'])
        self.assertTrue(reference_execution_valid(row))
        self.assertTrue(numeric_coverage(row))
        row['prefix_state_reuse']['reused_tokens'] = 1
        self.assertFalse(reference_execution_valid(row))
        self.assertFalse(numeric_coverage(row))

    def test_resumed_candidate_still_recomputes_on_full_sequence_mismatch(self):
        reader, store = FakeReader(), FakeStore()
        self.run_fake_request(reader, store)
        reader.calls.clear()
        row = self.run_fake_request(reader, store)
        self.assertEqual(row['answer'], '[7, 9]')
        self.assertTrue(reader.calls[0][1])
        self.assertEqual(row['prefix_validation']['status'], 'fail')
        self.assertFalse(row['prefix_validation']['full_greedy_token_ids_equal'])
        self.assertTrue(row['prefix_validation']['reference_consumed_as_fallback'])
        self.assertEqual(row['prefix_state_reuse']['status'], 'numeric_contract_full_recompute')
        self.assertEqual(row['prefix_state_reuse']['reused_tokens'], 0)
        self.assertEqual(row['prefix_state_reuse']['prefill_execution'], 'resumed_candidate_reference_fallback')
        self.assertTrue(row['numeric_audit_scheduled'])
        self.assertTrue(store.invalidated)
        self.assertFalse(reference_execution_valid(row))

    def test_sequence_and_stop_are_separate_contracts(self):
        reference = {'token_ids': [11, 12, 13], 'stop_reason': 'eos'}
        same = compare_outputs(reference, reference)
        self.assertTrue(same['full_greedy_token_ids_equal'])
        self.assertIsNone(same['first_divergence'])
        changed = compare_outputs(reference, {'token_ids': [11, 14, 13], 'stop_reason': 'eos'})
        self.assertFalse(changed['full_greedy_token_ids_equal'])
        self.assertEqual(changed['first_divergence'], 1)
        changed = compare_outputs(reference, {'token_ids': [11, 12], 'stop_reason': 'max_new_tokens'})
        self.assertEqual(changed['first_divergence'], 2)
        self.assertFalse(changed['stop_reason_equal'])

    def test_synthetic_probe_selection_is_fixed_and_label_free(self):
        first, second = synthetic_cases(), synthetic_cases()
        self.assertEqual(first, second)
        self.assertEqual([len(case['atoms']) for case in first], [8, 32, 64])
        self.assertTrue(all('gold' not in case and 'labels' not in case for case in first))



if __name__ == '__main__':
    unittest.main()
