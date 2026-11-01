from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pim_backend.functional_decode import (
    DeferredInt8Validation, Int8LinearContract, deferred_int8_finite_checks, quantize_int8_rows)


class TinyReader(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first = Int8LinearContract(torch.ones(16, 16))
        self.last = Int8LinearContract(torch.eye(16))

    def forward(self, input_ids, past_key_values=None):
        return self.last(self.first(input_ids))


class DeferredValidationTests(unittest.TestCase):
    def test_all_checks_remain_and_arithmetic_is_identical(self):
        reader = TinyReader().eval()
        activation = torch.arange(16).float().reshape(1, -1)
        expected = reader(activation)
        with DeferredInt8Validation(reader) as dispatch:
            self.assertTrue(torch.equal(reader(activation), expected))
            self.assertEqual(dispatch.calls, 0)
            self.assertTrue(torch.equal(reader(activation, past_key_values=object()), expected))
            self.assertEqual(dispatch.summary()['finite_predicates_checked'], 4)
            self.assertEqual(dispatch.summary()['host_confirmations'], 1)
        self.assertNotIn('forward', reader.__dict__)
        self.assertTrue(torch.equal(reader(activation), expected))

    def test_nonfinite_cannot_escape_forward_boundary(self):
        reader = TinyReader().eval()
        activation = torch.full((1, 16), float('nan'))
        with DeferredInt8Validation(reader):
            with self.assertRaisesRegex(ValueError, 'before Reader forward completion'):
                reader(activation, past_key_values=object())
            torch.testing.assert_close(reader(torch.zeros(1, 16), past_key_values=object()), torch.zeros(1, 16))
        with self.assertRaises(ValueError):
            quantize_int8_rows(activation)

    def test_overflow_is_rejected_after_conversion(self):
        reader = TinyReader().eval()
        reader.last.scales.fill_(float('inf'))
        with DeferredInt8Validation(reader):
            with self.assertRaises(ValueError):
                reader(torch.ones(1, 16), past_key_values=object())

    def test_error_resets_context_and_nested_scopes_fail(self):
        with self.assertRaisesRegex(RuntimeError, 'original failure'):
            with deferred_int8_finite_checks():
                raise RuntimeError('original failure')
        with deferred_int8_finite_checks() as stats:
            quantize_int8_rows(torch.ones(1, 16))
            with self.assertRaises(ValueError):
                with deferred_int8_finite_checks():
                    pass
        self.assertEqual(stats['checks'], 1)
        with self.assertRaises(ValueError):
            quantize_int8_rows(torch.full((1, 16), float('nan')))


if __name__ == '__main__':
    unittest.main()
