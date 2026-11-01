import sys
from pathlib import Path
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pim_backend.lm_head_functional import LmHeadBankDispatcher


class TinyReader(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lm_head = torch.nn.Linear(256, 512, bias=False, dtype=torch.bfloat16)

    def forward(self, values):
        return self.lm_head(values)


class LmHeadFunctionalTests(unittest.TestCase):
    def test_shadow_preserves_host_output_and_argmax(self):
        torch.manual_seed(0)
        reader = TinyReader().eval()
        values = torch.randn(1, 1, 256, dtype=torch.bfloat16)
        host = reader(values)
        dispatcher = LmHeadBankDispatcher(reader, mode='shadow', banks=32, input_tile=256)
        with dispatcher:
            with dispatcher.decode():
                observed = reader(values)
        self.assertTrue(torch.equal(observed, host))
        summary = dispatcher.summary()
        self.assertEqual(summary['calls'], 1)
        self.assertTrue(summary['records'][0]['argmax_equal'])
        self.assertTrue(summary['records'][0]['bitwise_equal'])
        self.assertFalse(summary['physical_pim_executed'])

    def test_replace_uses_bank_output_for_single_row_only(self):
        torch.manual_seed(1)
        reader = TinyReader().eval()
        values = torch.randn(1, 1, 256, dtype=torch.bfloat16)
        dispatcher = LmHeadBankDispatcher(reader, mode='replace', banks=32, input_tile=256)
        with dispatcher:
            with dispatcher.decode():
                output = reader(values)
                reader(torch.randn(2, 1, 256, dtype=torch.bfloat16))
        self.assertEqual(tuple(output.shape), (1, 1, 512))
        self.assertEqual(dispatcher.calls, 1)
        self.assertEqual(dispatcher.replaced_calls, 1)
        self.assertEqual(dispatcher.skipped_non_single_row_calls, 1)


if __name__ == '__main__':
    unittest.main()
