from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pim_backend.matched_decode import MatchedDecodeDispatcher
from tests import test_functional_decode


class MatchedDecodeTests(unittest.TestCase):
    def test_prefill_unchanged_selected_host_flag_restored(self):
        reader, residency = test_functional_decode.FunctionalDecodeTests().fixture()
        observed = []
        reader[0].register_forward_pre_hook(lambda module, arguments: observed.append(
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction))
        original = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        try:
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
            with MatchedDecodeDispatcher(reader, residency, mode='shadow') as dispatcher:
                reader(torch.ones(2, 13).bfloat16())
                self.assertEqual(observed, [True])
                observed.clear()
                original_forward = dispatcher.originals[0][1]

                def capture(activation):
                    observed.append(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)
                    return original_forward(activation)

                wrapped = dispatcher._wrapper(reader[0], capture, next(iter(residency.pool)), '0')
                with dispatcher.decode():
                    wrapped(torch.ones(1, 1, 13).bfloat16())
                self.assertEqual(observed, [False])
                self.assertTrue(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)
        finally:
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = original

    def test_precision_setting_restored_on_exception(self):
        reader, residency = test_functional_decode.FunctionalDecodeTests().fixture()
        original = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction

        def fail(activation):
            raise RuntimeError('host failure')

        with MatchedDecodeDispatcher(reader, residency, mode='shadow') as dispatcher:
            wrapped = dispatcher._wrapper(reader[0], fail, next(iter(residency.pool)), '0')
            with self.assertRaises(RuntimeError), dispatcher.decode():
                wrapped(torch.ones(1, 1, 13).bfloat16())
        self.assertEqual(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction, original)


if __name__ == '__main__':
    unittest.main()
