import sys
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT))

from profile_main_pim import quantize_symmetric_int8, topk_overlap
from pim_backend.lpddr_bank_pim import BankPimConfig, BankPimModel, bank_int8_topk


class MainPimProfileTests(unittest.TestCase):
    def test_global_symmetric_int8_preserves_scale_and_bounds(self):
        values = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
        quantized, scale = quantize_symmetric_int8(values)
        self.assertEqual(quantized.dtype, np.int8)
        self.assertAlmostEqual(scale, 2.0 / 127.0)
        self.assertEqual(int(quantized.min()), -127)
        self.assertEqual(int(quantized.max()), 127)

    def test_int8_bank_topk_is_compared_to_fp32_reference(self):
        generator = np.random.default_rng(17)
        matrix = generator.normal(size=(257, 96)).astype(np.float32)
        queries = generator.normal(size=(4, 96)).astype(np.float32)
        quantized, _ = quantize_symmetric_int8(matrix)
        overlaps = []
        for query in queries:
            reference = np.argsort(-(matrix @ query))[:32]
            query_int8, _ = quantize_symmetric_int8(query)
            candidate, _ = bank_int8_topk(quantized, query_int8, banks=32, top_k=32)
            overlaps.append(topk_overlap(reference, candidate))
        self.assertGreaterEqual(min(overlaps), 0.9)

    def test_host_model_accounts_for_fp32_bytes_and_pim_for_int8(self):
        model = BankPimModel(BankPimConfig())
        host = model.resident_score_scan(129, 64, top_k=16,
                                         route='host', element_bytes=4)
        pim = model.resident_score_scan(129, 64, top_k=16,
                                        route='pim', element_bytes=1)
        self.assertEqual(host['logical_read_bytes'], 129 * 64 * 4)
        self.assertEqual(pim['logical_read_bytes'], 129 * 64)
        self.assertTrue(pim['admission']['admitted'])
        self.assertEqual(pim['route'], 'pim')
        self.assertGreater(host['cycles'], 0)
        self.assertGreater(pim['cycles'], 0)


if __name__ == '__main__':
    unittest.main()
