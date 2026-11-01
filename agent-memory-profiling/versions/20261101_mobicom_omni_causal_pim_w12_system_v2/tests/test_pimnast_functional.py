from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pim_backend.pimnast_functional import PimnastInt8Layout, TorchPimnastInt8Layout, PimnastReaderDispatch
from pim_backend.pimnast_functional import frozen_route


class PimnastFunctionalTests(unittest.TestCase):
    def test_frozen_route_strips_future_counts_and_rejects_invalid_profiles(self):
        from copy import deepcopy
        projection = {'complete': True, 'operand_bits': 8, 'per_user': [{
            'run': 'artifacts/frozen/reference_user_0.json', 'declared_pim_pool_bytes': 1024,
            'group_residency_policy': {'future_call_count_used_by_objective': False},
            'matrices': [{'name': 'linear', 'shape': '16x16', 'model_route': 'pim',
                          'candidate_resident_bytes': 512, 'calls': 9000}]}]}
        candidates, capacity = frozen_route(projection, '0')
        self.assertEqual(capacity, 1024)
        self.assertNotIn('calls', candidates['linear'])
        projection['per_user'][0]['matrices'][0]['calls'] = 1
        self.assertEqual(frozen_route(projection, '0'), (candidates, capacity))
        for user in ('1', '3', 'holdout'):
            with self.assertRaises(ValueError):
                frozen_route(projection, user)
        invalid = deepcopy(projection)
        invalid['per_user'][0]['group_residency_policy']['future_call_count_used_by_objective'] = True
        with self.assertRaises(ValueError):
            frozen_route(invalid, '0')
        invalid = deepcopy(projection)
        invalid['per_user'] *= 2
        with self.assertRaises(ValueError):
            frozen_route(invalid, '0')

    def test_full_outputs_match_independent_int64_reference(self):
        generator = np.random.default_rng(20261023)
        for rows, columns in ((16, 16), (32, 256), (2048, 32), (4096, 16), (2560, 48)):
            with self.subTest(rows=rows, columns=columns):
                weights = generator.integers(-128, 128, (rows, columns), dtype=np.int8)
                vector = generator.integers(-128, 128, columns, dtype=np.int8)
                layout = PimnastInt8Layout(weights, allow_row_padding=True)
                np.testing.assert_array_equal(layout.gemv(vector),
                                              weights.astype(np.int64) @ vector.astype(np.int64))
                self.assertEqual(layout.gemv(vector).dtype, np.int32)

    def test_every_weight_has_unique_address_and_fixed_bank_owned_output(self):
        weights = np.arange(4096 * 32, dtype=np.int64).reshape(4096, 32).astype(np.int8)
        layout = PimnastInt8Layout(weights)
        addresses = set()
        for row in range(len(weights)):
            for column in range(weights.shape[1]):
                address = layout.address(row, column)
                self.assertEqual(address['bank'], row // 16 % 128)
                self.assertEqual(address['channel'], (row // 16 % 128) // 16)
                self.assertEqual(layout.storage[address['bank'], address['bank_offset']],
                                 weights[row, column])
                physical = address['physical_byte_address']
                self.assertEqual(physical // 256 % 128, address['bank'])
                addresses.add(physical)
        self.assertEqual(len(addresses), weights.size)
        self.assertEqual(layout.address(0, 1)['bank_offset'], 16)
        self.assertEqual(layout.address(2048, 0)['bank_offset'], 512)

    def test_padding_capacity_and_snapshot_are_explicit(self):
        weights = np.ones((32, 16), dtype=np.int8)
        layout = PimnastInt8Layout(weights, allow_row_padding=True)
        weights[:] = 5
        np.testing.assert_array_equal(layout.gemv(np.ones(16, dtype=np.int8)), np.full(32, 16))
        inventory = layout.inventory()
        self.assertEqual(inventory['active_banks'], 2)
        self.assertEqual(inventory['resident_layout_bytes'], 128 * 2048)
        self.assertEqual(inventory['row_padding_bytes'], (2048 - 32) * 16)
        self.assertEqual(inventory['dram_alignment_padding_bytes'], 128 * (2048 - 256))
        self.assertIsNone(inventory['timing_ns'])
        with self.assertRaises(ValueError):
            layout.storage[0, 0] = 0

    def test_rejects_unsupported_contracts(self):
        for weights in (np.zeros((0, 16), np.int8), np.zeros((17, 16), np.int8),
                        np.zeros((16, 17), np.int8), np.zeros((16, 16), np.uint8),
                        np.zeros((16, 131072), np.int8)):
            with self.assertRaises(ValueError):
                PimnastInt8Layout(weights, allow_row_padding=True)
        with self.assertRaises(ValueError):
            PimnastInt8Layout(np.zeros((16, 16), np.int8))
        layout = PimnastInt8Layout(np.zeros((16, 16), np.int8), allow_row_padding=True)
        for vector in (np.zeros((1, 16), np.int8), np.zeros(15, np.int8), np.zeros(16, np.float32)):
            with self.assertRaises(ValueError):
                layout.gemv(vector)
        for coordinate in ((-1, 0), (16, 0), (0, 16), (1.5, 0)):
            with self.assertRaises(ValueError):
                layout.address(*coordinate)

    def test_signed_extremes_and_zero_vector(self):
        weights = np.full((2048, 16), -128, dtype=np.int8)
        layout = PimnastInt8Layout(weights)
        np.testing.assert_array_equal(layout.gemv(np.full(16, -128, dtype=np.int8)),
                                      np.full(2048, 16 * 128 ** 2))
        np.testing.assert_array_equal(layout.gemv(np.zeros(16, dtype=np.int8)), np.zeros(2048))

    def test_torch_bank_layout_execution_matches_numpy_and_int64(self):
        import torch
        generator = np.random.default_rng(19)
        for rows, columns in ((32, 16), (2560, 48), (4096, 32)):
            weights = generator.integers(-128, 128, (rows, columns), dtype=np.int8)
            vector = generator.integers(-128, 128, columns, dtype=np.int8)
            reference = PimnastInt8Layout(weights, allow_row_padding=True)
            layout = TorchPimnastInt8Layout(torch.from_numpy(weights),
                                            workspace_bytes=128 * 16 * columns * 8)
            np.testing.assert_array_equal(layout.storage.numpy(), reference.storage)
            np.testing.assert_array_equal(layout.gemv(torch.from_numpy(vector)).numpy(),
                                          reference.gemv(vector))
            self.assertEqual(layout.round_chunk, 1)
        with self.assertRaises(ValueError):
            TorchPimnastInt8Layout(torch.zeros((16, 16), dtype=torch.int8), workspace_bytes=1)

    def test_dispatch_preserves_host_prefill_and_restores_methods(self):
        import torch
        from pim_backend.functional_decode import Int8LinearContract

        class TinyReader(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = Int8LinearContract(torch.ones((16, 16)))

            def forward(self, input_ids, past_key_values=None):
                return self.linear(input_ids)

        reader = TinyReader().eval()
        candidates = {'linear': {'model_route': 'pim', 'candidate_resident_bytes': 128 * 2048}}
        vector = torch.arange(16).float().reshape(1, -1)
        expected = reader(vector)
        with PimnastReaderDispatch(reader, candidates) as dispatcher:
            torch.testing.assert_close(reader(vector), expected, rtol=0, atol=0)
            self.assertEqual(dispatcher.calls['linear'], 0)
            torch.testing.assert_close(reader(vector, past_key_values=object()), expected, rtol=0, atol=0)
            self.assertEqual(dispatcher.calls['linear'], 1)
            self.assertEqual(dispatcher.host_fallback_calls, 1)
            self.assertEqual(dispatcher.summary()['unobserved_selected_modules'], [])
        self.assertNotIn('integer_product', reader.linear.__dict__)
        torch.testing.assert_close(reader(vector), expected, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            with PimnastReaderDispatch(reader, {'linear': dict(candidates['linear'], candidate_resident_bytes=1)}):
                pass
        self.assertNotIn('integer_product', reader.linear.__dict__)


if __name__ == '__main__':
    unittest.main()
