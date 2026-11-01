import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pim_backend.decode_gemv import bank_tiled_gemv, gemv_cost, project_trace, select_exact_capacity
from pim_backend.lpddr_bank_pim import BankPimConfig


class DecodeGemvTests(unittest.TestCase):
    def test_function_matches_full_reference_with_ragged_tiles_and_bias(self):
        generator = np.random.default_rng(20261020)
        for shape, banks, tile in [((37, 517), 32, 256), ((3, 7), 32, 4), ((1, 1), 1, 1)]:
            weights = generator.normal(size=shape).astype(np.float32)
            activation = generator.normal(size=shape[1]).astype(np.float32)
            bias = generator.normal(size=shape[0]).astype(np.float32)
            actual = bank_tiled_gemv(weights, activation, banks=banks, input_tile=tile, bias=bias)
            np.testing.assert_allclose(actual, weights @ activation + bias, rtol=1e-4, atol=2e-5)

    def test_invalid_inputs_fail_closed(self):
        for weights, activation in [(np.ones((2, 3)), np.ones(2)),
                                    (np.ones((2, 3)), np.array([1., 2., np.nan]))]:
            with self.assertRaises(ValueError):
                bank_tiled_gemv(weights, activation)
        with self.assertRaises(ValueError):
            bank_tiled_gemv(np.ones((2, 3)), np.ones(3), input_tile=0)
        with self.assertRaises(ValueError):
            bank_tiled_gemv(np.ones((2, 3)), np.ones(3), operand_dtype='bf16x')

    def test_bf16_operand_rounding_and_fp32_accumulation_are_explicit(self):
        import torch
        generator = np.random.default_rng(20261021)
        weights = generator.normal(size=(35, 517)).astype(np.float32)
        activation = generator.normal(size=517).astype(np.float32)
        rounded_weights = torch.from_numpy(weights).bfloat16().float().numpy()
        rounded_activation = torch.from_numpy(activation).bfloat16().float().numpy()
        expected = rounded_weights @ rounded_activation
        actual = bank_tiled_gemv(weights, activation, operand_dtype='bf16')
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=3e-5)

    def operator(self, name, inputs=2048, outputs=2048, count=5):
        return {"name": name, "phase": "decode", "rows": 1, "in_features": inputs,
                "out_features": outputs, "element_bytes": 2,
                "weight_bytes": inputs * outputs * 2, "calls": count}

    def test_placement_is_capacity_bounded_and_not_output_length_oracle(self):
        cfg = BankPimConfig(pim_capacity_bytes=12 * 1024**2)
        short = project_trace([self.operator("a"), self.operator("b")], cfg)
        long = project_trace([self.operator("a", count=50), self.operator("b", count=50)], cfg)
        self.assertEqual(short["resident_matrices"], 1)
        self.assertLessEqual(short["resident_bytes"], cfg.pim_capacity_bytes)
        self.assertEqual([item["route"] for item in short["per_operator"]],
                         [item["route"] for item in long["per_operator"]])
        self.assertGreater(short["cold_hybrid_linear_cycles"], short["warm_hybrid_linear_cycles"])
        self.assertIsNone(short["end_to_end_speedup"])

    def test_buffer_capacity_and_unprofitable_cases_fallback(self):
        tiny = project_trace([self.operator("tiny", inputs=1, outputs=1)], BankPimConfig())
        self.assertEqual(tiny["resident_matrices"], 0)
        small_buffer = project_trace([self.operator("a")], BankPimConfig(local_buffer_bytes_per_bank=32))
        self.assertEqual(small_buffer["resident_matrices"], 0)
        prefill = self.operator("prefill")
        prefill.update(phase="prefill", rows=128)
        self.assertEqual(project_trace([prefill], BankPimConfig())["executed_decode_linear_calls"], 0)

    def test_transfer_commands_and_full_copy_are_charged(self):
        cost = gemv_cost(2048, 2049, 2, BankPimConfig())
        self.assertGreater(cost["stage_cycles"], 0)
        self.assertGreater(cost["pim_command_fence_cycles"], 0)
        self.assertGreater(cost["pim_external_bytes"], 0)
        self.assertGreaterEqual(cost["resident_bytes"], 2048 * 2049 * 2)
        self.assertEqual(cost["pim_external_bytes"], 2048 * 2 * 32 + 2049 * 4)

    def test_trace_shape_mismatch_rejected(self):
        operator = self.operator("bad")
        operator["weight_bytes"] += 1
        with self.assertRaises(ValueError):
            project_trace([operator], BankPimConfig())

    def test_capacity_placement_is_exact_not_gain_density_greedy(self):
        candidates = {
            'large': {'resident_bytes': 6, 'capacity_feasible': True, 'buffer_feasible': True,
                      'host_cycles': 12, 'pim_cycles': 0},
            'small_a': {'resident_bytes': 5, 'capacity_feasible': True, 'buffer_feasible': True,
                        'host_cycles': 9, 'pim_cycles': 0},
            'small_b': {'resident_bytes': 5, 'capacity_feasible': True, 'buffer_feasible': True,
                        'host_cycles': 9, 'pim_cycles': 0},
        }
        selected, used, gain = select_exact_capacity(candidates, 10)
        self.assertEqual(selected, {'small_a', 'small_b'})
        self.assertEqual(used, 10)
        self.assertEqual(gain, 18)

    def test_exact_dp_agrees_with_exhaustive_small_instances(self):
        generator = np.random.default_rng(20261020)
        for _ in range(30):
            candidates = {str(index): {'resident_bytes': int(generator.integers(1, 15)),
                'host_cycles': int(generator.integers(0, 30)), 'pim_cycles': 5,
                'capacity_feasible': True, 'buffer_feasible': True} for index in range(8)}
            budget = int(generator.integers(0, 40))
            selected, used, gain = select_exact_capacity(candidates, budget)
            oracle = 0
            names = list(candidates)
            for mask in range(1 << len(names)):
                subset = [names[index] for index in range(len(names)) if mask & (1 << index)]
                if sum(candidates[name]['resident_bytes'] for name in subset) <= budget:
                    oracle = max(oracle, sum(candidates[name]['host_cycles'] - candidates[name]['pim_cycles']
                                             for name in subset))
            self.assertEqual(gain, oracle)
            self.assertLessEqual(used, budget)

    def test_exact_dp_does_not_allocate_unbounded_tables(self):
        candidates = {'small': {'resident_bytes': 1, 'host_cycles': 2, 'pim_cycles': 1,
                                'capacity_feasible': True, 'buffer_feasible': True}}
        with self.assertRaises(ValueError):
            select_exact_capacity(candidates, 2_000_000)


if __name__ == "__main__":
    unittest.main()
