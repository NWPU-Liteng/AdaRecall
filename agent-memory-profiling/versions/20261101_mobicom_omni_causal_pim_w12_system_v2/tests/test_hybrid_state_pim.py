import json
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pim_backend.hybrid_state import (HybridStateLayout, HybridStatePimPlanner,
                                      bank_online_softmax_attention,
                                      bank_gated_delta_state_step,
                                      bank_recurrent_state_step)
from pim_backend.lpddr_bank_pim import BankPimConfig


class HybridStatePimTests(unittest.TestCase):
    def test_bank_reference_matches_whole_state_reference(self):
        generator = np.random.default_rng(7)
        state = generator.normal(size=(8, 4, 3)).astype(np.float32)
        query = generator.normal(size=(8, 4)).astype(np.float32)
        key = generator.normal(size=(8, 4)).astype(np.float32)
        value = generator.normal(size=(8, 3)).astype(np.float32)
        decay = generator.uniform(0.1, 0.95, size=(8,)).astype(np.float32)
        updated, output = bank_recurrent_state_step(
            state, query, key, value, decay, banks=4)
        expected_state = decay[:, None, None] * state
        expected_state += key[:, :, None] * value[:, None, :]
        expected_output = np.einsum('hk,hkv->hv', query, expected_state)
        np.testing.assert_allclose(updated, expected_state, rtol=0, atol=0)
        np.testing.assert_allclose(output, expected_output, rtol=0, atol=0)

    def test_qwen_gated_delta_bank_step_matches_direct_reference(self):
        generator = np.random.default_rng(23)
        state = generator.normal(size=(8, 5, 4)).astype(np.float32)
        query = generator.normal(size=(8, 5)).astype(np.float32)
        key = generator.normal(size=(8, 5)).astype(np.float32)
        value = generator.normal(size=(8, 4)).astype(np.float32)
        gate_log_decay = generator.uniform(-1.0, -0.01, size=(8,)).astype(np.float32)
        beta = generator.uniform(0.0, 1.0, size=(8,)).astype(np.float32)

        updated, output = bank_gated_delta_state_step(
            state, query, key, value, gate_log_decay, beta, banks=4)
        q = query / np.sqrt(np.sum(query * query, axis=-1, keepdims=True) + 1e-6)
        k = key / np.sqrt(np.sum(key * key, axis=-1, keepdims=True) + 1e-6)
        decayed = state * np.exp(gate_log_decay[:, None, None])
        delta = (value - np.sum(decayed * k[:, :, None], axis=1)) * beta[:, None]
        expected_state = decayed + k[:, :, None] * delta[:, None, :]
        expected_output = np.sum(
            expected_state * (q / np.sqrt(5))[:, :, None], axis=1)
        np.testing.assert_allclose(updated, expected_state, rtol=0, atol=0)
        np.testing.assert_allclose(output, expected_output, rtol=2e-6, atol=1e-7)

    def test_native_inventory_layout_preserves_mixed_state_scope(self):
        inventory = {
            'full_attention_bytes': 100,
            'layers': [
                {
                    'kind': 'linear_attention',
                    'tensor_shapes': [[1, 4, 2], [1, 2, 3, 3]],
                    'dtypes': ['torch.bfloat16', 'torch.float32'],
                },
                {
                    'kind': 'full_attention',
                    'tensor_shapes': [[1, 2, 8, 3], [1, 2, 8, 3]],
                    'dtypes': ['torch.bfloat16'],
                },
            ],
        }
        layout = HybridStateLayout.from_cache_inventory(inventory)
        self.assertEqual(layout.linear_layers, 1)
        self.assertEqual(layout.recurrent_state_bytes, 2 * 3 * 3 * 4)
        self.assertEqual(layout.conv_state_bytes, 4 * 2 * 2)
        self.assertEqual(layout.full_attention_bytes, 100)
        self.assertEqual(layout.full_attention_layers, 1)
        self.assertEqual(layout.full_attention_context_tokens, 8)
        self.assertEqual(layout.full_attention_kv_heads, 2)
        self.assertEqual(layout.total_native_state_bytes,
                         layout.linear_state_bytes + 100)

    def test_bank_online_attention_matches_whole_attention_reference(self):
        generator = np.random.default_rng(19)
        query = generator.normal(size=(8, 16)).astype(np.float32)
        keys = generator.normal(size=(11, 2, 16)).astype(np.float32)
        values = generator.normal(size=(11, 2, 16)).astype(np.float32)
        output = bank_online_softmax_attention(query, keys, values, banks=4)

        head_map = np.arange(8) // 4
        scores = np.einsum("hd,nhd->hn", query, keys[:, head_map]) / np.sqrt(16)
        scores -= scores.max(axis=1, keepdims=True)
        probabilities = np.exp(scores)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        expected = np.einsum("hn,nhd->hd", probabilities, values[:, head_map])
        np.testing.assert_allclose(output, expected, rtol=1e-5, atol=1e-6)

    def test_multi_agent_capacity_admission_is_explicit(self):
        layout = HybridStateLayout(
            linear_layers=24,
            recurrent_state_bytes=32 * 128 * 128 * 4 * 24,
            conv_state_bytes=8192 * 4 * 2 * 24,
            full_attention_bytes=0,
        )
        planner = HybridStatePimPlanner(BankPimConfig(), reserved_bytes=64 * 1024**2)
        admission = planner.admission(layout, active_agents=32)
        self.assertGreater(admission['resident_agents'], 0)
        self.assertTrue(admission['admitted'])
        overflow = planner.admission(layout, active_agents=64)
        self.assertFalse(overflow['admitted'])
        self.assertIn('active_agent_capacity', overflow['reasons'])
        self.assertTrue(admission['full_attention_bytes_not_in_pim'] == 0)

    def test_plan_charges_host_boundary_and_keeps_e2e_unknown(self):
        layout = HybridStateLayout(
            linear_layers=2,
            recurrent_state_bytes=2 * 4 * 8 * 8 * 4,
            conv_state_bytes=2 * 8 * 2 * 2,
        )
        result = HybridStatePimPlanner(BankPimConfig()).plan(layout, active_agents=1)
        self.assertGreater(result['host_offloaded_step']['external_bytes'],
                           result['pim_step']['external_bytes'])
        self.assertGreater(result['pim_step']['bank_internal_bytes'], 0)
        self.assertTrue(result['full_attention_untouched'] if 'full_attention_untouched' in result else True)
        self.assertIsNone(result['end_to_end_speedup'])
        self.assertFalse(result['physical_pim_executed'])

    def test_unified_capacity_gate_counts_full_attention_kv(self):
        layout = HybridStateLayout(
            linear_layers=24,
            recurrent_state_bytes=51_380_000,
            conv_state_bytes=524_512,
            full_attention_bytes=786_399_232,
            full_attention_layers=8,
            full_attention_context_tokens=23_999,
            full_attention_query_heads=16,
            full_attention_kv_heads=4,
            full_attention_head_dim=256,
        )
        planner = HybridStatePimPlanner(
            BankPimConfig(), reserved_bytes=64 * 1024**2)
        linear = planner.admission(layout, active_agents=32, state_scope="linear")
        unified = planner.admission(layout, active_agents=32, state_scope="unified")
        self.assertEqual(linear["resident_agents"], 40)
        self.assertTrue(linear["admitted"])
        self.assertEqual(unified["resident_agents"], 2)
        self.assertFalse(unified["admitted"])
        self.assertEqual(unified["requested_state_bytes"],
                         32 * layout.full_hybrid_state_bytes)
        plan = planner.unified_plan(layout, active_agents=32)
        self.assertIsNone(
            plan["conditional_modeled_step_speedup_for_admitted_resident_state"])
        self.assertIsNone(plan["end_to_end_speedup"])
        self.assertFalse(plan["physical_pim_executed"])

    def test_unified_step_cost_includes_all_full_attention_layers(self):
        layout = HybridStateLayout(
            linear_layers=1,
            recurrent_state_bytes=4 * 8 * 8 * 4,
            conv_state_bytes=8 * 2 * 2,
            full_attention_bytes=8 * 2 * 4 * 16 * 2,
            full_attention_layers=2,
            full_attention_context_tokens=8,
            full_attention_query_heads=4,
            full_attention_kv_heads=2,
            full_attention_head_dim=16,
        )
        planner = HybridStatePimPlanner(BankPimConfig())
        result = planner.unified_step_cost(layout, route="pim")
        self.assertEqual(result["state_bytes"], layout.total_native_state_bytes)
        self.assertTrue(result["full_attention_managed_by_pim"])
        self.assertEqual(result["phases"][1]["name"], "full_attention_kv")
        self.assertGreater(result["phases"][1]["bank_internal_bytes"], 0)


if __name__ == '__main__':
    unittest.main()
