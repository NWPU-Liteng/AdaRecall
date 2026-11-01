from pathlib import Path
import sys
import tempfile
import unittest
import subprocess

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from project_omni_geniepim import (ADAPTATIONS, adapt_config, capacity_selection,
                                  integer_output_cost, residual_budget)
from project_omni_geniepim import repository_commit, source_tree_id


class GeniePimTests(unittest.TestCase):
    def test_integer_primitive_charges_int32_host_result_without_double_pim_return(self):
        cost = {'host_gemm_time_ns': 84512 / 90., 'host_compute_time_ns': 1.,
                'host_memory_bytes': 84512, 'pim_host_read_output_bytes': 8192}
        host, contract = integer_output_cost(cost, 32, 2560, 8)
        self.assertAlmostEqual(host, 84608 / 90.)
        self.assertEqual(contract['integer_host_output_bytes'], 128)
        self.assertEqual(contract['extra_pim_output_return_charge_bytes'], 0)
        self.assertIsNone(contract['activation_quantization_cost_ns'])
        self.assertIsNone(contract['fp32_scale_bias_and_bf16_conversion_cost_ns'])
        self.assertEqual(integer_output_cost(cost, 32, 2560, 16),
                         (cost['host_gemm_time_ns'], None))
        for field, invalid in (('host_memory_bytes', 1), ('pim_host_read_output_bytes', 32)):
            with self.assertRaises(ValueError):
                integer_output_cost(dict(cost, **{field: invalid}), 32, 2560, 8)

    def test_capacity_objective_ignores_future_counts_and_groups_siblings(self):
        candidates = {
            'layer.mlp.gate_proj': dict(resident_bytes=4, host_per_call_ns=12., pim_per_call_ns=2., calls=1, shape='2x2'),
            'layer.mlp.up_proj': dict(resident_bytes=4, host_per_call_ns=12., pim_per_call_ns=2., calls=1, shape='2x2'),
            'other': dict(resident_bytes=5, host_per_call_ns=10., pim_per_call_ns=5., calls=1000, shape='3x2')}
        individual, used, _ = capacity_selection(candidates, 5, group_gate_up=False)
        self.assertEqual(used, 4)
        self.assertEqual(len(individual), 1)
        grouped, used, policy = capacity_selection(candidates, 5, group_gate_up=True)
        self.assertEqual(grouped, {'other'})
        self.assertEqual(used, 5)
        self.assertFalse(policy['future_call_count_used_by_objective'])
        candidates['other']['calls'] = 1
        self.assertEqual(grouped, capacity_selection(candidates, 5, group_gate_up=True)[0])
        with self.assertRaises(ValueError):
            capacity_selection({'layer.mlp.gate_proj': candidates['layer.mlp.gate_proj']}, 8, group_gate_up=True)

    def test_gate_up_joint_placement_must_report_independent_baseline_regret(self):
        candidates = {
            'layer.mlp.gate_proj': dict(resident_bytes=4, host_per_call_ns=20., pim_per_call_ns=3., calls=5, shape='2x2'),
            'layer.mlp.up_proj': dict(resident_bytes=4, host_per_call_ns=20., pim_per_call_ns=3., calls=5, shape='2x2'),
        }
        independent, _, _ = capacity_selection(candidates, 4, group_gate_up=False)
        jointly_grouped, _, _ = capacity_selection(candidates, 4, group_gate_up=True)
        self.assertEqual(independent, {'layer.mlp.gate_proj'})
        self.assertEqual(jointly_grouped, set())

    def test_residual_threshold_keeps_unknown_work(self):
        self.assertAlmostEqual(residual_budget(100., 50., 1.3), 35 / .3)
        self.assertIsNone(residual_budget(100., 90., 1.3))
        for values in ((-1., 0., 1.3), (1., 0., 1.), (float('nan'), 0., 1.3)):
            with self.assertRaises(ValueError):
                residual_budget(*values)

    def test_adaptation_is_explicit_and_rejects_missing_source_parameter(self):
        source = '\n'.join(key + '=old' for key in ADAPTATIONS)
        generated = adapt_config(source)
        self.assertIn('OPERAND_SIZE_LIST=16', generated)
        self.assertIn('ASSUME_FREE_CROSS_SIMD_REDUCTION_LIST=False', generated)
        self.assertIn('OPERAND_SIZE_LIST=8', adapt_config(source, 8))
        self.assertIn('ACT_OPERAND_SIZE_LIST=8', adapt_config(source, 8))
        self.assertIn('PIM_TILE_SHAPE_DEGREE_LIST=16', adapt_config(source, 8))
        self.assertIn('PIM_TILE_ORDER_DEGREE_LIST=1', adapt_config(source, 8))
        with self.assertRaises(ValueError):
            adapt_config(source, 4)
        with self.assertRaises(ValueError):
            adapt_config('OPERAND_SIZE_LIST=8')

    def test_repository_commit_is_optional_for_source_snapshots(self):
        self.assertIsNone(repository_commit('/path/that/does/not/exist'))

    def test_source_tree_identity_matches_git_and_ignores_generated_files(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / 'nested').mkdir()
            (root / 'nested/file').write_bytes(b'data\n')
            (root / 'nested.ext').write_bytes(b'root\n')
            subprocess.run(['git', 'init', '-q', name], check=True)
            subprocess.run(['git', '-C', name, 'add', '.'], check=True)
            expected = subprocess.check_output(['git', '-C', name, 'write-tree'], text=True).strip()
            self.assertEqual(source_tree_id(root), expected)
            (root / '__pycache__').mkdir()
            (root / '__pycache__/cache').write_bytes(b'ignored')
            self.assertEqual(source_tree_id(root), expected)
            (root / 'nested/file').write_bytes(b'changed\n')
            self.assertNotEqual(source_tree_id(root), expected)
