import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from pim_backend.hyperpim_sc import (
    W8A16CostConfig, bank_resident_decoder_plan, grammar_residency,
    packed_bit_sliced_decoder_plan, phase_latency_projection,
    plane_major_layout_plan, w8a16_mixed_precision_overhead)
from pim_backend.hyperpim_timeline import (
    ENERGY_PARAM_KEYS, HyperPimTimelineConfig, energy_model, matrix_cycles,
    progressive_topk_cycles, refresh_elapsed, retrieval_gemv_workload,
    simulate_token, with_weight_bits)
from pim_backend.structured_control import StructuredControlConfig
from pim_replay import build_replay, load_profile


class ExistingHyperPimSnapshotTests(unittest.TestCase):
    def test_existing_functions_match_frozen_v1_snapshot(self):
        expected = json.loads(
            (ROOT / 'tests/fixtures/hyperpim_v1_snapshot.json').read_text())
        expected.pop('schema')
        small = HyperPimTimelineConfig(
            stacks=1, channels_per_stack=1, banks_per_channel=4,
            clock_hz=1_000_000_000, layer_fence_cycles=10)
        sequence = [
            {'name': 'layer0.q', 'inputs': 256, 'outputs': 512, 'layer': 0},
            {'name': 'layer0.o', 'inputs': 512, 'outputs': 256,
             'layer': 0, 'count': 2},
            {'name': 'layer1.mlp', 'inputs': 256, 'outputs': 1024,
             'layer': 1},
        ]
        control = StructuredControlConfig(
            stacks=1, channels_per_stack=1, banks_per_channel=4,
            clock_hz=1_000_000_000, capacity_bytes=1 << 30)
        matrices = [
            {'name': 'q', 'out_features': 64, 'in_features': 256},
            {'name': 'o', 'out_features': 32, 'in_features': 512},
        ]
        actual = {
            'refresh_elapsed': refresh_elapsed(
                150, 7, replace(small, refresh_interval_cycles=100,
                                refresh_block_cycles=20)),
            'matrix_cycles_w8': matrix_cycles(256, 64, small),
            'matrix_cycles_w12': matrix_cycles(
                2560, 64, with_weight_bits(small, 12)),
            'simulate_token': simulate_token(sequence, small),
            'bank_resident_decoder_plan': bank_resident_decoder_plan(
                matrices, control, group_size=128, scale_bytes_per_group=4,
                residual_bits=2),
            'packed_bit_sliced_decoder_plan': packed_bit_sliced_decoder_plan(
                matrices, weight_bits=12, group_size=128, row_bytes=2048,
                capacity_bytes=1 << 30),
            'w8a16_mixed_precision_overhead': w8a16_mixed_precision_overhead(
                matrices, 1.5, control, W8A16CostConfig(
                    group_size=128, conversion_elements_per_bank_cycle=16,
                    group_apply_ops_per_bank_cycle=16,
                    matrix_command_cycles=8,
                    broadcast_bandwidth_bytes_per_second=1_000_000_000,
                    residual_bits=2)),
            'phase_latency_projection': phase_latency_projection(
                measured_seconds=10.0, prefill_seconds=1.25,
                host_forward_ms_per_token=100.0, tokens=5,
                target_ms_per_token=20.0, host_grammar_ms_per_token=3.0),
            'grammar_residency': grammar_residency(1000, 200, 30, 40, control),
        }
        self.assertEqual(actual, expected)


class PlaneMajorLayoutTests(unittest.TestCase):
    def test_layout_reports_bank_plane_rows_and_row_major_comparison(self):
        plan = plane_major_layout_plan(15490, 1024)
        self.assertEqual(plan['rows_per_bank'], 61)
        self.assertEqual(plan['plane_bytes_per_row'], 128)
        self.assertEqual(plan['tile_count'], 1016)
        self.assertEqual(len(plan['banks']), 256)
        self.assertEqual(len(plan['banks'][0]['dram_rows_by_plane']), 12)
        self.assertEqual(plan['banks'][0]['dram_rows_by_plane'], [4] * 12)
        self.assertEqual(
            plan['plane_major_full_read_row_activations'], 12192)
        self.assertEqual(plan['row_major_full_read_row_activations'], 15490)
        self.assertLess(plan['plane_major_over_row_major_activations'], 1)

    def test_layout_rejects_invalid_geometry(self):
        with self.assertRaises(ValueError):
            plane_major_layout_plan(0, 1024)
        with self.assertRaises(ValueError):
            plane_major_layout_plan(10, 1000)
        with self.assertRaises(ValueError):
            plane_major_layout_plan(10, 1024, bits=11)


class ProgressiveCyclesTests(unittest.TestCase):
    def setUp(self):
        self.config = HyperPimTimelineConfig(
            stacks=1, channels_per_stack=1, banks_per_channel=4,
            clock_hz=1_000_000_000, refresh_interval_cycles=100000,
            refresh_block_cycles=100)
        self.layout = plane_major_layout_plan(
            512, 1024, row_bytes=self.config.row_bytes,
            bank_count=self.config.bank_count,
            tile_rows=16)
        self.rounds = [
            {'planes_in_round': 1, 'rows_alive_before': rows}
            for rows in (512, 300, 180, 120, 90, 70, 55, 48, 44, 42, 40, 39)
        ]

    def test_rounds_charge_tiles_reads_macs_bounds_and_threshold(self):
        result = progressive_topk_cycles(
            self.layout, self.rounds, self.config, 100)
        self.assertEqual(len(result['rounds']), 12)
        self.assertEqual(result['totals']['rounds'], 12)
        self.assertEqual(
            result['rounds'][-1]['planes_read_total'], 12)
        self.assertGreater(result['totals']['row_activations_total'], 0)
        self.assertGreater(result['totals']['internal_read_cycles'], 0)
        self.assertGreater(result['totals']['mac_cycles'], 0)
        self.assertGreater(result['totals']['bound_cycles'], 0)
        self.assertEqual(result['totals']['threshold_broadcast_cycles'], 1200)
        self.assertGreater(result['full_gemv']['active_cycles'], 0)
        self.assertEqual(
            result['tile_identity_source'], 'count_bound_no_row_ids')

    def test_explicit_tile_ids_are_used_when_available(self):
        layout = plane_major_layout_plan(
            500, 1024, row_bytes=1800,
            bank_count=self.config.bank_count, tile_rows=16)
        partial_tile = layout['banks'][0]['tiles'][-1]
        self.assertEqual(partial_tile['dram_rows_per_plane'], 1)
        rounds = [
            {
                'planes_in_round': 1,
                'rows_alive_before': partial_tile['row_count'],
                'active_tile_ids': [partial_tile['global_tile_id']],
            }
            for _ in range(12)
        ]
        result = progressive_topk_cycles(
            layout, rounds, self.config, 0)
        self.assertEqual(result['tile_identity_source'], 'provided_active_tile_ids')
        self.assertEqual(result['rounds'][0]['tile_identity_source'],
                         'provided_active_tile_ids')
        self.assertEqual(result['totals']['row_activations_total'], 12)

    def test_survivor_count_cannot_exceed_layout(self):
        rounds = [dict(item) for item in self.rounds]
        rounds[0]['rows_alive_before'] = self.layout['rows'] + 1
        with self.assertRaisesRegex(ValueError, 'exceeds layout rows'):
            progressive_topk_cycles(self.layout, rounds, self.config, 0)

    def test_missing_host_round_trip_fails_closed(self):
        with self.assertRaisesRegex(ValueError, 'host_round_trip_cycles'):
            progressive_topk_cycles(self.layout, self.rounds, self.config, None)


class WorkloadAndEnergyTests(unittest.TestCase):
    def test_retrieval_workload_uses_existing_matrix_model(self):
        result = retrieval_gemv_workload(15490, 1024)
        self.assertEqual(result['outputs'], 15490)
        self.assertEqual(result['inputs'], 1024)
        self.assertEqual(result['macs'], 15490 * 1024)
        self.assertGreater(result['cycles'], 0)
        self.assertEqual(result['sequence_item']['name'], 'dense_retrieval')

    def test_energy_model_never_supplies_missing_parameters(self):
        missing = energy_model({}, {})
        self.assertEqual(missing['status'], 'params_missing')
        self.assertEqual(missing['missing'], list(ENERGY_PARAM_KEYS))

        counts = {key: 1 for key in (
            'row_activations', 'internal_read_bytes', 'io_bytes',
            'bf16_macs', 'bound_updates', 'refresh_cycles')}
        params = {key: 2.0 for key in ENERGY_PARAM_KEYS}
        result = energy_model(counts, params)
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(result['total_pj'], 12.0)
        self.assertEqual(result['total_nj'], 0.012)

    def test_energy_model_rejects_unknown_parameters(self):
        with self.assertRaisesRegex(ValueError, 'unknown energy'):
            energy_model({}, {'invented': 1.0})


class PimReplayTests(unittest.TestCase):
    def test_replay_combines_trace_and_stats(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace_dir = root / 'trace'
            stats_dir = root / 'stats'
            trace_dir.mkdir()
            stats_dir.mkdir()
            trace = {
                'schema': 'omni-serving-workload-trace-v1',
                'person_id': '1',
                'question_id': '1_q_test',
                'dense_retrieval': {'matrix_shape': [512, 1024]},
                'reader_linear_operators': {
                    'contract_parse': [{
                        'name': 'layer.linear', 'shape': [32, 128],
                        'calls': 1,
                    }],
                },
                'stages': {
                    'contract_parse': {'decode_tokens': 2},
                    'witness_construct': {'decode_tokens': 0},
                    'answer_prefill': {'decode_tokens': 0},
                    'answer_decode': {'decode_tokens': 3},
                },
            }
            (trace_dir / '1' / '1_q_test.json').parent.mkdir()
            (trace_dir / '1' / '1_q_test.json').write_text(
                json.dumps(trace))
            rounds = [
                {'planes_in_round': 1, 'rows_alive_before': 512}
                for _ in range(12)
            ]
            row = {
                'person_id': '1',
                'question_id': '1_q_test',
            }
            (stats_dir / 'user_1_shard_0.json').write_text(json.dumps({
                'complete': True,
                'rows': [row],
                'progressive_lm_head_stats': {
                    'enabled': True,
                    'vocab_rows': 256,
                    'valid_vocab_size': 250,
                    'bits': 12,
                    'group_size': 128,
                    'plane_schedule': [1] * 12,
                    'build': {'plane_bytes': 256 * 12 * 128},
                    'phases': {
                        'contract_parse': {
                            'checked_tokens': 2, 'average_rounds': 12.0,
                            'progressive_calls': 2, 'forced_tokens': 0,
                            'weight_bytes_read': 2 * 128 * 12,
                        },
                        'witness_construct': {
                            'checked_tokens': 0, 'average_rounds': 12.0,
                            'progressive_calls': 0, 'forced_tokens': 0,
                            'weight_bytes_read': 0,
                        },
                        'answer': {
                            'checked_tokens': 5, 'average_rounds': 12.0,
                            'progressive_calls': 3, 'forced_tokens': 2,
                            'weight_bytes_read': 3 * 128 * 12,
                        },
                    },
                },
                'progressive_retrieval_stats': {
                    'enabled': True,
                    'mode': 'shadow',
                    'bits': 12,
                    'group_size': 128,
                    'demands': [{
                        'question_id': '1_q_test',
                        'demand_id': 'd0',
                        'rounds': rounds,
                    }],
                },
            }))
            profile = {
                'schema': 'pim-profile-v2',
                'source_path': 'inline',
                'source_sha256': 'a' * 64,
                'host_round_trip_cycles': 100,
                'timing_profile': {
                    'stacks': 1,
                    'channels_per_stack': 1,
                    'banks_per_channel': 4,
                    'clock_hz': 1_000_000_000,
                },
                'energy_params': {key: None for key in ENERGY_PARAM_KEYS},
            }
            result = build_replay(trace_dir, stats_dir, profile)
            self.assertEqual(result['question_count'], 1)
            phases = result['questions'][0]['phases']
            self.assertGreater(
                phases['contract_parse']['full_precision_gemv_cycles'], 0)
            self.assertGreater(
                phases['contract_parse'][
                    'grammar_mask_plus_progressive_cycles'], 0)
            self.assertEqual(
                phases['answer_decode']['checked_tokens'], 5)
            self.assertEqual(
                phases['answer_decode']['progressive_calls'], 3)
            self.assertEqual(
                phases['answer_decode']['forced_tokens'], 2)
            self.assertEqual(
                phases['answer_decode']['progressive_totals']['rounds'], 12)
            self.assertGreater(
                phases['dense_retrieval']['retrieval_cycles'], 0)
            self.assertEqual(
                set(result['energy_by_estimate']),
                {'full_precision_gemv', 'grammar_mask_only',
                 'grammar_mask_plus_progressive', 'retrieval'})
            self.assertTrue(all(
                energy['status'] == 'params_missing'
                for energy in result['energy_by_estimate'].values()))

    def test_profile_requires_filled_host_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'profile.yaml'
            path.write_text(yaml.safe_dump({
                'schema': 'pim-profile-v2',
                'host_round_trip_cycles': None,
                'timing_profile': {},
                'energy_params': {},
            }))
            with self.assertRaisesRegex(ValueError, 'host_round_trip_cycles'):
                load_profile(path)

    def test_replay_rejects_unattributable_shard_lm_stats(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace_dir = root / 'trace'
            stats_dir = root / 'stats'
            trace_dir.mkdir()
            stats_dir.mkdir()
            for question_id in ('1_q_a', '1_q_b'):
                (trace_dir / question_id).with_suffix('.json').write_text(
                    json.dumps({
                        'schema': 'omni-serving-workload-trace-v1',
                        'person_id': '1',
                        'question_id': question_id,
                        'reader_linear_operators': {},
                        'stages': {},
                    }))
            (stats_dir / 'user_1_shard_0.json').write_text(json.dumps({
                'complete': True,
                'progressive_lm_head_stats': {'enabled': True},
                'rows': [
                    {'person_id': '1', 'question_id': '1_q_a'},
                    {'person_id': '1', 'question_id': '1_q_b'},
                ],
            }))
            profile = {
                'schema': 'pim-profile-v2',
                'source_path': 'inline',
                'source_sha256': 'a' * 64,
                'host_round_trip_cycles': 100,
                'timing_profile': {},
                'energy_params': {key: None for key in ENERGY_PARAM_KEYS},
            }
            with self.assertRaisesRegex(ValueError, 'cannot be attributed'):
                build_replay(trace_dir, stats_dir, profile)


if __name__ == '__main__':
    unittest.main()
