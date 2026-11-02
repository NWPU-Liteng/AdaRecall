import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

from stage_profiler import (STAGE_NAMES, TRACE_SCHEMA, StageProfiler,
                            allowed_token_count, summarize_grammar_allowed)


class FakeModel:
    def __init__(self):
        self.bank_calls = 0
        self.prefill_fallback_calls = 0
        self.decode_fallback_calls = 0
        self.in_features = 8
        self.out_features = 4


class FakeDispatcher:
    def __init__(self):
        self.modules = {'model.language_model.layers.0.linear': FakeModel()}
        self.layout_instance_sha256 = 'a' * 64

    def forward(self, calls=1):
        module = self.modules['model.language_model.layers.0.linear']
        module.bank_calls += calls

    def counter_snapshot(self):
        modules = list(self.modules.values())
        return {
            'selected_modules': len(modules),
            'unique_layouts': len(modules),
            'bank_calls': sum(item.bank_calls for item in modules),
            'decode_fallback_calls': sum(item.decode_fallback_calls for item in modules),
            'prefill_fallback_calls': sum(item.prefill_fallback_calls for item in modules),
            'fallback_rows': 0,
            'bank_external_bytes': 0,
            'fallback_external_bytes': 0,
            'layout_instance_sha256': self.layout_instance_sha256,
        }


def reader_profile(*, prefill_ms=1.0, decode_ms=2.0, decode_tokens=2,
                   forced=1, allowed_mean=3.0):
    return {
        'prefill_wall_ms': prefill_ms,
        'decode_wall_ms': decode_ms,
        'prefill_tokens': 10,
        'n_images': 0,
        'decode_tokens': decode_tokens,
        'grammar_forced_tokens': forced,
        'grammar_allowed_mean': allowed_mean,
    }


class StageProfilerTests(unittest.TestCase):
    def test_fake_model_stage_times_and_w12_counters_accumulate(self):
        model = FakeDispatcher()
        times = iter([0.0, 0.010, 2.0, 2.004, 4.0, 4.007])
        synchronizations = []
        profiler = StageProfiler(
            w12_dispatch=model, clock=lambda: next(times),
            synchronize=lambda: synchronizations.append(True))

        with profiler.stage('query_embed'):
            model.forward(1)
        with profiler.stage('query_embed'):
            model.forward(2)
        with profiler.stage('query_embed'):
            model.forward(3)

        profile = profiler.as_dict(request_wall_ms=30.0)
        stage = profile['stages']['query_embed']
        self.assertAlmostEqual(stage['wall_ms'], 21.0)
        self.assertEqual(stage['w12']['bank_calls'], 6)
        self.assertEqual(stage['linear_operators'], [{
            'name': 'model.language_model.layers.0.linear',
            'shape': [4, 8],
            'calls': 6,
        }])
        self.assertEqual(len(synchronizations), 6)
        self.assertAlmostEqual(profile['coverage_fraction'], 0.7)

    def test_repeated_reader_stage_aggregates_weighted_grammar_mean(self):
        times = iter([0.0, 0.005, 1.0, 1.004])
        profiler = StageProfiler(
            clock=lambda: next(times), synchronize=lambda: None)

        with profiler.stage('witness_construct') as stage:
            stage.add_reader_profile(reader_profile(
                decode_tokens=2, forced=1, allowed_mean=2.0))
        with profiler.stage('witness_construct') as stage:
            stage.add_reader_profile(reader_profile(
                decode_tokens=1, forced=0, allowed_mean=6.0))

        profile = profiler.as_dict(request_wall_ms=10.0)['stages']['witness_construct']
        self.assertEqual(profile['reader_calls'], 2)
        self.assertEqual(profile['prefill_tokens'], 20)
        self.assertEqual(profile['decode_tokens'], 3)
        self.assertEqual(profile['grammar_forced_tokens'], 1)
        self.assertAlmostEqual(profile['grammar_allowed_mean'], 10.0 / 3.0)
        self.assertAlmostEqual(profile['prefill_wall_ms'], 2.0)
        self.assertAlmostEqual(profile['decode_wall_ms'], 4.0)

    def test_singleton_grammar_bitmask_is_forced(self):
        mask = np.zeros((1, 2), dtype=np.uint32)
        mask[0, 0] = 1 << 7
        allowed = allowed_token_count(mask, vocab_size=64)
        self.assertEqual(allowed, 1)
        summary = summarize_grammar_allowed([allowed, 9])
        self.assertEqual(summary['grammar_forced_tokens'], 1)
        self.assertEqual(summary['grammar_allowed_mean'], 5.0)

    def test_grammar_bitmask_excludes_padding_rows(self):
        mask = np.zeros((1, 2), dtype=np.uint32)
        mask[0] = (1 << 3) | (1 << 31)
        mask[0, 1] = 1 << 8
        self.assertEqual(allowed_token_count(mask, vocab_size=64), 3)
        self.assertEqual(allowed_token_count(
            mask, vocab_size=64, valid_vocab_size=40), 2)

    def test_workload_trace_contains_reader_linears_and_dense_shape(self):
        model = FakeDispatcher()
        times = iter([0.0, 0.004, 1.0, 1.003, 2.0, 2.001])
        profiler = StageProfiler(
            w12_dispatch=model, clock=lambda: next(times),
            synchronize=lambda: None)
        with profiler.stage('contract_parse') as stage:
            stage.add_reader_profile(reader_profile())
            model.forward(1)
        with profiler.stage('dense_retrieval') as stage:
            stage.set_metric('matrix_shape', [17, 1024])

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'trace' / '0' / '0_q_1.json'
            payload = profiler.write_workload_trace(
                path, person_id='0', question_id='0_q_1',
                question_sha256='b' * 64, request_wall_ms=10.0,
                dense_retrieval_shape=[17, 1024], image_count=2)
            written = json.loads(path.read_text())

        self.assertEqual(payload, written)
        self.assertEqual(written['schema'], TRACE_SCHEMA)
        self.assertEqual(written['dense_retrieval']['matrix_shape'], [17, 1024])
        self.assertEqual(written['reader_linear_operators']['contract_parse'][0]['calls'], 1)
        self.assertEqual(set(written['stages']), {'contract_parse', 'dense_retrieval'})

    def test_unknown_stage_and_nested_stages_are_rejected(self):
        profiler = StageProfiler(synchronize=lambda: None)
        with self.assertRaises(ValueError):
            with profiler.stage('unknown'):
                pass
        with self.assertRaises(RuntimeError):
            with profiler.stage('join'):
                with profiler.stage('query_embed'):
                    pass

    def test_stage_list_matches_plan(self):
        self.assertEqual(STAGE_NAMES, (
            'contract_parse', 'query_embed', 'dense_retrieval', 'rerank',
            'witness_construct', 'join', 'materialize', 'answer_prefill',
            'answer_decode'))


if __name__ == '__main__':
    unittest.main()
