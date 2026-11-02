"""Fine-grained serving stage timing and workload traces."""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
import json
import math
from pathlib import Path
import time

import numpy as np


SCHEMA = 'omni-serving-stage-profile-v1'
TRACE_SCHEMA = 'omni-serving-workload-trace-v1'
STAGE_NAMES = (
    'contract_parse',
    'query_embed',
    'dense_retrieval',
    'rerank',
    'witness_construct',
    'join',
    'materialize',
    'answer_prefill',
    'answer_decode',
)
READER_STAGE_NAMES = (
    'contract_parse',
    'witness_construct',
    'answer_prefill',
    'answer_decode',
)
_W12_COUNTER_FIELDS = (
    'bank_calls',
    'decode_fallback_calls',
    'prefill_fallback_calls',
    'fallback_rows',
    'bank_external_bytes',
    'fallback_external_bytes',
)


def _default_synchronize():
    import torch

    torch.cuda.synchronize()


def allowed_token_count(mask, vocab_size: int, valid_vocab_size: int | None = None) -> int:
    """Count allowed xgrammar mask bits within the valid tokenizer vocabulary."""
    if not isinstance(vocab_size, int) or isinstance(vocab_size, bool) or vocab_size < 1:
        raise ValueError('vocab_size must be a positive integer')
    valid = vocab_size if valid_vocab_size is None else valid_vocab_size
    if (not isinstance(valid, int) or isinstance(valid, bool)
            or valid < 0 or valid > vocab_size):
        raise ValueError('valid_vocab_size must be between zero and vocab_size')
    if hasattr(mask, 'detach'):
        values = mask.detach().cpu().numpy()
    else:
        values = np.asarray(mask)
    if values.ndim != 2 or values.shape[0] != 1:
        raise ValueError('one xgrammar token-bitmask row required')
    words = np.ascontiguousarray(values[0], dtype=np.uint32)
    bits = np.unpackbits(words.view(np.uint8), bitorder='little')
    if bits.size < vocab_size:
        raise ValueError('xgrammar token bitmask is shorter than the vocabulary')
    return int(bits[:valid].sum())


def summarize_grammar_allowed(allowed_counts) -> dict:
    """Summarize per-decode-step grammar-allowed set sizes."""
    values = list(allowed_counts)
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0
           for value in values):
        raise ValueError('grammar-allowed counts must be non-negative integers')
    return {
        'grammar_forced_tokens': sum(value == 1 for value in values),
        'grammar_allowed_mean': (sum(values) / len(values) if values else None),
    }


class StageSpan:
    """Mutable measurements attached to one stage invocation."""

    def __init__(self, name: str):
        self.name = name
        self.metrics = {}
        self.reader_calls = 0
        self.prefill_wall_ms = 0.0
        self.decode_wall_ms = 0.0
        self.prefill_tokens = 0
        self.n_images = 0
        self.decode_tokens = 0
        self.grammar_forced_tokens = 0
        self._grammar_allowed_sum = 0.0
        self._grammar_allowed_steps = 0

    def set_metric(self, name: str, value) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError('metric name must be a non-empty string')
        if name in self.metrics:
            raise ValueError(f'duplicate stage metric: {name}')
        self.metrics[name] = value

    def set_metrics(self, values: Mapping) -> None:
        for name, value in values.items():
            self.set_metric(name, value)

    def add_reader_profile(self, profile: Mapping) -> None:
        required = (
            'prefill_wall_ms',
            'decode_wall_ms',
            'prefill_tokens',
            'n_images',
            'decode_tokens',
            'grammar_forced_tokens',
            'grammar_allowed_mean',
        )
        missing = [name for name in required if name not in profile]
        if missing:
            raise ValueError(f'reader stage profile is missing: {missing}')
        decode_tokens = int(profile['decode_tokens'])
        forced_tokens = int(profile['grammar_forced_tokens'])
        if decode_tokens < 0 or not 0 <= forced_tokens <= decode_tokens:
            raise ValueError('invalid Reader decode or grammar-forced token count')
        self.reader_calls += 1
        self.prefill_wall_ms += float(profile['prefill_wall_ms'])
        self.decode_wall_ms += float(profile['decode_wall_ms'])
        self.prefill_tokens += int(profile['prefill_tokens'])
        self.n_images += int(profile['n_images'])
        self.decode_tokens += decode_tokens
        self.grammar_forced_tokens += forced_tokens
        allowed_mean = profile['grammar_allowed_mean']
        if allowed_mean is not None:
            if decode_tokens < 1:
                raise ValueError('grammar_allowed_mean requires decode tokens')
            self._grammar_allowed_sum += float(allowed_mean) * decode_tokens
            self._grammar_allowed_steps += decode_tokens

    def as_reader_profile(self) -> dict:
        allowed_mean = (self._grammar_allowed_sum / self._grammar_allowed_steps
                        if self._grammar_allowed_steps else None)
        return {
            'reader_calls': self.reader_calls,
            'prefill_wall_ms': self.prefill_wall_ms,
            'decode_wall_ms': self.decode_wall_ms,
            'prefill_tokens': self.prefill_tokens,
            'n_images': self.n_images,
            'decode_tokens': self.decode_tokens,
            'grammar_forced_tokens': self.grammar_forced_tokens,
            'grammar_allowed_mean': allowed_mean,
        }


class StageProfiler:
    """Synchronize, time, and counter-diff named serving stages."""

    def __init__(self, *, w12_dispatch=None, synchronize=None, clock=None):
        self.w12_dispatch = w12_dispatch
        self.synchronize = synchronize or _default_synchronize
        self.clock = clock or time.perf_counter
        self._entries = {}
        self._active = 0

    def _counter_snapshot(self):
        return (self.w12_dispatch.counter_snapshot()
                if self.w12_dispatch is not None else None)

    def _counter_delta(self, before, after):
        if before is None or after is None:
            return None
        from pim_backend.w12_functional_dispatch import W12FunctionalDispatch

        return W12FunctionalDispatch.counter_delta(before, after)

    def _operator_snapshot(self):
        if self.w12_dispatch is None:
            return None
        result = {}
        for name, module in self.w12_dispatch.modules.items():
            calls = (module.bank_calls + module.prefill_fallback_calls
                     + module.decode_fallback_calls)
            result[name] = {
                'name': name,
                'shape': [int(module.out_features), int(module.in_features)],
                'calls': int(calls),
            }
        return result

    @staticmethod
    def _operator_delta(before, after):
        if before is None or after is None:
            return None
        if set(before) != set(after):
            raise RuntimeError('W12 operator set changed across a stage')
        result = []
        for name in sorted(after):
            if before[name]['shape'] != after[name]['shape']:
                raise RuntimeError(f'W12 operator shape changed across a stage: {name}')
            calls = after[name]['calls'] - before[name]['calls']
            if calls < 0:
                raise RuntimeError(f'W12 operator counter regressed: {name}')
            if calls:
                result.append({'name': name, 'shape': after[name]['shape'],
                               'calls': calls})
        return result

    @staticmethod
    def _merge_w12(current, incoming):
        if current is None:
            return incoming
        if incoming is None:
            raise RuntimeError('W12 counters disappeared across repeated stages')
        for name in ('selected_modules', 'unique_layouts', 'layout_instance_sha256'):
            if current[name] != incoming[name]:
                raise RuntimeError(f'W12 stage identity changed: {name}')
        result = dict(current)
        for name in _W12_COUNTER_FIELDS:
            result[name] += incoming[name]
        return result

    @staticmethod
    def _merge_operators(current, incoming):
        if current is None:
            return incoming
        if incoming is None:
            raise RuntimeError('W12 operator counters disappeared across repeated stages')
        merged = {item['name']: dict(item) for item in current}
        for item in incoming:
            existing = merged.get(item['name'])
            if existing is None:
                merged[item['name']] = dict(item)
            else:
                if existing['shape'] != item['shape']:
                    raise RuntimeError('W12 operator shape changed across repeated stages')
                existing['calls'] += item['calls']
        return [merged[name] for name in sorted(merged)]

    def _merge_entry(self, span: StageSpan, wall_ms: float, w12, operators):
        entry = self._entries.get(span.name)
        reader_profile = (span.as_reader_profile() if span.reader_calls else None)
        if entry is None:
            entry = {'name': span.name, 'wall_ms': wall_ms, 'w12': w12}
            entry.update(span.metrics)
            if reader_profile is not None:
                entry.update(reader_profile)
            if operators is not None:
                entry['linear_operators'] = operators
            self._entries[span.name] = entry
            return
        entry['wall_ms'] += wall_ms
        entry['w12'] = self._merge_w12(entry.get('w12'), w12)
        for name, value in span.metrics.items():
            if name in entry and entry[name] != value:
                raise RuntimeError(f'conflicting repeated stage metric: {name}')
            entry[name] = value
        if reader_profile is not None:
            previous = None
            if entry.get('reader_calls'):
                previous = StageSpan(span.name)
                previous.reader_calls = int(entry['reader_calls'])
                previous.prefill_wall_ms = float(entry['prefill_wall_ms'])
                previous.decode_wall_ms = float(entry['decode_wall_ms'])
                previous.prefill_tokens = int(entry['prefill_tokens'])
                previous.n_images = int(entry['n_images'])
                previous.decode_tokens = int(entry['decode_tokens'])
                previous.grammar_forced_tokens = int(entry['grammar_forced_tokens'])
                allowed_mean = entry.get('grammar_allowed_mean')
                if allowed_mean is not None:
                    previous._grammar_allowed_sum = float(allowed_mean) * previous.decode_tokens
                    previous._grammar_allowed_steps = previous.decode_tokens
            if previous is None:
                entry.update(reader_profile)
            else:
                previous.reader_calls += reader_profile['reader_calls']
                previous.prefill_wall_ms += reader_profile['prefill_wall_ms']
                previous.decode_wall_ms += reader_profile['decode_wall_ms']
                previous.prefill_tokens += reader_profile['prefill_tokens']
                previous.n_images += reader_profile['n_images']
                previous.decode_tokens += reader_profile['decode_tokens']
                previous.grammar_forced_tokens += reader_profile['grammar_forced_tokens']
                if reader_profile['grammar_allowed_mean'] is not None:
                    previous._grammar_allowed_sum += (
                        reader_profile['grammar_allowed_mean'] * reader_profile['decode_tokens'])
                    previous._grammar_allowed_steps += reader_profile['decode_tokens']
                entry.update(previous.as_reader_profile())
        if operators is not None:
            entry['linear_operators'] = self._merge_operators(
                entry.get('linear_operators'), operators)

    @contextmanager
    def stage(self, name: str):
        if name not in STAGE_NAMES:
            raise ValueError(f'unknown stage name: {name}')
        if self._active:
            raise RuntimeError('nested stage profiling is not supported')
        self.synchronize()
        before_counters = self._counter_snapshot()
        before_operators = self._operator_snapshot()
        started = self.clock()
        span = StageSpan(name)
        self._active += 1
        try:
            yield span
        finally:
            self.synchronize()
            wall_ms = (self.clock() - started) * 1000.0
            after_counters = self._counter_snapshot()
            after_operators = self._operator_snapshot()
            w12 = self._counter_delta(before_counters, after_counters)
            operators = self._operator_delta(before_operators, after_operators)
            self._merge_entry(span, wall_ms, w12, operators)
            self._active -= 1

    def as_dict(self, *, request_wall_ms: float | None = None) -> dict:
        stages = {name: dict(value) for name, value in self._entries.items()}
        stage_wall_ms = math.fsum(item['wall_ms'] for item in stages.values())
        result = {
            'schema': SCHEMA,
            'stages': stages,
            'stage_wall_ms': stage_wall_ms,
        }
        if request_wall_ms is not None:
            if request_wall_ms < 0:
                raise ValueError('request_wall_ms must be non-negative')
            result['request_wall_ms'] = request_wall_ms
            result['coverage_fraction'] = (stage_wall_ms / request_wall_ms
                                           if request_wall_ms else None)
        return result

    def write_workload_trace(self, path, *, person_id: str, question_id: str,
                             question_sha256: str, request_wall_ms: float,
                             dense_retrieval_shape, image_count: int) -> dict:
        if len(dense_retrieval_shape) != 2:
            raise ValueError('dense retrieval shape must have two dimensions')
        profile = self.as_dict(request_wall_ms=request_wall_ms)
        reader_linears = {
            name: profile['stages'].get(name, {}).get('linear_operators', [])
            for name in READER_STAGE_NAMES
        }
        payload = {
            **profile,
            'schema': TRACE_SCHEMA,
            'person_id': str(person_id),
            'question_id': str(question_id),
            'question_sha256': question_sha256,
            'request_wall_ms': request_wall_ms,
            'image_count': int(image_count),
            'dense_retrieval': {
                'matrix_shape': [int(dense_retrieval_shape[0]),
                                 int(dense_retrieval_shape[1])],
            },
            'reader_linear_operators': reader_linears,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                                        sort_keys=True) + '\n')
        temporary.replace(path)
        return payload
