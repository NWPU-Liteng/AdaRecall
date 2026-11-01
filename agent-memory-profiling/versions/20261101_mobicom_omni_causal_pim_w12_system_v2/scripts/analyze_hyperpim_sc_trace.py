"""Aggregate label-free parser/grounding traces for the HyperPIM-SC Gate 0."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics


def percentile(values, fraction):
    values = sorted(values)
    if not values:
        return None
    index = min(len(values) - 1, max(0, math.ceil(fraction * len(values)) - 1))
    return values[index]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def operator_summary(trace):
    if not trace:
        return None
    by_phase = defaultdict(lambda: {'calls': 0, 'macs': 0, 'weight_bytes': 0})
    unique_decode_weights = {}
    unique_decode_scale_bytes = 0
    for row in trace.get('operators', []):
        phase = row['phase']
        calls = int(row['calls'])
        by_phase[phase]['calls'] += calls
        by_phase[phase]['macs'] += calls * int(row['rows']) * int(row['in_features']) * int(row['out_features'])
        by_phase[phase]['weight_bytes'] += calls * int(row['weight_bytes'])
        if phase == 'decode' and int(row['rows']) == 1:
            unique_decode_weights[row['name']] = int(row['weight_bytes'])
            unique_decode_scale_bytes += int(row['out_features']) * 4
    bf16_decode_bytes = sum(unique_decode_weights.values())
    return {
        'by_phase': dict(by_phase),
        'decode_unique_linear_count': len(unique_decode_weights),
        'decode_unique_bf16_weight_bytes': bf16_decode_bytes,
        'decode_unique_w8_weight_bytes': bf16_decode_bytes // 2 + unique_decode_scale_bytes,
        'decode_unique_scale_bytes': unique_decode_scale_bytes,
        'decode_single_row_linear_calls': int(trace.get('decode_single_row_linear_calls', 0)),
        'decode_single_row_macs': int(trace.get('decode_single_row_macs', 0)),
        'coverage': trace.get('coverage'),
        'measurement': trace.get('measurement'),
    }


def classify_operator(name):
    lowered = name.lower()
    if 'profilerstep' in lowered:
        return 'profiler_pseudo'
    if any(token in lowered for token in ('gemm', 'gemv', 'cutlass', 'wmma', 'aten::mm')):
        return 'linear_gemm'
    if 'attention' in lowered:
        return 'attention'
    if any(token in lowered for token in ('gateddelta', 'recurrent', 'conv1d')):
        return 'recurrent_control'
    if any(token in lowered for token in ('layernorm', 'rms', 'aten::mean', 'aten::pow',
                                           'aten::rsqrt', 'silu')):
        return 'norm_elementwise'
    if any(token in lowered for token in ('aten::cat', 'aten::copy', 'memcpy', 'aten::empty',
                                           'memset')):
        return 'data_movement'
    return 'other'


def profiler_summary(profile, phase):
    if not profile or phase not in profile:
        return None
    events = profile[phase].get('operator_events', [])
    categories = defaultdict(float)
    for event in events:
        if classify_operator(event['operator']) == 'profiler_pseudo':
            continue
        categories[classify_operator(event['operator'])] += event['self_device_microseconds']
    calls = int(profile[phase].get('profiled_model_forward_calls') or 0)
    return {
        'profiled_model_forward_calls': calls,
        'device_microseconds_by_category': dict(sorted(categories.items())),
        'device_microseconds_per_forward': ({name: value / calls for name, value in categories.items()}
                                            if calls else None),
        'top_self_device_events': [
            {key: event[key] for key in ('operator', 'calls', 'input_shapes',
                                         'self_device_microseconds', 'self_cpu_microseconds')}
            for event in events if classify_operator(event['operator']) != 'profiler_pseudo'
        ][:12],
    }


def grounding_attempts(plan):
    attempts = list(plan.get('frontier_attempts') or [])
    attempts += list((plan.get('grounding') or {}).get('frontier_attempts') or [])
    unique = []
    seen = set()
    for attempt in attempts:
        identity = id(attempt)
        if identity not in seen:
            seen.add(identity)
            unique.append(attempt)
    return unique


def request_row(row):
    plan = row['proof_plan']
    retrieval = plan.get('retrieval') or {}
    parse_timing = retrieval.get('timing') or {}
    parser = {
        'wall_seconds': parse_timing.get('query_parser_total_seconds'),
        'input_tokens': parse_timing.get('query_parser_input_tokens'),
        'output_tokens': (plan.get('query_parse') or {}).get('output_token_count'),
        'structured_call_seconds': (plan.get('query_parse') or {}).get('seconds'),
        'prefill_seconds': (plan.get('query_parse') or {}).get('timing', {}).get(
            'reader_prefill_seconds'),
        'decode_seconds': (plan.get('query_parse') or {}).get('timing', {}).get(
            'reader_decode_seconds'),
        'first_token_seconds': parse_timing.get('query_parser_first_token_seconds'),
        'decode_phase_profile': parse_timing.get('query_parser_decode_phase_profile'),
        'operator_trace': operator_summary(retrieval.get('linear_operator_trace')),
        'operator_profile': {
            phase: profiler_summary(retrieval.get('operator_profile'), phase)
            for phase in ('prefill', 'decode')
        },
    }
    grounding = []
    for attempt in grounding_attempts(plan):
        timing = attempt.get('reader_timing') or {}
        grounding.append({
            'level': attempt.get('level'),
            'candidate_count': attempt.get('grounding_candidate_count'),
            'contract_complete': attempt.get('contract_complete'),
            'wall_seconds': None,
            'input_tokens': timing.get('input_tokens'),
            'output_tokens': timing.get('output_tokens'),
            'prefill_seconds': timing.get('prefill_seconds'),
            'decode_seconds': timing.get('decode_seconds'),
            'first_token_seconds': timing.get('first_token_seconds'),
            'decode_phase_profile': timing.get('decode_phase_profile'),
            'operator_trace': operator_summary(attempt.get('linear_operator_trace')),
            'operator_profile': {
                phase: profiler_summary(attempt.get('operator_profile'), phase)
                for phase in ('prefill', 'decode')
            },
        })
    return {
        'person_id': row['person_id'],
        'question_id': row['question_id'],
        'service_seconds_not_comparable': row.get('service_seconds'),
        'execution_event_seconds': {event['phase']: event.get('seconds')
                                    for event in row.get('execution_events', [])},
        'parser': parser,
        'grounding_attempts': grounding,
    }


def add_stage_totals(rows):
    totals = defaultdict(float)
    tokens = 0
    available = 0
    for row in rows:
        calls = [row['parser']] + row['grounding_attempts']
        for call in calls:
            profile = call.get('decode_phase_profile')
            if not profile:
                continue
            available += 1
            tokens += int(profile.get('generated_tokens') or 0)
            for key, value in profile.items():
                if isinstance(value, (int, float)) and key != 'generated_tokens':
                    totals[key] += float(value)
    return ({'calls': available, 'generated_tokens': tokens,
             'seconds': dict(sorted(totals.items())),
             'milliseconds_per_token': {key: 1000 * value / tokens for key, value in totals.items()}
             if tokens else None} if available else None)


def accounting_totals(rows):
    result = {}
    for kind in ('parser', 'grounding'):
        calls = ([item['parser'] for item in rows] if kind == 'parser'
                 else [attempt for item in rows for attempt in item['grounding_attempts']])
        traces = [call['operator_trace'] for call in calls if call['operator_trace']]
        output_tokens = sum(int(call.get('output_tokens') or 0) for call in calls)
        input_tokens = sum(int(call.get('input_tokens') or 0) for call in calls)
        decode_linear_calls = sum(
            int(trace['decode_single_row_linear_calls']) for trace in traces)
        unique_w8_bytes = [int(trace['decode_unique_w8_weight_bytes']) for trace in traces]
        result[kind] = {
            'calls': len(calls),
            'calls_with_linear_trace': len(traces),
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'decode_linear_calls': decode_linear_calls,
            'decode_linear_calls_per_output_token': (
                decode_linear_calls / output_tokens if output_tokens else None),
            'decode_unique_w8_weight_bytes_min': min(unique_w8_bytes) if unique_w8_bytes else None,
            'decode_unique_w8_weight_bytes_median': (
                statistics.median(unique_w8_bytes) if unique_w8_bytes else None),
            'decode_unique_w8_weight_bytes_max': max(unique_w8_bytes) if unique_w8_bytes else None,
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run_paths = sorted(args.root.glob('user_*.json'))
    run_paths = [path for path in run_paths if not path.name.endswith('.partial.json')]
    if len(run_paths) != 4:
        raise ValueError('exactly four complete per-user trace files required')
    rows = []
    identities = []
    for path in run_paths:
        run = json.loads(path.read_text())
        if not run.get('complete') or not run.get('pre_reader_operator_profile', {}).get('enabled'):
            raise ValueError('complete pre-reader operator profile required: ' + str(path))
        for row in run['rows']:
            rows.append(request_row(row))
        identities.append({'path': str(path), 'person_id': run['person_id'],
                           'question_ids': run['question_ids'], 'sha256': digest(path)})
    if len(rows) != 16:
        raise ValueError('sixteen request traces required')

    parser_traces = [row['parser']['operator_trace'] for row in rows
                     if row['parser']['operator_trace']]
    grounding_traces = [attempt['operator_trace'] for row in rows
                        for attempt in row['grounding_attempts'] if attempt['operator_trace']]
    all_traces = parser_traces + grounding_traces
    footprints = [trace['decode_unique_w8_weight_bytes'] for trace in all_traces]
    summary = {
        'schema': 'hyperpim-sc-gate0-v1',
        'measurement_boundary': ('label-free development operator-shape trace; profiling wall times are not '
                                 'performance measurements; target-PIM cost is not executed'),
        'runs': identities,
        'requests': len(rows),
        'structured_calls': {'parser': len(parser_traces), 'grounding': len(grounding_traces)},
        'decode_unique_w8_weight_bytes': {
            'min': min(footprints), 'median': statistics.median(footprints), 'max': max(footprints),
        },
        'decode_unique_w8_weight_gib': {
            'min': min(footprints) / 1024**3,
            'median': statistics.median(footprints) / 1024**3,
            'max': max(footprints) / 1024**3,
        },
        'accounting_totals': accounting_totals(rows),
        'stage_timing_totals': add_stage_totals(rows),
        'rows': rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: summary[key] for key in
                      ('requests', 'structured_calls', 'decode_unique_w8_weight_gib',
                       'stage_timing_totals')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
