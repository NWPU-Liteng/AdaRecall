#!/usr/bin/env python3
"""Replay workload traces through the analytical PIM timing model."""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pim_backend.hyperpim_sc import plane_major_layout_plan
from pim_backend.hyperpim_timeline import (
    ENERGY_COUNT_KEYS, HyperPimTimelineConfig, energy_model,
    matrix_cycles, progressive_topk_cycles, simulate_token, with_weight_bits)
from pim_backend.structured_control import StructuredControlConfig, mask_scan_cost


TRACE_SCHEMA = 'omni-serving-workload-trace-v1'
REPLAY_SCHEMA = 'pim-replay-v1'
READER_PHASES = (
    'contract_parse', 'witness_construct', 'answer_prefill', 'answer_decode',
)
ESTIMATE_KEYS = (
    'full_precision_gemv', 'grammar_mask_only',
    'grammar_mask_plus_progressive', 'retrieval',
)
LM_PHASE_MAP = {
    'contract_parse': 'contract_parse',
    'witness_construct': 'witness_construct',
    'answer_decode': 'answer',
}


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _empty_counts():
    return {key: 0 for key in ENERGY_COUNT_KEYS}


def _add_counts(left, right):
    for key in ENERGY_COUNT_KEYS:
        left[key] += right.get(key, 0)


def _empty_estimate_counts():
    return {key: _empty_counts() for key in ESTIMATE_KEYS}


def _timing_config(profile):
    raw = profile.get('timing_profile') or {}
    if not isinstance(raw, Mapping):
        raise ValueError('timing_profile must be a mapping')
    try:
        return HyperPimTimelineConfig(**dict(raw))
    except TypeError as error:
        raise ValueError(f'invalid timing_profile: {error}') from error


def load_profile(path):
    payload = yaml.safe_load(Path(path).read_text())
    if not isinstance(payload, Mapping):
        raise ValueError('PIM profile must be a mapping')
    if payload.get('schema') != 'pim-profile-v2':
        raise ValueError('pim-profile-v2 schema required')
    host_cycles = payload.get('host_round_trip_cycles')
    if (host_cycles is None or isinstance(host_cycles, bool)
            or not isinstance(host_cycles, int) or host_cycles < 0):
        raise ValueError('host_round_trip_cycles must be filled before replay')
    energy_params = payload.get('energy_params') or {}
    if not isinstance(energy_params, Mapping):
        raise ValueError('energy_params must be a mapping')
    return {
        'schema': payload['schema'],
        'host_round_trip_cycles': host_cycles,
        'timing_profile': dict(payload.get('timing_profile') or {}),
        'energy_params': dict(energy_params),
        'source_path': str(path),
        'source_sha256': _digest(path),
    }


def _load_traces(trace_dir):
    trace_dir = Path(trace_dir)
    if not trace_dir.is_dir():
        raise ValueError(f'trace directory does not exist: {trace_dir}')
    traces = {}
    for path in sorted(trace_dir.rglob('*.json')):
        payload = json.loads(path.read_text())
        if payload.get('schema') != TRACE_SCHEMA:
            raise ValueError(f'unsupported workload trace schema: {path}')
        question_id = str(payload.get('question_id') or '')
        if not question_id or question_id in traces:
            raise ValueError(f'duplicate or missing trace question_id: {path}')
        traces[question_id] = {'path': str(path), 'payload': payload}
    if not traces:
        raise ValueError('no workload traces found')
    return traces


def _load_stats(stats_dir):
    stats_dir = Path(stats_dir)
    if not stats_dir.is_dir():
        raise ValueError(f'stats directory does not exist: {stats_dir}')
    rows = {}
    hashes = {}
    for path in sorted(stats_dir.glob('user_*_shard_*.json')):
        payload = json.loads(path.read_text())
        if not payload.get('complete'):
            continue
        hashes[str(path)] = _digest(path)
        file_rows = list(payload.get('rows', []))
        lm_stats = payload.get('progressive_lm_head_stats')
        if lm_stats is not None and lm_stats.get('enabled') and len(file_rows) != 1:
            raise ValueError(
                'progressive lm_head stats are shard-level and cannot be '
                f'attributed to {len(file_rows)} questions: {path}')
        retrieval_stats = payload.get('progressive_retrieval_stats') or {}
        for source_row in file_rows:
            row = dict(source_row)
            question_id = str(row.get('question_id') or '')
            if not question_id or question_id in rows:
                raise ValueError(f'duplicate or missing stats question_id: {path}')
            if lm_stats is not None:
                row['progressive_lm_head_stats'] = lm_stats
                row['progressive_lm_head_attribution'] = 'single_question_shard'
            if retrieval_stats.get('enabled'):
                demands = [
                    demand for demand in retrieval_stats.get('demands', [])
                    if str(demand.get('question_id') or question_id)
                    == question_id
                ]
                row['progressive_retrieval_stats'] = {
                    **retrieval_stats,
                    'demands': demands,
                    'demand_count': len(demands),
                }
            rows[question_id] = row
    if not rows:
        raise ValueError('no complete stats rows found')
    return rows, hashes


def _linear_sequence(trace, phase):
    operators = (trace.get('reader_linear_operators') or {}).get(phase) or []
    sequence = []
    for operator in operators:
        shape = operator.get('shape')
        calls = operator.get('calls')
        if (not isinstance(shape, list) or len(shape) != 2
                or isinstance(calls, bool) or not isinstance(calls, int)
                or calls <= 0):
            raise ValueError('invalid reader linear operator record')
        outputs, inputs = int(shape[0]), int(shape[1])
        if outputs <= 0 or inputs <= 0:
            raise ValueError('positive reader linear shapes required')
        sequence.append({
            'name': str(operator.get('name') or 'reader_linear'),
            'inputs': inputs,
            'outputs': outputs,
            'count': calls,
        })
    return sequence


def _full_gemv(sequence, config, bits):
    if not sequence:
        return {'cycles': 0, 'counts': _empty_counts()}
    simulated = simulate_token(sequence, with_weight_bits(config, bits))
    counts = _empty_counts()
    for item in sequence:
        estimate = matrix_cycles(item['inputs'], item['outputs'],
                                 with_weight_bits(config, bits))
        calls = item['count']
        counts['row_activations'] += (
            estimate['row_activations_per_bank']
            * estimate['active_banks'] * calls)
        counts['internal_read_bytes'] += (
            estimate['weight_bytes_per_bank']
            * estimate['active_banks'] * calls)
        counts['io_bytes'] += estimate['external_transfer_bytes'] * calls
        counts['bf16_macs'] += item['inputs'] * item['outputs'] * calls
    counts['refresh_cycles'] = simulated['totals']['refresh_cycles']
    return {'cycles': simulated['totals']['elapsed_cycles'], 'counts': counts}


def _grammar_circuit(config, decode_tokens, vocab_size):
    if isinstance(decode_tokens, bool) or not isinstance(decode_tokens, int):
        raise ValueError('decode token count must be an integer')
    if decode_tokens < 0:
        raise ValueError('decode token count must be nonnegative')
    if decode_tokens == 0:
        return {'cycles': 0, 'counts': _empty_counts()}
    if isinstance(vocab_size, bool) or not isinstance(vocab_size, int) or vocab_size <= 0:
        return {
            'cycles': None,
            'counts': _empty_counts(),
            'status': 'valid_vocab_size_missing',
        }
    control = StructuredControlConfig(
        stacks=config.stacks,
        channels_per_stack=config.channels_per_stack,
        banks_per_channel=config.banks_per_channel,
        clock_hz=config.clock_hz,
        bank_internal_bytes_per_cycle=config.bank_internal_bytes_per_cycle,
    )
    result = mask_scan_cost(decode_tokens, vocab_size, control)
    counts = _empty_counts()
    counts['io_bytes'] = result['descriptor_bytes_per_token'] * decode_tokens
    return {'cycles': result['cycles'], 'counts': counts, 'status': 'ok'}


def _lmhead_columns(stats):
    build = stats.get('build') or {}
    vocab_rows = int(stats.get('vocab_rows') or 0)
    bits = int(stats.get('bits') or 0)
    plane_bytes = int(build.get('plane_bytes') or 0)
    if vocab_rows <= 0 or bits not in (8, 10, 12) or plane_bytes <= 0:
        raise ValueError('lm_head stats cannot determine plane geometry')
    denominator = vocab_rows * bits
    if plane_bytes % denominator:
        raise ValueError('lm_head plane bytes do not align to rows and bits')
    return (plane_bytes // denominator) * 8


def _phase_rounds(phase_stats, stats):
    calls = int(phase_stats.get('progressive_calls') or 0)
    if calls == 0:
        return None
    rounds_total = phase_stats.get('average_rounds')
    if rounds_total is None:
        return None
    rounded = int(round(float(rounds_total)))
    if rounded <= 0 or abs(float(rounds_total) - rounded) > 1e-9:
        raise ValueError('average_rounds must be a positive integer for replay')
    schedule = list(stats.get('plane_schedule') or [])
    if len(schedule) != rounded:
        raise ValueError('lm_head plane schedule does not match average rounds')
    plane_bytes_per_row = math.ceil(
        int(stats['build']['plane_bytes']) / (int(stats['vocab_rows'])
                                             * int(stats['bits'])))
    weight_bytes = int(phase_stats.get('weight_bytes_read') or 0)
    if plane_bytes_per_row <= 0 or weight_bytes <= 0:
        raise ValueError('lm_head phase stats have no measurable plane reads')
    plane_rows_per_call = weight_bytes / plane_bytes_per_row / calls
    if plane_rows_per_call <= 0:
        raise ValueError('lm_head byte proxy produced no plane rows per call')
    plane_sum = sum(schedule)
    rounds = []
    for planes in schedule:
        rows = max(1, int(round(
            plane_rows_per_call * planes / plane_sum)))
        rounds.append({'planes_in_round': planes, 'rows_alive_before': rows})
    return rounds


def _lmhead_proxy(stats, phase, config, host_round_trip_cycles):
    if not stats.get('enabled') or phase is None:
        return None
    phase_stats = (stats.get('phases') or {}).get(phase)
    if not phase_stats or not int(phase_stats.get('progressive_calls') or 0):
        return None
    rounds = _phase_rounds(phase_stats, stats)
    if rounds is None:
        return None
    cols = _lmhead_columns(stats)
    layout = plane_major_layout_plan(
        int(stats['vocab_rows']), cols, bits=int(stats['bits']),
        group_size=int(stats['group_size']), row_bytes=config.row_bytes,
        tile_rows=16, bank_count=config.bank_count)
    per_token = progressive_topk_cycles(
        layout, rounds, config, host_round_trip_cycles)
    calls = int(phase_stats['progressive_calls'])
    totals = dict(per_token['totals'])
    for key in (
            'row_activations_total', 'internal_read_cycles', 'mac_cycles',
            'bound_cycles', 'threshold_broadcast_cycles',
            'internal_read_bytes', 'partial_macs', 'bound_updates',
            'active_cycles', 'refresh_cycles', 'elapsed_cycles',
            'active_ms', 'elapsed_ms'):
        totals[key] *= calls
    estimate = dict(per_token)
    estimate['per_progressive_call_totals'] = per_token['totals']
    estimate['totals'] = totals
    estimate['source'] = 't22_phase_byte_proxy'
    estimate['phase'] = phase
    estimate['checked_tokens'] = int(phase_stats.get('checked_tokens') or 0)
    estimate['progressive_calls'] = calls
    estimate['forced_tokens'] = int(phase_stats.get('forced_tokens') or 0)
    return estimate


def _retrieval_estimate(trace, stats, config, host_round_trip_cycles):
    retrieval = stats.get('progressive_retrieval_stats') or {}
    if not retrieval.get('enabled') or not retrieval.get('demands'):
        return None
    shape = (trace.get('dense_retrieval') or {}).get('matrix_shape')
    if (not isinstance(shape, list) or len(shape) != 2
            or int(shape[0]) <= 0 or int(shape[1]) <= 0):
        raise ValueError('trace dense_retrieval.matrix_shape is invalid')
    rows, cols = int(shape[0]), int(shape[1])
    bits = int(retrieval.get('bits') or 0)
    group_size = int(retrieval.get('group_size') or 0)
    layout = plane_major_layout_plan(
        rows, cols, bits=bits, group_size=group_size,
        row_bytes=config.row_bytes, tile_rows=16,
        bank_count=config.bank_count)
    demand_estimates = []
    counts = _empty_counts()
    total_cycles = 0
    for demand in retrieval['demands']:
        rounds = demand.get('rounds') or []
        if not rounds:
            continue
        estimate = progressive_topk_cycles(
            layout, rounds, config, host_round_trip_cycles)
        estimate['demand_id'] = demand.get('demand_id')
        demand_estimates.append(estimate)
        total_cycles += estimate['totals']['elapsed_cycles']
        totals = estimate['totals']
        counts['row_activations'] += totals['row_activations_total']
        counts['internal_read_bytes'] += totals['internal_read_bytes']
        counts['bf16_macs'] += totals['partial_macs']
        counts['bound_updates'] += totals['bound_updates']
        counts['refresh_cycles'] += totals['refresh_cycles']
    if not demand_estimates:
        return None
    full_config = with_weight_bits(config, bits)
    full = matrix_cycles(cols, rows, full_config)
    full_cycles = full['active_cycles'] * len(demand_estimates)
    full_counts = _empty_counts()
    full_counts['row_activations'] = (
        full['row_activations_per_bank'] * full['active_banks']
        * len(demand_estimates))
    full_counts['internal_read_bytes'] = (
        full['weight_bytes_per_bank'] * full['active_banks']
        * len(demand_estimates))
    full_counts['io_bytes'] = (
        full['external_transfer_bytes'] * len(demand_estimates))
    full_counts['bf16_macs'] = rows * cols * len(demand_estimates)
    return {
        'full_precision_gemv_cycles': full_cycles,
        'grammar_mask_only_cycles': 0,
        'grammar_mask_plus_progressive_cycles': total_cycles,
        'retrieval_cycles': total_cycles,
        'demand_count': len(demand_estimates),
        'demand_estimates': demand_estimates,
        'counts_by_estimate': {
            'full_precision_gemv': full_counts,
            'grammar_mask_only': _empty_counts(),
            'grammar_mask_plus_progressive': counts,
            'retrieval': counts,
        },
    }


def _reader_phase(trace, stats, phase, config, vocab_size,
                  host_round_trip_cycles, bits):
    sequence = _linear_sequence(trace, phase)
    full = _full_gemv(sequence, config, bits)
    stage = (trace.get('stages') or {}).get(phase) or {}
    grammar = _grammar_circuit(
        config, int(stage.get('decode_tokens') or 0), vocab_size)
    lm_phase = LM_PHASE_MAP.get(phase)
    progressive = _lmhead_proxy(
        stats, lm_phase, config, host_round_trip_cycles)
    progressive_cycles = (progressive['totals']['elapsed_cycles']
                          if progressive is not None else 0)
    progressive_counts = _empty_counts()
    if progressive is not None:
        totals = progressive['totals']
        progressive_counts['row_activations'] = totals['row_activations_total']
        progressive_counts['internal_read_bytes'] = totals['internal_read_bytes']
        progressive_counts['bf16_macs'] = totals['partial_macs']
        progressive_counts['bound_updates'] = totals['bound_updates']
        progressive_counts['refresh_cycles'] = totals['refresh_cycles']
    grammar_plus_progressive = dict(grammar['counts'])
    _add_counts(grammar_plus_progressive, progressive_counts)
    return {
        'full_precision_gemv_cycles': full['cycles'],
        'grammar_mask_only_cycles': grammar['cycles'],
        'grammar_mask_plus_progressive_cycles': (
            grammar['cycles'] + progressive_cycles
            if grammar['cycles'] is not None else None),
        'retrieval_cycles': None,
        'decode_tokens': int(stage.get('decode_tokens') or 0),
        'grammar_status': grammar.get('status'),
        'progressive_source': (
            progressive['source'] if progressive is not None else None),
        'progressive_totals': (
            progressive['totals'] if progressive is not None else None),
        'checked_tokens': (
            progressive['checked_tokens'] if progressive is not None else 0),
        'progressive_calls': (
            progressive['progressive_calls']
            if progressive is not None else 0),
        'forced_tokens': (
            progressive['forced_tokens'] if progressive is not None else 0),
        'counts_by_estimate': {
            'full_precision_gemv': full['counts'],
            'grammar_mask_only': grammar['counts'],
            'grammar_mask_plus_progressive': grammar_plus_progressive,
            'retrieval': _empty_counts(),
        },
    }


def build_replay(trace_dir, stats_dir, profile):
    traces = _load_traces(trace_dir)
    rows, stats_hashes = _load_stats(stats_dir)
    if set(traces) != set(rows):
        raise ValueError(
            'trace and stats question sets differ: '
            f'only_trace={sorted(set(traces) - set(rows))} '
            f'only_stats={sorted(set(rows) - set(traces))}')
    config = _timing_config(profile)
    host_cycles = profile['host_round_trip_cycles']
    questions = []
    aggregate_counts = _empty_estimate_counts()
    for question_id in sorted(traces):
        trace = traces[question_id]['payload']
        row = rows[question_id]
        lm_stats = row.get('progressive_lm_head_stats') or {}
        vocab_size = int(lm_stats.get('valid_vocab_size') or 0)
        bits = int((row.get('progressive_retrieval_stats') or {}).get('bits')
                   or lm_stats.get('bits') or 12)
        phases = {}
        for phase in READER_PHASES:
            result = _reader_phase(
                trace, lm_stats, phase, config, vocab_size,
                host_cycles, bits)
            phases[phase] = result
            for estimate_key in ESTIMATE_KEYS:
                _add_counts(
                    aggregate_counts[estimate_key],
                    result['counts_by_estimate'][estimate_key])
        retrieval = _retrieval_estimate(
            trace, row, config, host_cycles)
        if retrieval is not None:
            phases['dense_retrieval'] = retrieval
            for estimate_key in ESTIMATE_KEYS:
                _add_counts(
                    aggregate_counts[estimate_key],
                    retrieval['counts_by_estimate'][estimate_key])
        else:
            empty = _empty_counts()
            phases['dense_retrieval'] = {
                'full_precision_gemv_cycles': 0,
                'grammar_mask_only_cycles': 0,
                'grammar_mask_plus_progressive_cycles': 0,
                'retrieval_cycles': None,
                'counts_by_estimate': _empty_estimate_counts(),
            }
        questions.append({
            'question_id': question_id,
            'person_id': str(row.get('person_id')),
            'trace_path': traces[question_id]['path'],
            'trace_sha256': _digest(traces[question_id]['path']),
            'phases': phases,
        })
    energy_by_estimate = {
        key: energy_model(counts, profile.get('energy_params') or {})
        for key, counts in aggregate_counts.items()
    }
    return {
        'schema': REPLAY_SCHEMA,
        'complete': True,
        'profile': {
            'schema': profile['schema'],
            'source_path': profile['source_path'],
            'source_sha256': profile['source_sha256'],
            'host_round_trip_cycles': host_cycles,
            'timing_profile': profile['timing_profile'],
        },
        'question_count': len(questions),
        'questions': questions,
        'aggregate_counts_by_estimate': aggregate_counts,
        'energy_by_estimate': energy_by_estimate,
        'inputs_sha256': stats_hashes,
        'boundary': (
            'analytical replay over serving traces and frozen stats; '
            'lm_head phase rounds use a byte/round-count proxy because T2.2 '
            'does not retain survivor identities; no physical PIM measurement'),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('trace_dir', type=Path)
    parser.add_argument('stats_dir', type=Path)
    parser.add_argument('--profile-config', type=Path,
                        default=ROOT / 'configs/pim_profile_v2.yaml')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    profile = load_profile(args.profile_config)
    result = build_replay(args.trace_dir, args.stats_dir, profile)
    text = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    if args.output is not None:
        if args.output.exists():
            raise ValueError('fresh replay output required')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end='')


if __name__ == '__main__':
    main()
