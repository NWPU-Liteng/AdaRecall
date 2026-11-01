"""Project HyperPIM-SC from measured 200-request stages and grammar replay."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics

from pim_backend.structured_control import StructuredControlConfig, mask_scan_cost
from pim_backend.hyperpim_sc import (DecodeCostConfig, W8A16CostConfig,
                                     bank_resident_decoder_plan,
                                     grammar_residency,
                                     w8a16_mixed_precision_overhead)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return None
    return ordered[min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))]


def latency(values):
    if not values:
        return None
    return {
        'count': len(values),
        'sum_seconds': sum(values),
        'mean_seconds': statistics.fmean(values),
        'p50_seconds': percentile(values, 0.50),
        'p95_seconds': percentile(values, 0.95),
        'p99_seconds': percentile(values, 0.99),
    }


def median(values):
    values = [value for value in values if value is not None]
    if not values:
        raise ValueError('nonempty metric sample required')
    return statistics.median(values)


def load_runs(paths):
    rows = []
    provenance = {}
    for path in paths:
        run = json.loads(path.read_text())
        if not run.get('complete') or run.get('evaluation_label_files_read') != []:
            raise ValueError('complete label-isolated run required: ' + str(path))
        rows.extend(run['rows'])
        provenance[str(path)] = digest(path)
    identities = [row['question_id'] for row in rows]
    if len(rows) != 200 or len(set(identities)) != 200:
        raise ValueError('the frozen 200-request cohort is required')
    return rows, provenance


def sample_trace(trace_root):
    paths = sorted(path for path in trace_root.glob('user_*.json')
                   if not path.name.endswith('.partial.json'))
    if len(paths) != 4:
        raise ValueError('four complete sample trace files required')
    sample = None
    for path in paths:
        run = json.loads(path.read_text())
        for row in run['rows']:
            parse = row['proof_plan'].get('query_parse') or {}
            tokens = int(parse.get('output_token_count') or 0)
            operators = ((row['proof_plan'].get('retrieval') or {})
                         .get('linear_operator_trace') or {}).get('operators') or []
            if tokens > 1 and operators:
                sample = {'path': str(path), 'question_id': row['question_id'],
                          'output_tokens': tokens, 'operators': operators}
                break
        if sample:
            break
    if sample is None:
        raise ValueError('one decode operator trace required')
    return sample


def matrix_cost_per_token(sample, matrix_costs):
    costs = {row['matrix_id']: row for row in matrix_costs}
    steps = sample['output_tokens'] - 1
    calls = defaultdict(float)
    for operator in sample['operators']:
        if operator['phase'] != 'decode' or operator['rows'] != 1:
            continue
        rate = operator['calls'] / steps
        if not math.isclose(rate, round(rate), abs_tol=1e-9):
            raise ValueError('decode call count is not a fixed per-token signature')
        calls[f"{operator['out_features']}x{operator['in_features']}"] += round(rate)
    if sum(calls.values()) != 249:
        raise ValueError('expected 249 single-row decoder linear calls per token')
    if not set(calls) <= set(costs):
        raise ValueError('missing GeniePIM matrix cost for an executed shape')
    return {
        'steps': steps,
        'calls_per_token': dict(sorted(calls.items())),
        'host_linear_ms_per_token': sum(
            calls[shape] * costs[shape]['cost']['host_gemm_time_ns'] for shape in calls) / 1e6,
        'pim_linear_ms_per_token': sum(
            calls[shape] * costs[shape]['cost']['pim_time_ns'] for shape in calls) / 1e6,
    }


def gpu_nonlinear_ms_per_token(gate0):
    samples = []
    for row in gate0['rows']:
        calls = [row['parser']] + row['grounding_attempts']
        for call in calls:
            profile = (call.get('operator_profile') or {}).get('decode') or {}
            forwards = int(profile.get('profiled_model_forward_calls') or 0)
            categories = profile.get('device_microseconds_by_category') or {}
            if not forwards or not categories:
                continue
            total = sum(float(value) for value in categories.values())
            linear = float(categories.get('linear_gemm', 0.0))
            samples.append((total - linear) / forwards / 1000)
    if not samples:
        raise ValueError('profiled decode activity required')
    return {'median_ms_per_token': median(samples), 'samples': len(samples)}


def grammar_fill_calibration(gate0, grammar, probe_path):
    if probe_path is None:
        return {'factor': 1.0, 'source': None}
    probe = json.loads(probe_path.read_text())
    constrained = probe['constrained']
    profile = constrained.get('decode_phase_profile') or {}
    target_tokens = int(profile.get('generated_tokens') or 0)
    if target_tokens <= 0:
        raise ValueError('calibration probe must generate at least one token')
    question_id = probe['question_id']
    replay_call = next((call for call in grammar['calls']
                        if call['kind'] == 'parser' and call['question_id'] == question_id), None)
    if replay_call is None or len(replay_call['tokens']) < target_tokens:
        raise ValueError('calibration question must occur on a complete parser replay path')
    replay_seconds = sum(token['fill_seconds']
                         for token in replay_call['tokens'][:target_tokens])
    measured_seconds = float(profile['grammar_fill_seconds'])
    if replay_seconds <= 0 or measured_seconds <= 0:
        raise ValueError('positive calibration fill times required')
    return {
        'factor': measured_seconds / replay_seconds,
        'source': str(probe_path),
        'question_id': question_id,
        'tokens': target_tokens,
        'replay_seconds': replay_seconds,
        'measured_seconds': measured_seconds,
    }


def grammar_forward_samples(gate0, grammar, fill_factor=1.0):
    result = {}
    for kind in ('parser', 'grounding'):
        calls = [row for row in grammar['calls'] if row['kind'] == kind]
        call_by_id = defaultdict(list)
        for row in calls:
            call_by_id[(str(row['person_id']), row['question_id'])].append(row)
        call_ordinals = defaultdict(int)
        grammar_ms = []
        forward_ms = []
        prefill_seconds = []
        for trace_row in gate0['rows']:
            observed = ([trace_row['parser']] if kind == 'parser'
                        else trace_row['grounding_attempts'])
            for call in observed:
                key = (str(trace_row['person_id']), trace_row['question_id'])
                ordinal = call_ordinals[key]
                call_ordinals[key] += 1
                replay = (call_by_id.get(key) or [None])[ordinal] if ordinal < len(
                    call_by_id.get(key) or []) else None
                if replay is None or not call.get('decode_seconds'):
                    continue
                tokens = len(replay['tokens'])
                if not tokens:
                    continue
                fill = fill_factor * sum(token['fill_seconds'] for token in replay['tokens'])
                grammar_ms.append(1000 * fill / tokens)
                forward_ms.append(1000 * max(0.0, call['decode_seconds'] - fill) / tokens)
                if call.get('prefill_seconds') is not None:
                    prefill_seconds.append(float(call['prefill_seconds']))
        if not grammar_ms:
            raise ValueError('paired grammar replay and stage timing required for ' + kind)
        result[kind] = {
            'calls': len(grammar_ms),
            'grammar_ms_per_token_median': median(grammar_ms),
            'forward_ms_per_token_median': median(forward_ms),
            'prefill_seconds_median': median(prefill_seconds) if prefill_seconds else 0.0,
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    parser.add_argument('--gate0', type=Path, required=True)
    parser.add_argument('--grammar', type=Path, required=True)
    parser.add_argument('--matrix-costs', type=Path, required=True)
    parser.add_argument('--trace-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dispatch-ms-per-token', type=float, default=4.0)
    parser.add_argument('--target-safety-factor', type=float, default=1.15)
    parser.add_argument('--pim-token-test-cycles', type=int, default=64)
    parser.add_argument('--precision', choices=('w8a8', 'w8a16'), default='w8a16')
    parser.add_argument('--w8a16-group-size', type=int, default=128)
    parser.add_argument('--w8a16-mac-slowdown', type=float, default=1.0)
    parser.add_argument('--w8a16-conversion-lanes', type=int, default=16)
    parser.add_argument('--w8a16-broadcast-gbps', type=float, default=90.0693433912629)
    parser.add_argument('--w8a16-residual-bits', type=int, choices=(0, 2, 4), default=0)
    parser.add_argument('--fallback-weight-elements', type=int, default=0,
                        help='BF16 fallback Linear weight elements not resident in PIM')
    parser.add_argument('--calibration-probe', type=Path)
    args = parser.parse_args()
    if (not math.isfinite(args.dispatch_ms_per_token) or args.dispatch_ms_per_token < 0
            or not math.isfinite(args.target_safety_factor) or args.target_safety_factor < 1):
        raise ValueError('finite nonnegative dispatch and safety factor at least one required')
    if args.w8a16_group_size <= 0 or args.w8a16_conversion_lanes <= 0:
        raise ValueError('positive W8A16 group size and conversion lanes required')
    if args.w8a16_residual_bits not in (0, 2, 4):
        raise ValueError('residual bits must be zero, two or four')
    if not math.isfinite(args.w8a16_broadcast_gbps) or args.w8a16_broadcast_gbps <= 0:
        raise ValueError('positive finite W8A16 broadcast bandwidth required')
    if args.fallback_weight_elements < 0:
        raise ValueError('nonnegative BF16 fallback elements required')

    rows, provenance = load_runs(args.runs)
    gate0_path = args.gate0
    grammar_path = args.grammar
    matrix_path = args.matrix_costs
    gate0 = json.loads(gate0_path.read_text())
    grammar = json.loads(grammar_path.read_text())
    matrix_costs = json.loads(matrix_path.read_text())
    sample = sample_trace(args.trace_root)
    linear = matrix_cost_per_token(sample, matrix_costs)
    nonlinear = gpu_nonlinear_ms_per_token(gate0)
    calibration = grammar_fill_calibration(gate0, grammar, args.calibration_probe)
    grammar_profiles = grammar_forward_samples(gate0, grammar, calibration['factor'])
    pim_grammar = mask_scan_cost(
        1, int(grammar['vocab_size']),
        StructuredControlConfig(token_test_cycles=args.pim_token_test_cycles))
    pim_grammar_ms = pim_grammar['milliseconds_per_token']
    decoder_matrices = []
    for shape, count in linear['calls_per_token'].items():
        outputs, inputs = (int(value) for value in shape.split('x'))
        if not math.isclose(count, round(count), abs_tol=1e-9):
            raise ValueError('nonintegral decoder matrix multiplicity')
        decoder_matrices.extend({
            'name': f'{shape}:{index}', 'out_features': outputs, 'in_features': inputs,
        } for index in range(round(count)))
    total_weight_elements = sum(matrix['out_features'] * matrix['in_features']
                                for matrix in decoder_matrices)
    if args.fallback_weight_elements > total_weight_elements:
        raise ValueError('BF16 fallback exceeds decoder weight elements')
    resident_fraction = 1.0 - args.fallback_weight_elements / total_weight_elements
    fallback_fraction = 1.0 - resident_fraction
    mixed_precision = None
    if args.precision == 'w8a16':
        mixed_precision = w8a16_mixed_precision_overhead(
            decoder_matrices, linear['pim_linear_ms_per_token'],
            StructuredControlConfig(token_test_cycles=args.pim_token_test_cycles),
            W8A16CostConfig(
                group_size=args.w8a16_group_size,
                conversion_elements_per_bank_cycle=args.w8a16_conversion_lanes,
                group_apply_ops_per_bank_cycle=args.w8a16_conversion_lanes,
                mac_slowdown=args.w8a16_mac_slowdown,
                broadcast_bandwidth_bytes_per_second=args.w8a16_broadcast_gbps * 1e9,
                residual_bits=args.w8a16_residual_bits))
        resident_pim_linear_ms = mixed_precision['pim_linear_ms_per_token'] * resident_fraction
        resident_overhead_ms = ((mixed_precision['total_overhead_ms_per_token']
                                 + mixed_precision['residual_mac_ms_per_token'])
                                * resident_fraction)
        host_fallback_linear_ms = (linear['host_linear_ms_per_token']
                                   * fallback_fraction)
        pim_linear_ms = (resident_pim_linear_ms + resident_overhead_ms
                         + host_fallback_linear_ms)
    else:
        resident_pim_linear_ms = linear['pim_linear_ms_per_token'] * resident_fraction
        resident_overhead_ms = 0.0
        host_fallback_linear_ms = linear['host_linear_ms_per_token'] * fallback_fraction
        pim_linear_ms = resident_pim_linear_ms + host_fallback_linear_ms
    gpu_nonlinear_ms = nonlinear['median_ms_per_token']
    cost_config = DecodeCostConfig(
        host_linear_ms_per_token=linear['host_linear_ms_per_token'],
        pim_linear_ms_per_token=pim_linear_ms,
        gpu_nonlinear_ms_per_token=gpu_nonlinear_ms,
        grammar_scan_ms_per_token=pim_grammar_ms,
        dispatch_ms_per_token=args.dispatch_ms_per_token,
        safety_factor=args.target_safety_factor)
    pim_reader_ms = cost_config.target_ms(structured=False)
    pim_structured_ms = cost_config.target_ms(structured=True)
    target_ms = pim_structured_ms

    baseline_service = []
    baseline_ttft = []
    control_service = []
    control_ttft = []
    target_service = []
    target_ttft = []
    requests = []
    decode_tokens = {'parser': 0, 'grounding': 0, 'reader': 0}
    for row in rows:
        parse = row['proof_plan'].get('query_parse') or {}
        grounding = row['proof_plan'].get('grounding') or {}
        parse_tokens = int(parse.get('output_token_count') or 0)
        grounding_tokens = int(grounding.get('output_token_count') or 0)
        parse_seconds = float(parse.get('seconds') or 0.0)
        grounding_seconds = float(grounding.get('seconds') or 0.0)
        reader_tokens = len(row.get('answer_token_ids') or [])
        reader_decode = float(row.get('decode_seconds') or 0.0)
        decode_tokens['parser'] += parse_tokens
        decode_tokens['grounding'] += grounding_tokens
        decode_tokens['reader'] += reader_tokens
        parse_profile = grammar_profiles['parser']
        grounding_profile = grammar_profiles['grounding']

        def projected_call(seconds, tokens, profile):
            decode = max(0.0, seconds - profile['prefill_seconds_median'])
            grammar_seconds = tokens * profile['grammar_ms_per_token_median'] / 1000
            forward_seconds = tokens * profile['forward_ms_per_token_median'] / 1000
            control = profile['prefill_seconds_median'] + max(
                forward_seconds, tokens * pim_grammar_ms / 1000)
            target = profile['prefill_seconds_median'] + tokens * target_ms / 1000
            return {
                'baseline_decode_seconds': decode,
                'estimated_grammar_seconds': grammar_seconds,
                'estimated_forward_seconds': forward_seconds,
                'control_total_seconds': control,
                'target_total_seconds': target,
            }

        parse_projection = projected_call(parse_seconds, parse_tokens, parse_profile)
        grounding_projection = projected_call(
            grounding_seconds, grounding_tokens, grounding_profile)
        control_parse_total = min(parse_seconds, parse_projection['control_total_seconds'])
        control_grounding_total = min(
            grounding_seconds, grounding_projection['control_total_seconds'])
        target_parse_total = min(parse_seconds, parse_projection['target_total_seconds'])
        target_grounding_total = min(
            grounding_seconds, grounding_projection['target_total_seconds'])
        service = float(row['service_seconds'])
        ttft = float(row['ttft_seconds'])
        control_service.append(service - (parse_seconds - control_parse_total)
                               - (grounding_seconds - control_grounding_total))
        control_ttft.append(ttft - (parse_seconds - control_parse_total)
                            - (grounding_seconds - control_grounding_total))
        target_reader_decode = min(
            reader_decode, reader_tokens * pim_reader_ms / 1000
        ) if reader_tokens else reader_decode
        target_service.append(service - (parse_seconds - target_parse_total)
                              - (grounding_seconds - target_grounding_total)
                              - (reader_decode - target_reader_decode))
        target_ttft.append(ttft - (parse_seconds - target_parse_total)
                           - (grounding_seconds - target_grounding_total))
        baseline_service.append(service)
        baseline_ttft.append(ttft)
        requests.append({
            'person_id': row['person_id'], 'question_id': row['question_id'],
            'baseline_service_seconds': service, 'baseline_ttft_seconds': ttft,
            'parse_tokens': parse_tokens, 'grounding_tokens': grounding_tokens,
            'reader_tokens': reader_tokens,
            'control_service_seconds': control_service[-1],
            'control_ttft_seconds': control_ttft[-1],
            'target_service_seconds': target_service[-1],
            'target_ttft_seconds': target_ttft[-1],
            'parse_projection': parse_projection,
            'grounding_projection': grounding_projection,
        })

    baseline = {'service': latency(baseline_service), 'ttft': latency(baseline_ttft)}
    control = {'service': latency(control_service), 'ttft': latency(control_ttft)}
    target = {'service': latency(target_service), 'ttft': latency(target_ttft)}
    total_decode_tokens = sum(decode_tokens.values())
    weight_record = gate0.get('decode_unique_w8_weight_bytes')
    if isinstance(weight_record, dict):
        weight_bytes = weight_record.get('median') or weight_record.get('max')
    else:
        weight_bytes = weight_record
    if weight_bytes is None:
        weight_bytes = gate0['accounting_totals']['parser'][
            'decode_unique_w8_weight_bytes_median']
    host_linear_seconds = total_decode_tokens * linear['host_linear_ms_per_token'] / 1000
    pim_linear_seconds = total_decode_tokens * linear['pim_linear_ms_per_token'] / 1000
    pim_actual_linear_seconds = total_decode_tokens * pim_linear_ms / 1000
    weight_stream_bytes = total_decode_tokens * int(weight_bytes)
    scale_contract = ({'group_size': args.w8a16_group_size,
                       'scale_bytes_per_group': 4,
                       'residual_bits': args.w8a16_residual_bits,
                       'residual_scale_bytes': 4}
                      if args.precision == 'w8a16' else {'scale_bytes_per_row': 4})
    decoder_residency = bank_resident_decoder_plan(
        decoder_matrices, StructuredControlConfig(
            token_test_cycles=args.pim_token_test_cycles), **scale_contract)
    matrix_costs_by_shape = {row['matrix_id']: row for row in matrix_costs}
    geniepim_padded_weight_bytes = 0
    for shape, count in linear['calls_per_token'].items():
        cost = matrix_costs_by_shape[shape]['cost']
        padded_rows = max(1, math.ceil(
            cost['pim_matrix_scalar_num_occupied_dram_rows_per_bank']))
        geniepim_padded_weight_bytes += padded_rows * 2048 * 128 * round(count)
    decoder_residency['geniepim_bank_padded_weight_bytes'] = (
        geniepim_padded_weight_bytes)
    decoder_residency['geniepim_padding_over_logical_weight'] = (
        geniepim_padded_weight_bytes / decoder_residency['logical_weight_bytes'])
    pool_residency = grammar_residency(
        int(math.ceil((geniepim_padded_weight_bytes
                       + decoder_residency['bank_padded_scale_bytes']
                       + decoder_residency['bank_padded_residual_bytes'])
                      * resident_fraction)),
        token_table_bytes=248320 * 2560 * 2,
        grammar_state_bytes=64 * 1024**2,
        scratch_bytes=256 * 1024**2,
        config=StructuredControlConfig(token_test_cycles=args.pim_token_test_cycles))
    output_elements_per_token = sum(
        int(shape.split('x')[0]) * count
        for shape, count in linear['calls_per_token'].items())
    linear_calls_per_token = sum(linear['calls_per_token'].values())
    host_control_service = control['service']['sum_seconds']
    target_service_total = target['service']['sum_seconds']
    result = {
        'schema': 'hyperpim-sc-closure-projection-v2',
        'requests': len(rows),
        'boundary': (
            'trace-driven projection: baseline service/TTFT are measured server wall clock; '
            'grammar/forward split comes from exact CPU grammar replay; PIM costs are analytical; '
            'W8A16 conversion, scale and BF16 broadcast costs are explicit; this is not a '
            'physical PIM measurement'),
        'inputs_sha256': {
            **provenance,
            str(gate0_path): digest(gate0_path), str(grammar_path): digest(grammar_path),
            str(matrix_path): digest(matrix_path),
        },
        'matrix_linear_cost': linear,
        'precision_contract': {
            'precision': args.precision,
            'weight_format': 'int8_bank_resident_single_copy',
            'activation_format': 'bf16' if args.precision == 'w8a16' else 'int8',
            'w8a16_group_size': args.w8a16_group_size if args.precision == 'w8a16' else None,
            'w8a16_residual_bits': (args.w8a16_residual_bits
                                    if args.precision == 'w8a16' else None),
            'fallback_weight_elements': args.fallback_weight_elements,
            'total_weight_elements': total_weight_elements,
            'resident_weight_fraction': resident_fraction,
            'host_bf16_fallback_fraction': fallback_fraction,
            'mixed_precision_model': mixed_precision,
            'boundary': ('W8A16 arithmetic is an analytical bank-local mixed-precision model; '
                         'functional GPU replay is not a PIM timing measurement'),
        },
        'gpu_nonlinear_profile': nonlinear,
        'grammar_forward_split': grammar_profiles,
        'grammar_fill_calibration': calibration,
        'pim_grammar_model': pim_grammar,
        'target_ms_per_token': {
            'structured': pim_structured_ms,
            'free_text_reader': pim_reader_ms,
        },
        'projected_token_cost_ms': {
            'host_linear': linear['host_linear_ms_per_token'],
            'pim_int8_linear': linear['pim_linear_ms_per_token'],
            'pim_mixed_precision_overhead': (
                resident_overhead_ms),
            'pim_residual_mac': (
                mixed_precision['residual_mac_ms_per_token'] * resident_fraction
                if mixed_precision else 0.0),
            'pim_resident_linear': resident_pim_linear_ms,
            'host_bf16_fallback_linear': host_fallback_linear_ms,
            'pim_linear': pim_linear_ms,
            'gpu_nonlinear': gpu_nonlinear_ms,
            'pim_grammar_scan': pim_grammar_ms,
            'dispatch': args.dispatch_ms_per_token,
        },
        'decode_token_accounting': {
            **decode_tokens,
            'total': total_decode_tokens,
            'resident_w8_weight_bytes': int(weight_bytes),
            'logical_weight_stream_bytes': weight_stream_bytes,
            'host_linear_seconds': host_linear_seconds,
            'pim_linear_seconds': pim_linear_seconds,
            'pim_actual_linear_seconds': pim_actual_linear_seconds,
            'pim_resident_w8_weight_bytes': int(int(weight_bytes) * resident_fraction),
            'pim_resident_residual_bytes': int(
                decoder_residency['bank_padded_residual_bytes'] * resident_fraction),
            'host_fallback_bf16_stream_bytes': int(
                total_decode_tokens * args.fallback_weight_elements * 2),
            'pim_resident_w8_stream_bytes': int(
                total_decode_tokens * int(weight_bytes) * resident_fraction),
            'pim_linear_with_mixed_precision_seconds': (
                total_decode_tokens * pim_linear_ms / 1000),
            'w8a16_group_scale_bytes': decoder_residency['bank_padded_scale_bytes']
            if args.precision == 'w8a16' else None,
            'w8a16_scale_groups': decoder_residency['logical_scale_groups']
            if args.precision == 'w8a16' else None,
            'w8a16_residual_bits': (args.w8a16_residual_bits
                                    if args.precision == 'w8a16' else None),
            'host_linear_effective_bandwidth_gbps': (
                int(weight_bytes) / (linear['host_linear_ms_per_token'] / 1000) / 1e9),
            'pim_linear_effective_bandwidth_gbps': (
                int(weight_bytes) / (linear['pim_linear_ms_per_token'] / 1000) / 1e9),
            'linear_calls_per_token': linear_calls_per_token,
            'dispatch_us_per_linear_call': (
                1000 * args.dispatch_ms_per_token / linear_calls_per_token),
            'pim_output_elements_per_token': output_elements_per_token,
            'pim_output_int32_bytes_per_token': output_elements_per_token * 4,
            'activation_broadcast_bytes_per_layer': 2560 * 2,
            'activation_broadcast_bytes_if_per_matrix': int(
                linear_calls_per_token) * 2560 * 2,
            'decoder_bank_residency': decoder_residency,
            'pool_residency': pool_residency,
        },
        'scenarios': {
            'baseline_measured': baseline,
            'host_control_lower_bound': {
                **control,
                'service_speedup': baseline['service']['sum_seconds']
                / control['service']['sum_seconds'],
                'ttft_speedup': baseline['ttft']['sum_seconds']
                / control['ttft']['sum_seconds'],
                'boundary': (
                    'counterfactual Host baseline that grants a zero-cost, exact grammar control '
                    'plane; it is more favorable to Host than any measured mask-cache fallback'),
            },
            'hyperpim_sc_target': {
                **target,
                'service_speedup': baseline['service']['sum_seconds']
                / target['service']['sum_seconds'],
                'ttft_speedup': baseline['ttft']['sum_seconds']
                / target['ttft']['sum_seconds'],
                'service_p95_speedup': baseline['service']['p95_seconds']
                / target['service']['p95_seconds'],
                'ttft_p95_speedup': baseline['ttft']['p95_seconds']
                / target['ttft']['p95_seconds'],
                'increment_over_host_control_lower_bound': {
                    'service_speedup': host_control_service / target_service_total,
                    'ttft_speedup': (control['ttft']['sum_seconds']
                                     / target['ttft']['sum_seconds']),
                    'service_p95_speedup': (control['service']['p95_seconds']
                                            / target['service']['p95_seconds']),
                    'ttft_p95_speedup': (control['ttft']['p95_seconds']
                                         / target['ttft']['p95_seconds']),
                },
            },
        },
        'requests_detail': requests,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({
        'requests': result['requests'],
        'matrix_linear_cost': result['matrix_linear_cost'],
        'gpu_nonlinear_profile': result['gpu_nonlinear_profile'],
        'grammar_forward_split': result['grammar_forward_split'],
        'pim_grammar_ms_per_token': pim_grammar_ms,
        'target_ms_per_token': result['target_ms_per_token'],
        'projected_token_cost_ms': result['projected_token_cost_ms'],
        'scenarios': {
            name: {metric: value[metric]['sum_seconds'] for metric in ('service', 'ttft')}
            for name, value in result['scenarios'].items()
        },
        'speedups': {name: {metric: value.get(f'{metric}_speedup')
                            for metric in ('service', 'ttft')}
                     for name, value in result['scenarios'].items()
                     if name != 'baseline_measured'},
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
