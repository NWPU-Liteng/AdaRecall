"""Conditional full-decode cost accounting, never a full execution trace."""
from __future__ import annotations

from collections import Counter
import math


def decode_calls(row):
    tokens = row['answer_token_ids']
    if not tokens or row['stop_reason'] not in {'eos', 'max_new_tokens'}:
        raise ValueError('complete nonempty greedy answer and stop reason required')
    return len(tokens) - 1


def same_platform_linear_lifecycle(run, candidates, selected, placement_ns):
    rows = run['rows']
    identities = [row['question_id'] for row in rows]
    if (not run.get('complete') or run.get('evaluation_label_files_read') != [] or not rows
            or len(identities) != len(set(identities))
            or len({row['person_id'] for row in rows}) != 1
            or not candidates or not set(selected) <= set(candidates)
            or not math.isfinite(placement_ns) or placement_ns < 0):
        raise ValueError('one complete unique label-isolated user stream and frozen finite placement required')
    steps = [decode_calls(row) for row in rows]
    total_steps = sum(steps)
    if any(item['calls'] != total_steps for item in candidates.values()):
        raise ValueError('every mapped matrix must have the complete observed per-token decode call count')
    for item in candidates.values():
        if any(not math.isfinite(item[name]) or item[name] < 0
               for name in ('host_per_call_ns', 'pim_per_call_ns')):
            raise ValueError('finite same-platform nonnegative matrix costs required')
    host_per_step = sum(item['host_per_call_ns'] for item in candidates.values())
    hybrid_per_step = sum(item['pim_per_call_ns'] if name in selected else item['host_per_call_ns']
                          for name, item in candidates.items())
    requests = [{'question_id': row['question_id'], 'person_id': row['person_id'],
                 'observed_decode_invocations': count,
                 'known_linear_host_ns': count * host_per_step,
                 'known_linear_warm_hybrid_ns': count * hybrid_per_step,
                 'known_linear_persistent_stream_ns_with_minimum_placement': count * hybrid_per_step + (
                     placement_ns if index == 0 else 0),
                 'known_linear_cold_request_ns_with_minimum_placement': count * hybrid_per_step + placement_ns,
                 'full_request_host_ns': None, 'full_request_hybrid_ns': None}
                for index, (row, count) in enumerate(zip(rows, steps))]
    return {'requests': requests, 'decode_invocations': total_steps,
            'complete_aggregate_matrix_call_check': True,
            'known_linear_host_ns': total_steps * host_per_step,
            'known_linear_warm_hybrid_ns': total_steps * hybrid_per_step,
            'persistent_stream_minimum_layout_writes': 1,
            'cold_per_request_minimum_layout_writes': len(rows),
            'persistent_stream_minimum_layout_write_ns': placement_ns,
            'cold_per_request_minimum_layout_write_ns': len(rows) * placement_ns,
            'unmapped_work_ns': None,
            'boundary': 'same frozen architecture scalar GEMV costs, retrospective observed token counts only; placement not selected with future lengths; minimum writes are not full cold costs; prefill, quantization, FP32 scaling, conversions, nonlinear kernels, fences and contention remain unknown, never zero'}


def lifecycle_for_user(run, schedules, costs, placement, copy_range):
    rows = run['rows']
    if len(rows) != len(schedules) or not run['complete'] or run['evaluation_label_files_read'] != []:
        raise ValueError('complete label-isolated user stream and first-step schedules required')
    expected_calls = Counter()
    expected_shapes = {}
    totals = Counter()
    requests = []
    for row, schedule in zip(rows, schedules):
        if row['question_id'] != schedule['question_id'] or row['person_id'] != schedule['person_id']:
            raise ValueError('per-user order and request identity changed')
        steps = decode_calls(row)
        first_step = schedule['known_linear_subgraph']
        request = {'question_id': row['question_id'], 'observed_decode_invocations': steps}
        for field in ('host_min_cycles', 'host_max_cycles', 'hybrid_min_cycles', 'hybrid_max_cycles',
                      'host_external_bytes', 'hybrid_external_bytes', 'pim_calls'):
            request[field] = steps * first_step[field]
            totals[field] += request[field]
        modeled = [event for event in schedule['events'] if event['kind'] == 'modeled_linear']
        for event in modeled:
            identity = event['parameter_identity']
            aliases = identity.split('||')
            cost = costs[identity]
            expected_calls[identity] += steps
            expected_shapes[identity] = (cost['inputs'], cost['outputs'], cost['element_bytes'])
            if event['route'] != ('pim' if identity in placement else 'host'):
                raise ValueError('lifecycle may not change frozen placement')
            if not all(alias.startswith('parameter:') and alias.endswith('.weight') for alias in aliases):
                raise ValueError('unsupported parameter/module identity')
        requests.append(request)
    observed_calls = Counter()
    for operator in run['operator_trace']['operators']:
        if operator['phase'] != 'decode':
            continue
        matches = [identity for identity in expected_shapes
                   if 'parameter:' + operator['name'] + '.weight' in identity.split('||')]
        if len(matches) != 1 or operator['rows'] != 1:
            raise ValueError('complete decode Linear trace must match first-step physical identities')
        identity = matches[0]
        if ((operator['in_features'], operator['out_features'], operator['element_bytes'])
                != expected_shapes[identity] or operator['calls'] < 1
                or operator['weight_bytes'] != costs[identity]['resident_bytes']):
            raise ValueError('decode shape, weight bytes or call count changed')
        observed_calls[identity] += operator['calls']
    if observed_calls != expected_calls:
        raise ValueError('full decode aggregate counts do not certify repeated first-step Linear signature')
    host = totals['host_min_cycles']
    warm = totals['hybrid_max_cycles']
    cold_stream = warm + copy_range['max_cycles']
    cold_request = warm + len(rows) * copy_range['max_cycles']
    steps = sum(request['observed_decode_invocations'] for request in requests)
    per_step_gain = (host - warm) / steps if steps else 0
    return {'person_id': rows[0]['person_id'], 'requests': len(rows),
        'decode_invocations': steps, 'aggregate_linear_call_identity_check': 'pass',
        'linear_calls': sum(observed_calls.values()), 'totals': dict(totals), 'per_request': requests,
        'placement_copies_persistent_stream': 1, 'placement_copies_cold_per_request': len(rows),
        'persistent_stream_hybrid_max_cycles': cold_stream,
        'cold_per_request_hybrid_max_cycles': cold_request,
        'warm_known_linear_speedup_lower_scenario': host / warm if warm else None,
        'persistent_stream_known_linear_speedup_lower_scenario': host / cold_stream if cold_stream else None,
        'cold_per_request_known_linear_speedup_lower_scenario': host / cold_request if cold_request else None,
        'break_even_decode_invocations': math.ceil(copy_range['max_cycles'] / per_step_gain)
                                       if per_step_gain > 0 else None,
        'unmodeled': ['later-token non-Linear execution/dependencies and growing attention',
                      'opaque inclusive durations', 'prefill/layout conversion/migration',
                      'refresh during unknown work and interference from other SoC clients'],
        'migration_policy': 'immutable placement pinned within each stream; zero planned migration, not measured absence of migration cost',
        'refresh_policy': 'each known primitive already includes worst-start-phase refresh; never extrapolate unknown kernels as zero time',
        'boundary': 'full-decode aggregate Linear identities/counts validated; first-step Linear costs repeated conditionally; not full-token DAG replay or complete-system latency'}


def normalized_service_sensitivity(decode_fraction, linear_speedup, linear_decode_fraction):
    values = (decode_fraction, linear_speedup, linear_decode_fraction)
    if (any(not math.isfinite(value) for value in values) or not 0 <= decode_fraction <= 1
            or not 0 <= linear_decode_fraction <= 1 or linear_speedup <= 0):
        raise ValueError('finite fractions and positive modeled Linear ratio required')
    normalized = 1 - decode_fraction + decode_fraction * (
        1 - linear_decode_fraction + linear_decode_fraction / linear_speedup)
    return {'assumed_linear_fraction_of_decode': linear_decode_fraction,
        'normalized_service_ratio': normalized, 'conditional_service_speedup': 1 / normalized,
        'interpretation': 'dimensionless Amdahl sensitivity using server stage composition; neither seconds-plus-cycles arithmetic nor a calibrated LPDDR full-system prediction'}
