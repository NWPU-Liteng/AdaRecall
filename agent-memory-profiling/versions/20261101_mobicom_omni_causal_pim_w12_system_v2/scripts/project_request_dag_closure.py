"""Project a complete request DAG from measured Host rows and a PIM phase model.

The sensitivity scenario deliberately applies a modeled Linear speedup to all
decode and TTFT time while charging zero PIM boundary, placement, copy, fence,
refresh, and contention overhead. The uncalibrated phase model and this broad
eligibility assumption do not make the result a physical or mathematical upper
bound; it is a favorable sensitivity projection only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _finite_nonnegative(value, field):
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f'{field} must be finite and nonnegative')
    return float(value)


def _rows_by_question(run, name):
    if not run.get('complete') or run.get('evaluation_label_files_read') != []:
        raise ValueError(f'{name} must be complete and label isolated')
    rows = run.get('rows')
    if not isinstance(rows, list) or not rows:
        raise ValueError(f'{name} must contain nonempty rows')
    result = {}
    for row in rows:
        question_id = row.get('question_id')
        if not isinstance(question_id, str) or not question_id or question_id in result:
            raise ValueError(f'{name} has missing or duplicate question identity')
        result[question_id] = row
    return result


def _validate_host_rows(rows):
    for question_id, row in rows.items():
        values = {
            field: _finite_nonnegative(row.get(field), f'{question_id}.{field}')
            for field in ('service_seconds', 'ttft_seconds', 'prefill_seconds', 'decode_seconds')
        }
        if values['service_seconds'] + 1e-6 < values['ttft_seconds']:
            raise ValueError(f'{question_id}: service is shorter than TTFT')
        if not math.isclose(values['service_seconds'] - values['ttft_seconds'],
                            values['decode_seconds'], abs_tol=1e-3):
            raise ValueError(f'{question_id}: service, TTFT and decode do not partition')
        if values['ttft_seconds'] + 1e-6 < values['prefill_seconds']:
            raise ValueError(f'{question_id}: TTFT is shorter than prefill')


def _validate_functional_quality(host_rows, functional_run):
    functional_rows = _rows_by_question(functional_run, 'functional candidate')
    if set(host_rows) != set(functional_rows):
        raise ValueError('functional pair question identities differ')
    mismatches = []
    for question_id in host_rows:
        host, candidate = host_rows[question_id], functional_rows[question_id]
        if host.get('question_sha256') != candidate.get('question_sha256'):
            mismatches.append({'question_id': question_id, 'kind': 'question_identity'})
            continue
        if host.get('answer_token_ids') != candidate.get('answer_token_ids'):
            mismatches.append({'question_id': question_id, 'kind': 'answer_tokens'})
        if host.get('answer') != candidate.get('answer'):
            mismatches.append({'question_id': question_id, 'kind': 'answer_text'})
        if host.get('stop_reason') != candidate.get('stop_reason'):
            mismatches.append({'question_id': question_id, 'kind': 'stop_reason'})
    dispatch = functional_run.get('lm_head_functional_dispatch', {})
    if dispatch.get('physical_pim_executed') is not False:
        raise ValueError('functional candidate must not be mistaken for physical PIM')
    return {
        'quality_contract_pass': not mismatches,
        'mismatches': mismatches,
        'physical_pim_executed': False,
        'latency_comparable': False,
        'boundary': 'GPU functional emulation validates output only; it is not a PIM latency measurement',
    }


def _nearest_rank(values, quantile):
    ordered = sorted(values)
    if not ordered:
        raise ValueError('quantile requires nonempty values')
    index = max(1, math.ceil(quantile * len(ordered))) - 1
    return ordered[index]


def _replay_speedup(replay):
    try:
        speedup = float(replay['aggregate']['linear_phase_speedup_warm'])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError('replay must contain modeled warm Linear speedup') from error
    if not math.isfinite(speedup) or speedup <= 1:
        raise ValueError('modeled warm Linear speedup must exceed one')
    return speedup


def project_request_dag(host_run, replay, *, functional_run=None,
                        linear_fraction_of_decode=1.0, ttft_fraction=1.0,
                        pim_boundary_seconds=0.0, pim_placement_seconds=0.0,
                        pim_copy_seconds=0.0, pim_fence_seconds=0.0,
                        pim_refresh_seconds=0.0, contention_multiplier=1.0,
                        target_speedup=1.05):
    host_rows = _rows_by_question(host_run, 'Host run')
    _validate_host_rows(host_rows)
    speedup = _replay_speedup(replay)
    fractions = (linear_fraction_of_decode, ttft_fraction)
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in fractions):
        raise ValueError('decode and TTFT eligible fractions must be in [0, 1]')
    overhead = {
        'pim_boundary_seconds': _finite_nonnegative(pim_boundary_seconds, 'PIM boundary'),
        'pim_placement_seconds': _finite_nonnegative(pim_placement_seconds, 'PIM placement'),
        'pim_copy_seconds': _finite_nonnegative(pim_copy_seconds, 'PIM copy'),
        'pim_fence_seconds': _finite_nonnegative(pim_fence_seconds, 'PIM fence'),
        'pim_refresh_seconds': _finite_nonnegative(pim_refresh_seconds, 'PIM refresh'),
    }
    if not math.isfinite(contention_multiplier) or contention_multiplier < 1:
        raise ValueError('contention multiplier must be finite and at least one')
    if not math.isfinite(target_speedup) or target_speedup <= 1:
        raise ValueError('target speedup must exceed one')

    functional = None
    if functional_run is not None:
        functional = _validate_functional_quality(host_rows, functional_run)

    rows = []
    for question_id, host in host_rows.items():
        service = float(host['service_seconds'])
        ttft = float(host['ttft_seconds'])
        prefill = float(host['prefill_seconds'])
        decode = float(host['decode_seconds'])
        eligible_decode = decode * linear_fraction_of_decode
        residual_decode = decode - eligible_decode
        pim_compute = eligible_decode / speedup * contention_multiplier
        charged_overhead = sum(overhead.values())
        candidate_decode = residual_decode + pim_compute + charged_overhead
        candidate_service = service - decode + candidate_decode
        eligible_ttft = ttft * ttft_fraction
        candidate_ttft = (ttft - eligible_ttft + eligible_ttft / speedup
                          + charged_overhead)
        rows.append({
            'question_id': question_id,
            'host_dag': {
                'query_and_evidence': service - prefill - decode,
                'reader_prefill': prefill,
                'reader_decode': decode,
                'critical_path': ['query_and_evidence', 'reader_prefill', 'reader_decode'],
            },
            'candidate_dag': {
                'query_and_evidence': service - prefill - decode,
                'reader_prefill': prefill,
                'reader_decode_residual': residual_decode,
                'pim_compute': pim_compute,
                **overhead,
                'critical_path': ['query_and_evidence', 'reader_prefill',
                                  'reader_decode_residual', 'pim_compute',
                                  'pim_boundary_seconds', 'pim_placement_seconds',
                                  'pim_copy_seconds', 'pim_fence_seconds',
                                  'pim_refresh_seconds'],
            },
            'host_service_seconds': service,
            'candidate_service_seconds': candidate_service,
            'host_ttft_seconds': ttft,
            'candidate_ttft_seconds': candidate_ttft,
            'decode_eligible_fraction': linear_fraction_of_decode,
            'ttft_eligible_fraction': ttft_fraction,
        })

    host_service = [row['host_service_seconds'] for row in rows]
    candidate_service = [row['candidate_service_seconds'] for row in rows]
    host_ttft = [row['host_ttft_seconds'] for row in rows]
    candidate_ttft = [row['candidate_ttft_seconds'] for row in rows]
    host_p95 = _nearest_rank(host_service, .95)
    candidate_p95 = _nearest_rank(candidate_service, .95)
    host_ttft_p95 = _nearest_rank(host_ttft, .95)
    candidate_ttft_p95 = _nearest_rank(candidate_ttft, .95)
    service_speedup = sum(host_service) / sum(candidate_service)
    ttft_speedup = sum(host_ttft) / sum(candidate_ttft)
    quality_pass = functional is None or functional['quality_contract_pass']
    gates = {
        'service_speedup_pass': service_speedup >= target_speedup,
        'ttft_speedup_pass': ttft_speedup >= target_speedup,
        'service_p95_not_worse': candidate_p95 <= host_p95,
        'ttft_p95_not_worse': candidate_ttft_p95 <= host_ttft_p95,
        'quality_contract_pass': quality_pass,
    }
    return {
        'schema_version': 'request-dag-pim-closure-v2',
        'projection_kind': 'optimistic_sensitivity_projection' if not any(overhead.values()) and (
            linear_fraction_of_decode == 1 and ttft_fraction == 1 and contention_multiplier == 1
        ) else 'request_level_projection',
        'projection_interpretation': (
            'favorable sensitivity scenario, not a physical measurement or strict bound; '
            'the phase model is uncalibrated and decode/TTFT eligibility is assumed'
        ),
        'measurement_status': 'not_measured',
        'modeled_warm_linear_speedup': speedup,
        'optimistic_assumptions': {
            'all_decode_is_eligible': linear_fraction_of_decode == 1,
            'all_ttft_is_eligible': ttft_fraction == 1,
            'zero_pim_boundary_and_transfer_cost': not any(overhead.values()),
            'no_shared_memory_contention': contention_multiplier == 1,
            'quality_source': 'functional candidate exact output comparison' if functional else 'not supplied',
        },
        'request_count': len(rows),
        'host_totals': {
            'service_seconds': sum(host_service), 'ttft_seconds': sum(host_ttft),
            'service_p95_seconds': host_p95, 'ttft_p95_seconds': host_ttft_p95,
        },
        'candidate_totals': {
            'service_seconds': sum(candidate_service), 'ttft_seconds': sum(candidate_ttft),
            'service_p95_seconds': candidate_p95, 'ttft_p95_seconds': candidate_ttft_p95,
        },
        'speedups': {'service': service_speedup, 'ttft': ttft_speedup,
                     'service_p95': host_p95 / candidate_p95,
                     'ttft_p95': host_ttft_p95 / candidate_ttft_p95},
        'gates': gates,
        'positive_e2e_gate_pass': all(gates.values()),
        'stop_reason': None if all(gates.values()) else 'positive_e2e_gate_not_met_in_projection',
        'functional_quality': functional,
        'unmodeled_or_zeroed_for_sensitivity_projection': [
            'PIM dispatch and Host/PIM boundary', 'weight placement and cold load',
            'activation/output copies', 'fence and refresh',
            'shared LPDDR contention', 'residual non-Linear Reader work',
            'device calibration, power and thermal effects'],
        'rows': rows,
        'boundary': (
            'request DAG projection over measured Host wall-clock rows, an uncalibrated '
            'PIM phase model, and favorable eligibility assumptions; this is '
            'not a strict upper bound, no physical PIM latency or device measurement, and '
            'no claim of positive E2E speedup'),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host-run', type=Path, required=True)
    parser.add_argument('--replay', type=Path, required=True)
    parser.add_argument('--functional-run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--linear-fraction-of-decode', type=float, default=1.0)
    parser.add_argument('--ttft-fraction', type=float, default=1.0)
    parser.add_argument('--contention-multiplier', type=float, default=1.0)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('fresh output required')
    host_run = json.loads(args.host_run.read_text())
    replay = json.loads(args.replay.read_text())
    functional = json.loads(args.functional_run.read_text()) if args.functional_run else None
    result = project_request_dag(
        host_run, replay, functional_run=functional,
        linear_fraction_of_decode=args.linear_fraction_of_decode,
        ttft_fraction=args.ttft_fraction,
        contention_multiplier=args.contention_multiplier)
    result['host_run_sha256'] = hashlib.sha256(args.host_run.read_bytes()).hexdigest()
    result['replay_sha256'] = hashlib.sha256(args.replay.read_bytes()).hexdigest()
    result['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if args.functional_run:
        result['functional_run_sha256'] = hashlib.sha256(
            args.functional_run.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'speedups': result['speedups'], 'gates': result['gates'],
                      'positive_e2e_gate_pass': result['positive_e2e_gate_pass']},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
