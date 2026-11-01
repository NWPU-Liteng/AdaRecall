"""Audit a Host/full versus adaptive-frontier serving pair."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def nearest_rank(values, quantile):
    if not values or not 0. <= quantile <= 1.:
        raise ValueError('nonempty values and quantile in [0, 1] required')
    ordered = sorted(float(value) for value in values)
    index = max(0, min(len(ordered) - 1,
                       int(math.ceil(quantile * len(ordered))) - 1))
    return ordered[index]


def latency_summary(values):
    if not values or any(not math.isfinite(float(value)) or float(value) < 0 for value in values):
        raise ValueError('finite nonnegative latency values required')
    values = [float(value) for value in values]
    return {
        'count': len(values),
        'mean_seconds': sum(values) / len(values),
        'p50_seconds': nearest_rank(values, .5),
        'p95_seconds': nearest_rank(values, .95),
        'max_seconds': max(values),
        'sum_seconds': sum(values),
    }


def load_run(path):
    payload = json.loads(Path(path).read_text())
    if payload.get('complete') is not True or not isinstance(payload.get('rows'), list):
        raise ValueError(f'complete run with rows required: {path}')
    rows = payload['rows']
    identities = [row.get('question_id') for row in rows]
    if (not identities or any(not isinstance(identity, str) or not identity for identity in identities)
            or len(set(identities)) != len(identities)):
        raise ValueError(f'unique question identities required: {path}')
    if payload.get('evaluation_label_files_read', []) != []:
        raise ValueError(f'serving labels must be isolated: {path}')
    return payload


def _stage_seconds(row, phase):
    events = {event.get('phase'): float(event.get('seconds', 0.))
              for event in row.get('execution_events', [])}
    value = events.get(phase)
    if value is None or not math.isfinite(value) or value < 0:
        raise ValueError(f'missing or invalid {phase} event for {row["question_id"]}')
    return value


def _frontier_summary(row):
    frontier = row.get('grounding_frontier')
    levels = row.get('grounding_frontier_levels', [])
    if not isinstance(frontier, dict):
        return None
    selected_level = row.get('proof_plan', {}).get('grounding', {}).get('frontier_selected_level')
    selected = next((level for level in levels if level.get('level') == selected_level), None)
    first = levels[0] if levels else None
    return {
        'full_candidate_count': int(frontier.get('state_count', 0)),
        'full_packed_bytes': int(frontier.get('packed_bytes', 0)),
        'first_candidate_count': (first.get('candidate_count') if first else None),
        'selected_candidate_count': (selected.get('candidate_count') if selected else None),
        'selected_level': selected_level,
        'first_lost_source_provenance_count': (first.get('lost_source_provenance_count', 0)
                                               if first else 0),
        'first_lost_source_relation_count': (first.get('lost_source_relation_count', 0)
                                             if first else 0),
        'first_lost_binding_key_count': (first.get('lost_binding_key_count', 0)
                                         if first else 0),
        'attempt_count': len(row.get('proof_plan', {}).get('grounding', {}).get(
            'frontier_attempts', [])),
    }


def _row_summary(row):
    service = float(row['service_seconds'])
    ttft = row.get('ttft_seconds')
    if not math.isfinite(service) or service <= 0:
        raise ValueError(f'invalid service time for {row["question_id"]}')
    if ttft is not None and (not math.isfinite(float(ttft)) or not 0 <= float(ttft) <= service):
        raise ValueError(f'invalid TTFT for {row["question_id"]}')
    return {
        'question_id': row['question_id'],
        'question_sha256': row.get('question_sha256'),
        'answer_sha256': row.get('answer_sha256'),
        'answer_token_ids': row.get('answer_token_ids'),
        'stop_reason': row.get('stop_reason'),
        'input_tokens': row.get('input_tokens'),
        'service_seconds': service,
        'ttft_seconds': None if ttft is None else float(ttft),
        'prefill_seconds': float(row.get('prefill_seconds', 0.)),
        'decode_seconds': float(row.get('decode_seconds', 0.)),
        'grounding_seconds': _stage_seconds(row, 'source_grounding'),
        'frontier': _frontier_summary(row),
        'query_plan_sha256': hashlib.sha256(json.dumps({
            'demands': row.get('proof_plan', {}).get('demands', []),
            'relations': row.get('proof_plan', {}).get('relations', []),
            'query_contract': row.get('proof_plan', {}).get('query_contract'),
        }, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
    }


def _pair_mismatches(reference, candidate):
    mismatches = []
    for field in ('question_sha256', 'query_plan_sha256'):
        if reference.get(field) != candidate.get(field):
            mismatches.append(field)
    reference_frontier = reference.get('frontier')
    candidate_frontier = candidate.get('frontier')
    if reference_frontier and candidate_frontier:
        for field in ('full_candidate_count', 'full_packed_bytes'):
            if reference_frontier.get(field) != candidate_frontier.get(field):
                mismatches.append('frontier_' + field)
    if reference.get('stop_reason') != candidate.get('stop_reason'):
        mismatches.append('stop_reason')
    if reference.get('answer_token_ids') != candidate.get('answer_token_ids'):
        mismatches.append('answer_token_ids')
    if reference.get('answer_sha256') != candidate.get('answer_sha256'):
        mismatches.append('answer_sha256')
    return mismatches


def _protocol_identity(run):
    fields = (
        'prepared_audit_sha256', 'cohort_sha256', 'index_sha256', 'reader_model',
        'embedder_model', 'reader_processor_input_budget', 'total_text_tokens',
        'max_images', 'max_new_tokens', 'query_contract_enabled',
    )
    return {field: run.get(field) for field in fields}


def audit_pair(reference_path, candidate_path, *, min_speedup=1.05,
               min_ttft_speedup=1.05, min_requests=4):
    if not math.isfinite(min_speedup) or min_speedup <= 1:
        raise ValueError('min_speedup must be finite and greater than one')
    if not math.isfinite(min_ttft_speedup) or min_ttft_speedup <= 1:
        raise ValueError('min_ttft_speedup must be finite and greater than one')
    if type(min_requests) is not int or min_requests < 1:
        raise ValueError('min_requests must be a positive integer')
    reference_run = load_run(reference_path)
    candidate_run = load_run(candidate_path)
    reference_protocol = _protocol_identity(reference_run)
    candidate_protocol = _protocol_identity(candidate_run)
    protocol_mismatches = [field for field in reference_protocol
                           if reference_protocol[field] != candidate_protocol[field]]
    reference = {_row_summary(row)['question_id']: _row_summary(row)
                 for row in reference_run['rows']}
    candidate = {_row_summary(row)['question_id']: _row_summary(row)
                 for row in candidate_run['rows']}
    if set(reference) != set(candidate):
        raise ValueError('paired runs must contain exactly the same questions')
    pairs = []
    for question_id in reference:
        before, after = reference[question_id], candidate[question_id]
        pairs.append({'question_id': question_id,
                      'mismatches': _pair_mismatches(before, after),
                      'reference': before, 'candidate': after,
                      'input_token_delta': (None if before['input_tokens'] is None
                                            or after['input_tokens'] is None else
                                            after['input_tokens'] - before['input_tokens']),
                      'service_speedup': (before['service_seconds'] / after['service_seconds']
                                          if after['service_seconds'] > 0 else None),
                      'ttft_speedup': (before['ttft_seconds'] / after['ttft_seconds']
                                       if before['ttft_seconds'] is not None
                                       and after['ttft_seconds'] is not None
                                       and after['ttft_seconds'] > 0 else None)})
    reference_service = latency_summary([item['reference']['service_seconds'] for item in pairs])
    candidate_service = latency_summary([item['candidate']['service_seconds'] for item in pairs])
    reference_ttft_values = [item['reference']['ttft_seconds'] for item in pairs
                             if item['reference']['ttft_seconds'] is not None]
    candidate_ttft_values = [item['candidate']['ttft_seconds'] for item in pairs
                             if item['candidate']['ttft_seconds'] is not None]
    reference_ttft = (latency_summary(reference_ttft_values)
                      if len(reference_ttft_values) == len(pairs) else None)
    candidate_ttft = (latency_summary(candidate_ttft_values)
                      if len(candidate_ttft_values) == len(pairs) else None)
    mismatches = [item for item in pairs if item['mismatches']]
    service_p50_speedup = reference_service['p50_seconds'] / candidate_service['p50_seconds']
    service_p95_speedup = reference_service['p95_seconds'] / candidate_service['p95_seconds']
    ttft_p50_speedup = (reference_ttft['p50_seconds'] / candidate_ttft['p50_seconds']
                        if reference_ttft and candidate_ttft else None)
    frontier_rows = [item['candidate']['frontier'] for item in pairs
                     if item['candidate']['frontier'] is not None]
    risk = {
        'requests_with_lost_source_provenance': sum(
            item['first_lost_source_provenance_count'] > 0 for item in frontier_rows),
        'requests_with_lost_source_relations': sum(
            item['first_lost_source_relation_count'] > 0 for item in frontier_rows),
        'requests_with_lost_binding_keys': sum(
            item['first_lost_binding_key_count'] > 0 for item in frontier_rows),
        'requests_with_adaptive_attempts': sum(item['attempt_count'] > 1 for item in frontier_rows),
    }
    quality_gate = not mismatches and all(
        item['candidate']['stop_reason'] == 'eos' for item in pairs)
    latency_gate = (service_p50_speedup >= min_speedup
                    and service_p95_speedup >= min_speedup
                    and (ttft_p50_speedup is None or ttft_p50_speedup >= min_ttft_speedup))
    complete_frontier_gate = not frontier_rows or all(
        item['selected_level'] is not None or item['attempt_count'] == 0
        for item in frontier_rows)
    request_count_gate = len(pairs) >= min_requests
    return {
        'schema': 'adaptive-frontier-pair-audit-v1',
        'reference_path': str(reference_path),
        'candidate_path': str(candidate_path),
        'request_count': len(pairs),
        'minimum_request_count': min_requests,
        'request_count_gate': 'pass' if request_count_gate else 'failed_insufficient_pair_requests',
        'protocol_mismatches': protocol_mismatches,
        'protocol_gate': 'pass' if not protocol_mismatches else 'failed_protocol_mismatch',
        'pair_mismatches': mismatches,
        'quality_gate': 'pass' if quality_gate else 'failed_exact_output_contract',
        'latency_gate': 'pass' if latency_gate else 'failed_minimum_e2e_speedup',
        'frontier_execution_gate': 'pass' if complete_frontier_gate else 'failed_missing_selected_level',
        'promotion_decision': ('pass' if not protocol_mismatches and quality_gate and latency_gate and complete_frontier_gate
                               and request_count_gate
                               else 'reject'),
        'reference_service': reference_service,
        'candidate_service': candidate_service,
        'reference_ttft': reference_ttft,
        'candidate_ttft': candidate_ttft,
        'measured_e2e_speedup': reference_service['sum_seconds'] / candidate_service['sum_seconds'],
        'service_p50_speedup': service_p50_speedup,
        'service_p95_speedup': service_p95_speedup,
        'ttft_p50_speedup': ttft_p50_speedup,
        'frontier_risk': risk,
        'pairs': pairs,
        'boundary': 'paired server traces only; exact-output quality contract is conservative and no PIM/device claim is made',
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--min-speedup', type=float, default=1.05)
    parser.add_argument('--min-ttft-speedup', type=float, default=1.05)
    parser.add_argument('--min-requests', type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('fresh output required')
    result = audit_pair(args.reference, args.candidate,
                        min_speedup=args.min_speedup,
                        min_ttft_speedup=args.min_ttft_speedup,
                        min_requests=args.min_requests)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in (
        'request_count', 'request_count_gate', 'protocol_gate', 'protocol_mismatches',
        'quality_gate', 'latency_gate', 'promotion_decision',
        'measured_e2e_speedup', 'service_p50_speedup', 'service_p95_speedup',
        'ttft_p50_speedup')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
