"""Audit stored service traces and freeze a small label-free profiling cohort."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics


def _percentile(values, fraction):
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def load_rows(paths):
    rows = []
    for path in paths:
        payload = json.loads(Path(path).read_text())
        if not payload.get('complete'):
            raise ValueError(f'incomplete run: {path}')
        if payload.get('evaluation_label_files_read'):
            raise ValueError(f'run accessed evaluation labels: {path}')
        for row in payload.get('rows', []):
            if not row.get('question_id') or not math.isfinite(float(row['service_seconds'])):
                raise ValueError(f'invalid trace row in {path}')
            row = dict(row)
            row['_trace_path'] = str(path)
            rows.append(row)
    identities = [(row.get('person_id'), row['question_id']) for row in rows]
    if len(set(identities)) != len(identities):
        raise ValueError('duplicate user/question identities across input runs')
    return rows


def _event(row, phase):
    return next((event for event in row.get('execution_events', [])
                 if event.get('phase') == phase), {})


def _structured_call(row, key):
    call = row.get('proof_plan', {}).get(key, {})
    timing = call.get('timing', {})
    return {
        'total_seconds': call.get('seconds'),
        'input_tokens': timing.get('structured_input_tokens'),
        'prepare_seconds': timing.get('prepare_seconds'),
        'grammar_compile_seconds': timing.get('grammar_compile_seconds'),
        'prefill_seconds': timing.get('reader_prefill_seconds'),
        'decode_seconds': timing.get('reader_decode_seconds'),
        'first_token_seconds': timing.get('first_token_seconds'),
        'postprocess_seconds': timing.get('postprocess_seconds'),
    }


def _row_profile(row):
    retrieval_event = _event(row, 'query_parse_and_retrieval')
    grounding_event = _event(row, 'source_grounding')
    query_call = _structured_call(row, 'query_parse')
    grounding_call = _structured_call(row, 'grounding')
    retrieval_total = retrieval_event.get('seconds')
    grounding_total = grounding_event.get('seconds')
    grounding_attempts = row.get('proof_plan', {}).get('grounding', {}).get('frontier_attempts', [])
    return {
        'person_id': str(row.get('person_id')),
        'question_id': row['question_id'],
        'service_seconds': row['service_seconds'],
        'ttft_seconds': row.get('ttft_seconds'),
        'query_parse_and_retrieval_seconds': retrieval_total,
        'query_parser_reader_seconds': query_call['total_seconds'],
        'retrieval_nonreader_residual_seconds': (
            max(0.0, retrieval_total - query_call['total_seconds'])
            if retrieval_total is not None and query_call['total_seconds'] is not None else None),
        'query_parser_call': query_call,
        'grounding_stage_seconds': grounding_total,
        'grounding_reader_last_call_seconds': grounding_call['total_seconds'],
        'grounding_attempt_count': len(grounding_attempts),
        'grounding_call_count_exact': (len(grounding_attempts) == 1),
        'grounding_nonreader_residual_lower_bound_seconds': (
            max(0.0, grounding_total - grounding_call['total_seconds'])
            if grounding_total is not None and grounding_call['total_seconds'] is not None else None),
        'grounding_call': grounding_call,
        'candidate_count': row.get('proof_plan', {}).get('grounding', {}).get('candidate_count'),
        'demand_count': row.get('proof_plan', {}).get('retrieval', {}).get('demand_count'),
        'materialized_image_count': len(row.get('images', [])),
        'input_tokens': row.get('input_tokens'),
        'trace_path': row['_trace_path'],
    }


def _select_profile_rows(rows, per_user=3):
    selected = []
    for person_id in sorted({str(row.get('person_id')) for row in rows}):
        user_rows = [row for row in rows if str(row.get('person_id')) == person_id]
        if len(user_rows) < per_user:
            raise ValueError(f'user {person_id} has too few rows for a profiling cohort')
        grounding_values = [float(_event(row, 'source_grounding').get('seconds', 0.0))
                            for row in user_rows]
        targets = [_percentile(grounding_values, value)
                   for value in (0.25, 0.50, 0.75)[:per_user]]
        chosen = []
        for target in targets:
            candidate = min((row for row in user_rows if row not in chosen),
                            key=lambda row: abs(float(_event(row, 'source_grounding').get('seconds', 0.0))
                                                - target))
            chosen.append(candidate)
        if any(row.get('images') for row in user_rows) and not any(row.get('images') for row in chosen):
            image_row = min((row for row in user_rows if row.get('images')),
                            key=lambda row: abs(float(_event(row, 'source_grounding').get('seconds', 0.0))
                                                - _percentile(grounding_values, 0.50)))
            replace_index = min(range(len(chosen)),
                                key=lambda index: abs(float(_event(chosen[index], 'source_grounding').get('seconds', 0.0))
                                                      - _percentile(grounding_values, 0.50)))
            chosen[replace_index] = image_row
        selected.extend(chosen)
    return selected


def audit(paths, *, expected_requests=200, per_user=3):
    rows = load_rows(paths)
    if len(rows) != expected_requests:
        raise ValueError(f'expected {expected_requests} unique requests, got {len(rows)}')
    profiles = [_row_profile(row) for row in rows]
    profiling_rows = _select_profile_rows(rows, per_user=per_user)
    selected_profiles = [_row_profile(row) for row in profiling_rows]

    def total(field):
        values = [profile[field] for profile in profiles if profile[field] is not None]
        return sum(values) if values else None

    grounding_attempt_counts = [profile['grounding_attempt_count'] for profile in profiles]
    result = {
        'schema': 'proof-query-opportunity-audit-v1',
        'requests': len(rows),
        'evaluation_labels_read': False,
        'source_trace_rows': len(profiles),
        'aggregate_seconds': {
            'service': total('service_seconds'),
            'query_parse_and_retrieval': total('query_parse_and_retrieval_seconds'),
            'query_parser_reader': total('query_parser_reader_seconds'),
            'source_grounding_stage': total('grounding_stage_seconds'),
            'final_recorded_grounding_reader_calls': total('grounding_reader_last_call_seconds'),
        },
        'grounding_attempt_count': {
            'min': min(grounding_attempt_counts),
            'p50': statistics.median(grounding_attempt_counts),
            'max': max(grounding_attempt_counts),
            'single_attempt_request_fraction': sum(count == 1 for count in grounding_attempt_counts) / len(rows),
        },
        'profiling_cohort_rule': (
            f'{per_user} requests per development user nearest to grounding-stage p25/p50/p75; '
            'if no image request is selected for a user, replace its median-nearest row with the nearest image request'),
        'profiling_question_ids': [
            {'person_id': row['person_id'], 'question_id': row['question_id']}
            for row in selected_profiles],
        'profiling_profiles': selected_profiles,
        'rows': profiles,
        'boundary': (
            'existing trace audit only; old runs lack prepare/prefill/decode split inside structured calls; '
            'grounding non-reader residual is a lower bound when multiple attempts occurred; '
            'no PIM execution, quality change, or speedup claim'),
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('runs', nargs='+', type=Path)
    parser.add_argument('--expected-requests', type=int, default=200)
    parser.add_argument('--per-user', type=int, default=3)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.runs, expected_requests=args.expected_requests, per_user=args.per_user)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in (
        'schema', 'requests', 'evaluation_labels_read', 'aggregate_seconds',
        'grounding_attempt_count', 'profiling_question_ids', 'boundary')},
        ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
