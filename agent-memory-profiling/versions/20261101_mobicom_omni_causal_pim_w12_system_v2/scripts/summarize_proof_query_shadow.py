"""Summarize proof-query shadow telemetry without making a PIM claim."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics


def load_rows(paths):
    rows = []
    for path in paths:
        payload = json.loads(Path(path).read_text())
        if not payload.get('complete'):
            raise ValueError(f'incomplete run: {path}')
        rows.extend(payload.get('rows', []))
    return rows


def summarize(rows):
    shadow_rows = [row for row in rows if row.get('proof_plan', {}).get('proof_query_shadow')]
    if not shadow_rows:
        return {
            'schema': 'proof-query-shadow-summary-v1',
            'requests': len(rows),
            'shadow_requests': 0,
            'boundary': 'no proof-query shadow telemetry present',
        }
    records = []
    for row in shadow_rows:
        shadow = row['proof_plan']['proof_query_shadow']
        result = shadow['result']
        coverage = shadow.get('host_coverage', {})
        input_count = result['input_candidate_count']
        eligible_count = result['eligible_candidate_count']
        records.append({
            'question_id': row['question_id'],
            'service_seconds': row['service_seconds'],
            'shadow_seconds': shadow['seconds'],
            'input_candidate_count': input_count,
            'eligible_candidate_count': eligible_count,
            'candidate_reduction_fraction': (1.0 - eligible_count / input_count
                                              if input_count else 0.0),
            'descriptor_bytes': shadow['candidate_table']['descriptor_bytes'],
            'proof_tuple_count': result['proof_tuple_count'],
            'host_coverage': coverage.get('coverage'),
            'safe_to_replace_host_grounding': shadow['safe_to_replace_host_grounding'],
            'fallback_reason': result['fallback_reason'],
        })
    reductions = [item['candidate_reduction_fraction'] for item in records]
    coverage = [item['host_coverage'] for item in records]
    safe = [item['safe_to_replace_host_grounding'] for item in records]
    return {
        'schema': 'proof-query-shadow-summary-v1',
        'requests': len(rows),
        'shadow_requests': len(records),
        'candidate_reduction_fraction': {
            'mean': statistics.fmean(reductions),
            'p50': statistics.median(reductions),
            'max': max(reductions),
        },
        'host_coverage_rate': sum(value is True for value in coverage) / len(coverage),
        'safe_replacement_rate': sum(value is True for value in safe) / len(safe),
        'shadow_seconds_total': sum(item['shadow_seconds'] for item in records),
        'boundary': 'host reference structural filtering only; no semantic answer, PIM execution, or speedup claim',
        'rows': records,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('runs', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(summarize(load_rows(args.runs)),
                                      ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
