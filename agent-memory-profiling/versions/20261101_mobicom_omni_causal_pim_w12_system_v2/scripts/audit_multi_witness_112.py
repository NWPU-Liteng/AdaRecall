"""Offline multi-witness domain audit over retained pilot result rows."""
from __future__ import annotations

import argparse
from collections import Counter
import glob
import json
import math
from pathlib import Path
import statistics

from omni_multi_witness import audit_multi_witness


def _rows(pattern):
    paths = sorted(glob.glob(pattern))
    rows = []
    for path in paths:
        document = json.loads(Path(path).read_text())
        for row in document.get('rows', []):
            rows.append({**row, '_result_path': path})
    return rows


def _object_costs(index_root, person_id):
    source_path = Path(index_root) / f'proof_index_user_{person_id}' / 'source.json'
    image_cost_path = Path(index_root) / f'proof_index_user_{person_id}' / 'image_costs.json'
    source = json.loads(source_path.read_text())
    image_costs = json.loads(image_cost_path.read_text())
    costs = {}
    for identity, item in source['objects'].items():
        if item['kind'] == 'text':
            costs[identity] = float(len(item['raw_text'].encode('utf-8')) + 32)
        else:
            costs[identity] = float(image_costs[item['source_sha256']])
    return costs


def _quantiles(values, points=(0.5, 0.9)):
    values = sorted(float(value) for value in values if value is not None)
    if not values:
        return {}
    return {f'p{int(point * 100)}': values[min(len(values) - 1,
                                               int(point * (len(values) - 1)))]
            for point in points}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-root', type=Path, required=True)
    parser.add_argument('--index-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-candidate-alternatives', type=int, default=5)
    parser.add_argument('--expected-requests', type=int)
    args = parser.parse_args()
    rows = _rows(str(args.result_root / 'user_*_shard_0.json'))
    if args.expected_requests is not None and len(rows) != args.expected_requests:
        raise ValueError(f'expected {args.expected_requests} rows, found {len(rows)}')
    costs_by_user = {}
    request_reports, errors = [], []
    for row in rows:
        person_id = str(row.get('person_id'))
        plan = row.get('proof_plan') or {}
        demands = plan.get('demands') or []
        witnesses = (plan.get('proof_hypergraph') or {}).get('witnesses') or []
        candidate_records = (row.get('grounding_frontier') or {}).get('records') or []
        if not isinstance(candidate_records, list):
            candidate_records = []
        if person_id not in costs_by_user:
            costs_by_user[person_id] = _object_costs(args.index_root, person_id)
        object_costs = costs_by_user[person_id]
        selected_object_ids = plan.get('accepted_object_ids') or []
        try:
            audit = audit_multi_witness(
                demands, witnesses, candidates=candidate_records,
                relations=plan.get('relations') or [],
                selected_object_ids=selected_object_ids,
                object_costs=object_costs,
                max_candidate_alternatives=args.max_candidate_alternatives)
        except ValueError as error:
            errors.append({'question_id': row.get('question_id'), 'error': str(error)})
            continue
        candidate_widths = [item['candidate_only_branch_count']
                            for item in audit['domains']]
        source_widths = [item['source_located_branch_count']
                         for item in audit['domains']]
        request_reports.append({
            'question_id': row.get('question_id'),
            'person_id': person_id,
            'selection_mode': plan.get('selection_mode'),
            'proof_status': plan.get('proof_status'),
            'demand_count': len(demands),
            'candidate_widths': candidate_widths,
            'source_located_widths': source_widths,
            'new_source_candidates': sum(
                item['new_source_candidate_count'] for item in audit['domains']),
            'new_source_families': sum(
                item['new_source_family_count'] for item in audit['domains']),
            'coverage': audit['coverage'],
            'costs': audit['costs'],
            'domains': audit['domains'],
        })
    all_demands = [item for report in request_reports for item in report['domains']]
    comparable = [item for item in request_reports
                  if item['costs']['selected_object_cost'] is not None
                  and item['costs']['candidate_only_lower_bound'] is not None
                  and item['costs']['selected_object_cost'] > 0]
    reductions = [item['costs']['candidate_only_reduction_fraction']
                  for item in comparable
                  if item['costs']['candidate_only_reduction_fraction'] is not None]
    summary = {
        'schema': 'multi-witness-pilot-audit-v1',
        'requests': len(request_reports),
        'errors': len(errors),
        'demands': len(all_demands),
        'selection_modes': dict(Counter(
            report['selection_mode'] for report in request_reports)),
        'source_located_branch_width': dict(Counter(
            item['source_located_branch_count'] for item in all_demands)),
        'candidate_branch_width': {
            'histogram': dict(Counter(item['candidate_only_branch_count']
                                      for item in all_demands)),
            **_quantiles([item['candidate_only_branch_count'] for item in all_demands]),
        },
        'new_source_candidates': {
            **_quantiles([item['new_source_candidate_count'] for item in all_demands]),
            'total': sum(item['new_source_candidate_count'] for item in all_demands),
        },
        'new_source_families': {
            **_quantiles([item['new_source_family_count'] for item in all_demands]),
            'total': sum(item['new_source_family_count'] for item in all_demands),
        },
        'coverage': {
            'demands_with_source_located_witness': sum(
                item['source_located_branch_count'] > 0 for item in all_demands),
            'demands_with_independent_source_located_or': sum(
                item['source_located_or_available'] for item in all_demands),
            'demands_with_semantic_or': sum(
                item['semantic_or_available'] for item in all_demands),
            'demands_with_candidate_only_or': sum(
                item['candidate_only_branch_count'] >= 2 for item in all_demands),
            'demands_with_candidate_only_or_across_source_families': sum(
                item['new_source_family_count'] >= 2 for item in all_demands),
        },
        'cost_proxy': {
            'units': 'text UTF-8 bytes plus 32-byte state header; image encoded token cost',
            'comparable_requests': len(comparable),
            'selected_cost_total': sum(item['costs']['selected_object_cost']
                                       for item in comparable),
            'candidate_lower_bound_total': sum(item['costs']['candidate_only_lower_bound']
                                                for item in comparable),
            'candidate_reduction_fraction': {
                'mean': statistics.mean(reductions) if reductions else None,
                **_quantiles(reductions),
            },
        },
        'boundary': (
            'The candidate-only branch is a lexical/reranker source candidate, not an independently '
            'verified sufficient witness. Its cost is a lower bound on the search opportunity and '
            'must not be reported as achieved Reader savings or quality.'),
        'request_reports': request_reports,
        'errors_detail': errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: value for key, value in summary.items()
                      if key not in {'request_reports', 'errors_detail'}},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
