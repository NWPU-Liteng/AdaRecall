"""Summarize evidence-state prefilter qualification telemetry only."""
from __future__ import annotations

import argparse
from collections import Counter
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
    shadow_rows = [row for row in rows
                   if row.get('proof_plan', {}).get('evidence_state_prefilter_shadow')]
    if not shadow_rows:
        return {
            'schema': 'evidence-state-prefilter-shadow-summary-v1',
            'requests': len(rows),
            'shadow_requests': 0,
            'boundary': 'no evidence-state prefilter telemetry present',
        }
    records = []
    for row in shadow_rows:
        shadow = row['proof_plan']['evidence_state_prefilter_shadow']
        records.append({
            'question_id': row['question_id'],
            'service_seconds': row['service_seconds'],
            'shadow_seconds': shadow['seconds'],
            'input_candidate_count': shadow['input_candidate_count'],
            'filtered_candidate_count': shadow['filtered_candidate_count'],
            'candidate_reduction_fraction': shadow['candidate_reduction_fraction'],
            'proposed_candidate_reduction_fraction': shadow.get(
                'proposed_candidate_reduction_fraction', shadow['candidate_reduction_fraction']),
            'seed_candidate_count': shadow.get('seed_candidate_count'),
            'package_expansion_count': shadow.get('package_expansion_count'),
            'package_proposed': shadow.get('package_proposed', False),
            'coverage_gate_passed': shadow.get('coverage_gate_passed'),
            'effective_candidate_count': shadow.get('effective_candidate_count'),
            'effective_frontier_candidate_count': shadow.get(
                'effective_frontier_candidate_count'),
            'effective_frontier_reduction_fraction': shadow.get(
                'effective_frontier_reduction_fraction', 0.0),
            'frontier_audit': shadow.get('frontier_audit'),
            'exact_metadata': shadow['exact_metadata'],
            'applied': shadow['applied'],
            'fallback_reason': shadow['fallback_reason'],
            'answer_path_unchanged': shadow['answer_path_unchanged'],
        })
    reductions = [item['candidate_reduction_fraction'] for item in records]
    frontier_audits = [item['frontier_audit'] for item in records
                       if item['frontier_audit'] is not None]
    reasons = Counter(item['fallback_reason'] or 'none' for item in records)
    applied_audits = [item for item in frontier_audits
                      if item['host_witness_coverage'] is not None]
    frontier_reductions = [item['baseline_frontier_reduction_fraction']
                           for item in applied_audits]
    effective_frontier_reductions = [item['effective_frontier_reduction_fraction']
                                     for item in records]
    return {
        'schema': 'evidence-state-prefilter-shadow-summary-v1',
        'requests': len(rows),
        'shadow_requests': len(records),
        'exact_metadata_rate': sum(item['exact_metadata'] for item in records) / len(records),
        'applied_rate': sum(item['applied'] for item in records) / len(records),
        'candidate_reduction_fraction': {
            'mean': statistics.fmean(reductions),
            'p50': statistics.median(reductions),
            'max': max(reductions),
        },
        'baseline_frontier_candidate_reduction_fraction': ({
            'mean': statistics.fmean(frontier_reductions),
            'p50': statistics.median(frontier_reductions),
            'max': max(frontier_reductions),
        } if frontier_reductions else None),
        'coverage_gated_frontier_candidate_reduction_fraction': {
            'mean': statistics.fmean(effective_frontier_reductions),
            'p50': statistics.median(effective_frontier_reductions),
            'max': max(effective_frontier_reductions),
        },
        'applied_host_witness_coverage_rate': (
            sum(item['host_witness_coverage'] for item in applied_audits) / len(applied_audits)
            if applied_audits else None),
        'applied_requests_with_missing_host_witnesses': sum(
            bool(item['host_witness_missing_ids']) for item in applied_audits),
        'coverage_gate_pass_rate': (
            sum(item['coverage_gate_passed'] is True for item in records
                if item['coverage_gate_passed'] is not None)
            / sum(item['coverage_gate_passed'] is not None for item in records)
            if any(item['coverage_gate_passed'] is not None for item in records) else None),
        'fail_open_after_coverage_failure': sum(
            item['fallback_reason'] == 'host_witness_coverage_failed' for item in records),
        'fallback_reasons': dict(sorted(reasons.items())),
        'shadow_seconds_total': sum(item['shadow_seconds'] for item in records),
        'answer_path_unchanged': all(item['answer_path_unchanged'] for item in records),
        'boundary': ('Host reference evidence-state prefilter qualification only; no semantic '
                     'answer replacement, physical PIM execution, or speedup claim'),
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
