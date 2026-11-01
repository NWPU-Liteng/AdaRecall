"""Preserve paired repetitions and untargeted-stage drift, without significance claims."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_omni_registered_cohort import digest


def summarize_repeats(audits):
    if len(audits) < 2:
        raise ValueError('at least two complete paired repetitions required')
    first = audits[0]
    identities = [(user['person_id'], user['requests'], user['measurement_contract'])
                  for user in first['per_user']]
    for audit in audits:
        if (not audit['complete'] or audit['execution_integrity'] != 'pass'
                or audit['requests'] != first['requests'] or audit['pair_mismatches']
                or audit['evaluation_label_files_read'] != []
                or [(user['person_id'], user['requests'], user['measurement_contract'])
                    for user in audit['per_user']] != identities):
            raise ValueError('same full cohort, hardware, startup and paired-equivalence contracts required')
    totals = {side: sum(audit[side]['service_seconds']['sum_seconds'] for audit in audits)
              for side in ('reference', 'candidate')}
    stages = {}
    for field in first['reference']['fine_grained_service_stages']['disjoint_seconds']:
        stages[field] = {side: sum(audit[side]['fine_grained_service_stages']['disjoint_seconds'][field][
            'sum_seconds'] for audit in audits) for side in ('reference', 'candidate')}
        stages[field]['observed_difference_seconds'] = stages[field]['reference'] - stages[field]['candidate']
    observations = []
    for audit in audits:
        rows = {side: [] for side in ('reference', 'candidate')}
        for side in rows:
            side_key = 'reference_service_seconds' if side == 'reference' else 'candidate_service_seconds'
            for user in audit['per_user']:
                rows[side].append({'person_id': user['person_id'], 'service_seconds': user[side_key]})
        observations.append({'requests': audit['requests'],
            'measured_server_end_to_end_speedup': audit['measured_server_end_to_end_speedup'],
            'reference_service_seconds': audit['reference']['service_seconds'],
            'candidate_service_seconds': audit['candidate']['service_seconds'],
            'per_user': rows})
    return {'schema_version': 'omni-paired-service-repetitions-v1', 'complete': True,
        'unique_requests': first['requests'], 'paired_repetitions': len(audits),
        'request_observations_per_path': first['requests'] * len(audits),
        'source_pairs': observations, 'service_totals_seconds': totals,
        'pooled_observed_server_latency_ratio': totals['reference'] / totals['candidate'],
        'disjoint_stage_totals_seconds': stages,
        'untargeted_stages_not_reuse_credit': ['input_preparation_seconds', 'prefill_seconds',
                                              'decode_seconds', 'answer_conversion_seconds'],
        'causally_attributable_reuse_speedup': None, 'statistically_significant': None,
        'actual_pim_routing_executed': False, 'modeled_lpddr_end_to_end_speedup': None,
        'evaluation_label_files_read': [], 'main_experiment_closed': False, 'publication_ready': False,
        'boundary': 'two ordered cold-stream repetitions are descriptive robustness checks, not new independent users, a randomized causal study or population significance; do not credit untargeted vision/decode timing drift to embedding reuse'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audits', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('fresh repeat summary required')
    result = summarize_repeats([json.loads(path.read_text()) for path in args.audits])
    result['inputs_sha256'] = {str(path): digest(path) for path in args.audits}
    result['producer_sha256'] = digest(Path(__file__))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps({key: result[key] for key in ('unique_requests', 'paired_repetitions',
        'pooled_observed_server_latency_ratio', 'causally_attributable_reuse_speedup')}))


if __name__ == '__main__':
    main()
