"""Summarize one frozen serving stream without reading labels or tuning it."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from audit_mobilemem_development import distribution


def summarize(payload):
    rows = payload['rows']
    valid = [row for row in rows if row.get('status') == 'ok']
    plans = [row['evidence_plan'] for row in valid]
    retrieval = [row['pim_retrieval'] for row in valid]
    numeric = [row.get('reranker_state_reuse', {}).get('numeric_validation', {}) for row in valid]
    return {
        'questions': len(rows), 'runtime_ok': len(valid),
        'selection': {key: payload['configuration'].get(key) for key in
                      ('limit', 'sample_count', 'shard_count', 'cross_query_prefix', 'budget_tokens')},
        'visible_atoms': distribution([row['visible_atoms'] for row in valid]),
        'closure': {
            'input_seeds': sum(plan['input_bundle_count'] for plan in plans),
            'unique_closures': sum(plan['bundle_count'] for plan in plans),
            'accepted_closures': sum(len(plan['accepted_closure_indices']) for plan in plans),
            'objective_schemas': dict(Counter(plan['objective_schema'] for plan in plans)),
            'over_budget_closures': sum(len(plan['infeasible_closure_indices']) for plan in plans),
            'materialized_tokens': sum(plan['used_tokens'] for plan in plans),
            'shared_tokens_between_unique_closures': sum(plan['avoided_duplicate_tokens'] for plan in plans),
            'eliminated_duplicate_seed_reference_tokens': sum(
                plan['duplicate_seed_reference_tokens_avoided'] for plan in plans),
            'search_certified_requests': sum(plan['optimality_certified'] for plan in plans),
            'maximum_search_gap': max((plan['absolute_gap'] for plan in plans), default=None),
            'boundary': 'reference-count savings are logical accounting, not measured tokens saved versus a baseline',
        },
        'reader_state': {
            'requests_with_submitted_reuse': sum(row['prefix_state_reuse']['reused_tokens'] > 0 for row in valid),
            'submitted_reused_tokens': sum(row['prefix_state_reuse']['reused_tokens'] for row in valid),
            'prompt_tokens': sum(row['execution_stages']['first']['prompt_tokens'] for row in valid),
            'numeric_status': dict(Counter(row['prefix_validation']['status'] for row in valid)),
            'fallback_requests': [row['question_id'] for row in valid
                                  if row['prefix_state_reuse']['status'] == 'numeric_contract_full_recompute'],
            'warmed_reference_prefill_delta_seconds': distribution([
                row['prefix_state_reuse'].get('diagnostic_net_prefill_seconds') for row in valid
                if row['prefix_state_reuse']['reused_tokens'] > 0]),
        },
        'reranker_state': {
            'numeric_status': dict(Counter(record.get('status', 'not_audited') for record in numeric)),
            'max_probability_error': max((record.get('max_probability_error', 0) for record in numeric), default=None),
            'prefix_candidate_seconds': sum(record.get('candidate_seconds', 0) for record in numeric),
            'full_reference_seconds': sum(record.get('reference_seconds', 0) for record in numeric),
            'boundary': 'same-run warmed reference is a numerical check and timing diagnosis, not controlled speedup',
        },
        'pim': {
            'routes': dict(Counter(trace['selected_route'] for trace in retrieval)),
            'fallback_reasons': dict(Counter(trace.get('fallback_reason') for trace in retrieval)),
            'host_cycles': sum(trace['host_cycles'] for trace in retrieval),
            'selected_cycles': sum(trace['selected_cycles'] for trace in retrieval),
            'actual_copy_bytes': sum(trace['cold_copy_bytes'] for trace in retrieval),
            'hypothetical_cold_copy_bytes': sum(trace['hypothetical_cold_copy_bytes'] for trace in retrieval),
            'boundary': 'uncalibrated modeled primitive cycles, not mobile E2E latency',
        },
        'service_including_all_validation_seconds': distribution([row['reader_service_seconds'] for row in rows]),
        'ttft_service_seconds': distribution([row['final_answer_ttft_service_seconds'] for row in valid]),
        'peak_gpu_bytes': distribution([row['peak_allocated_bytes'] for row in rows]),
        'publication_ready': False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('fresh mechanism summary required')
    payload = json.loads(args.result.read_text())
    if not payload.get('complete') or [row['question_id'] for row in payload['rows']] != payload['selected_question_ids']:
        raise ValueError('complete registered stream required')
    report = summarize(payload)
    report['provenance'] = {'result_sha256': hashlib.sha256(args.result.read_bytes()).hexdigest(),
                            'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
