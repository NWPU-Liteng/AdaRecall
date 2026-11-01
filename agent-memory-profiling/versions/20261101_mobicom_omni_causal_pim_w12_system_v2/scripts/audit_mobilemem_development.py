"""Audit the closed execution without turning modelled costs into device claims."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from system_audit import SCHEMA, system_checks, reference_execution_valid
from model_identity import model_role_checks


def distribution(values):
    values = [float(value) for value in values if value is not None]
    return ({'count': len(values), 'mean': float(np.mean(values)),
             'p50': float(np.percentile(values, 50)), 'p95': float(np.percentile(values, 95)),
             'min': min(values), 'max': max(values)} if values else None)


def numeric_coverage(row):
    validation = row.get('prefix_validation', {})
    if validation.get('status') == 'reference_execution':
        return reference_execution_valid(row)
    if row.get('numeric_audit_scheduled') is False:
        return validation.get('status') == 'not_requested'
    return (validation.get('status') in {'pass', 'fail'}
            and validation.get('full_greedy_token_ids_equal') is not None
            and validation.get('stop_reason_equal') is not None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--scores', type=Path, required=True)
    parser.add_argument('--judge', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--protocol', choices=('audit', 'production'), default='audit')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('fresh audit output required')
    payload = json.loads(args.result.read_text())
    scores = json.loads(args.scores.read_text())
    rows = payload['rows']
    if [row['question_id'] for row in rows] != [row['question_id'] for row in scores['rows']]:
        raise ValueError('execution and scoring mismatch')
    if hashlib.sha256(args.result.read_bytes()).hexdigest() != scores['provenance']['result_sha256']:
        raise ValueError('scoring belongs to another result file')
    if scores['summary'].get('answer_field', 'answer') != 'answer':
        raise ValueError('promotion audit must score the submitted answer, not another stage')
    valid = [row for row in rows if row['status'] == 'ok']
    host_cycles = sum(row.get('pim_retrieval', {}).get('host_cycles', 0) for row in rows)
    selected_cycles = sum(row.get('pim_retrieval', {}).get('selected_cycles', 0) for row in rows)
    reuse_deltas = []
    for row in valid:
        validation = row.get('prefix_validation', {})
        if (validation.get('status') == 'pass' and row['prefix_state_reuse']['reused_tokens']
                and 'reference_full_prefill_seconds' in validation):
            reuse_deltas.append(validation['reference_full_prefill_seconds']
                                - row['execution_stages']['first']['prefill_seconds'])
    checks = {'registered_questions_complete': bool(rows) and payload['complete'] and
              [row['question_id'] for row in rows] == payload['selected_question_ids'],
              'unique_questions': len({row['question_id'] for row in rows}) == len(rows),
              'development_only': all(row['split'] == 'development' for row in rows),
              'runtime_failures_zero': len(valid) == len(rows),
              'temporal_validity': all(row['temporal_validity'] for row in scores['rows']),
              'int8_reference_exact': all(row.get('pim_retrieval', {}).get('scan_reference_exact') for row in rows),
              'token_budget_respected': all(row.get('serialized_memory_tokens', float('inf')) <= payload['configuration']['budget_tokens'] for row in rows),
              'output_limit_not_reached': all(not row['execution_stages'][stage]['generation_limit_reached']
                                              for row in valid for stage in ('first', 'proof')),
              'numeric_mismatch_recomputed': all(row.get('prefix_validation', {}).get('status') != 'fail'
                                                 or row.get('prefix_state_reuse', {}).get('status') == 'numeric_contract_full_recompute'
                                                 for row in valid),
              'numeric_validation_covered': all(numeric_coverage(row) for row in valid),
              'numeric_audit_schedule_nonempty': not payload['configuration'].get('cross_query_prefix')
                  or any(row.get('numeric_audit_scheduled') for row in valid),
              'protocol_consistent': bool(payload['configuration']['validate_prefix']) == (args.protocol == 'audit'),
              'feature_residency_bounded': all(row.get('feature_cache', {}).get('resident_bytes', float('inf'))
                                               <= row.get('feature_cache', {}).get('capacity_bytes', 0) for row in rows)}
    judge = None
    if args.judge:
        judge = json.loads(args.judge.read_text())
        if [row['question_id'] for row in judge['rows']] != [row['question_id'] for row in rows]:
            raise ValueError('judge denominator mismatch')
        if hashlib.sha256(args.scores.read_bytes()).hexdigest() != judge['provenance']['scores_sha256']:
            raise ValueError('judge belongs to another scoring file')
        for row, request, judgment in zip(rows, scores['judge_requests'], judge['rows']):
            prediction_sha256 = hashlib.sha256(row.get('answer', '').encode()).hexdigest()
            if (prediction_sha256 != request['prediction_sha256']
                    or prediction_sha256 != judgment['prediction_sha256']
                    or request['prompt_sha256'] != judgment['prompt_sha256']):
                raise ValueError('judge prediction or prompt mismatch')
        checks['judge_complete_and_parsable'] = judge['complete'] and judge['summary']['invalid_judgments'] == 0
    integrated_checks = system_checks(rows, payload['configuration'])
    integrated_checks.update(model_role_checks(payload.get('provenance', {})))
    checks.update(integrated_checks)
    report = {'schema': SCHEMA,
              'provenance': {'result_sha256': hashlib.sha256(args.result.read_bytes()).hexdigest(),
                             'scores_sha256': hashlib.sha256(args.scores.read_bytes()).hexdigest(),
                             'judge_sha256': hashlib.sha256(args.judge.read_bytes()).hexdigest() if args.judge else None},
              'system_checks': integrated_checks, 'system_v2_pass': all(integrated_checks.values()),
              'engineering_checks': checks, 'engineering_pass': all(checks.values()), 'protocol': args.protocol,
              'numeric_boundary': 'reference_execution directly serves unsplit full-prefill greedy output and is not an equivalence test; every actual resumed candidate is checked by full independent greedy execution under unsplit_cold_reference_v1; legacy scheduled checks retain their original scope; observed differences fall back',
              'questions': len(rows), 'runtime_failures': len(rows) - len(valid),
              'execution_selection': {key: payload['configuration'].get(key) for key in
                                      ('limit', 'sample_count', 'shard_count', 'shard_index')},
              'source_event_hit_recall': scores['summary']['mean_source_event_hit_recall'],
              'local_judge': judge['summary'] if judge else None,
              'query_type_counts': dict(Counter(row['question_type'] for row in scores['rows'])),
              'service_including_validation_seconds': distribution([row['reader_service_seconds'] for row in rows]),
              'service_excluding_validation_seconds': distribution([row.get('service_excluding_numeric_validation_seconds') for row in rows]),
              'draft_ttft_service_seconds': distribution([row.get('first_answer_ttft_service_seconds') for row in rows]),
              'proof_stage_ttft_service_seconds': distribution([row.get('final_answer_ttft_service_seconds') for row in rows]),
              'ttft_boundary': 'stage-ready timestamps only; single-pass runner returns after generation, not streaming delivery',
              'submitted_answer_modes': dict(Counter(row.get('verification_mode', 'proof_answer_submitted') for row in valid)),
              'proof_role_boundary': 'when draft_answer_preserved_proof_metadata, proof is diagnostic only and cannot improve submitted answers; all proof costs are included',
              'peak_gpu_gib': distribution([row['peak_allocated_bytes'] / 1024**3 for row in rows]),
              'prefix_status_counts': dict(Counter(row.get('prefix_state_reuse', {}).get('status', 'missing') for row in rows)),
              'prefix_numeric_status_counts': dict(Counter(row.get('prefix_validation', {}).get('status', 'missing') for row in rows)),
              'numeric_audit_coverage': sum(bool(row.get('numeric_audit_scheduled')) for row in rows) / max(1, len(rows)),
              'direct_reference_requests': sum(reference_execution_valid(row) for row in valid),
              'resumed_candidate_requests': sum(row.get('prefix_state_reuse', {}).get('prefill_execution') in
                  {'resumed_candidate', 'resumed_candidate_reference_fallback'} for row in valid),
              'prefix_reused_tokens': sum(row.get('prefix_state_reuse', {}).get('reused_tokens', 0) for row in rows),
              'generation_limit_counts': {stage: sum(row['execution_stages'][stage]['generation_limit_reached'] for row in valid)
                                           for stage in ('first', 'proof')},
              'snapshot_seconds': distribution([row.get('snapshot_seconds') for row in rows]),
              'restore_seconds': distribution([row.get('restore_seconds') for row in rows]),
              'diagnostic_prefill_delta_seconds': distribution(reuse_deltas),
              'reuse_delta_boundary': 'full-reference prefill minus reused-request first prefill including checkpoint/restore costs; snapshot creation is included on admission requests, and this is diagnostic rather than controlled end-to-end speedup',
              'search_gap': distribution([row.get('evidence_plan', {}).get('absolute_gap') for row in rows]),
              'search_expanded_nodes': distribution([row.get('evidence_plan', {}).get('expanded_nodes') for row in rows]),
              'search_seconds': distribution([row.get('union_search_seconds') for row in rows]),
              'pim': {'host_cycles': host_cycles, 'selected_cycles_including_cold_writes': selected_cycles,
                      'modeled_score_scan_speedup': host_cycles / selected_cycles if selected_cycles else None,
                      'routes': dict(Counter(row.get('pim_retrieval', {}).get('selected_route') for row in rows)),
                      'cold_copy_bytes_charged': sum(row.get('pim_retrieval', {}).get('new_resident_bytes', 0) for row in rows),
                      'boundary': 'fixed uncalibrated LPDDR5 profile; retrieval INT8 primitive only; cold starts at every shard; no device E2E, power or energy claim'},
              'pim_reader_residual': {
                  'modeled_attention_host_cycles': sum(row.get('pim_reader_residual', {}).get('modeled_attention_host_cycles', 0) for row in rows),
                  'modeled_attention_selected_cycles': sum(row.get('pim_reader_residual', {}).get('modeled_attention_selected_cycles', 0) for row in rows),
                  'routes_per_decode_step': dict(Counter(step['selected_route'] for row in rows
                      for step in row.get('pim_reader_residual', {}).get('attention_decode_steps', []))),
                  'boundary': 'attention-only per-layer phase costs including KV cold writes, multiplied by full-attention layer count; no combined full-model speedup; linear-weight warm scenarios remain independent envelopes'},
              'publication_ready': False,
              'remaining': ['reference-judge final scoring and independent human audit',
                            'development128 and full768 with frozen gates, then sealed551',
                            'cross-query recurrent-state contract and stronger temporal witnesses',
                            'edge-device or trace-backend calibration',
                            'after closure only: causal module attribution, representative baselines and uncertainty'],
              'boundary': payload['boundary']}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
