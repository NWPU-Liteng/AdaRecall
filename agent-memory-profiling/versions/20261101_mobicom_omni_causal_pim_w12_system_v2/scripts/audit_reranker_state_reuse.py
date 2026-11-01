"""Offline same-input checks against previously recorded FP32 reranker outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('fresh reuse audit output required')
    reference, result = [json.loads(path.read_text()) for path in (args.reference, args.result)]
    if (not reference.get('complete') or not result.get('complete')
            or reference['selected_question_ids'] != result['selected_question_ids']
            or len(result['rows']) != len(result['selected_question_ids'])
            or [row['question_id'] for row in result['rows']] != result['selected_question_ids']
            or [row['question_id'] for row in reference['rows']] != reference['selected_question_ids']):
        raise ValueError('complete aligned input traces required')
    identity_checks = {key: reference['provenance'][key] == result['provenance'][key]
                       for key in ('reranker_weights', 'prepared_audit_sha256', 'runtime')}
    for name in ('source_closed_scorer.py', 'intact_bundle_scorer.py', 'joint_bundle_scorer.py',
                 'causal_materialization.py'):
        identity_checks[name] = (reference['provenance']['code_sha256'][name]
                                 == result['provenance']['code_sha256'][name])
    rows = []
    for old, new in zip(reference['rows'], result['rows']):
        left, right = old.get('retrieval_audit', {}), new.get('retrieval_audit', {})
        same_inputs = left.get('reranker_input_sha256') == right.get('reranker_input_sha256')
        previous, current = left.get('reranker_scores', []), right.get('reranker_scores', [])
        error = max((abs(before - after) for before, after in zip(previous, current)), default=None)
        count = result['configuration']['candidate_count']
        orders = [sorted(range(len(values)), key=lambda index: (-values[index], index))[:count]
                  for values in (previous, current)]
        first_old, first_new = [row.get('execution_stages', {}).get('first', {}) for row in (old, new)]
        reuse = new.get('reranker_state_reuse') or {}
        checks = {'runtime_ok': old.get('status') == new.get('status') == 'ok',
                  'same_scoring_inputs': same_inputs and bool(previous) and len(previous) == len(current),
                  'probability_tolerance': error is not None and error <= 1e-4,
                  'top_candidate_order_equal': orders[0] == orders[1],
                  'selected_witnesses_equal': old.get('selected_atoms') == new.get('selected_atoms'),
                  'submitted_answer_equal': old.get('answer') == new.get('answer'),
                  'stop_reason_equal': first_old.get('stop_reason') == first_new.get('stop_reason'),
                  'generated_token_count_equal': first_old.get('generated_tokens') == first_new.get('generated_tokens')}
        rows.append({'question_id': new['question_id'], 'checks': checks,
                     'max_probability_error': error,
                     'reference_rerank_seconds': old.get('rerank_seconds'),
                     'shared_rerank_seconds': new.get('rerank_seconds'),
                     'avoided_prefix_token_evaluations': reuse.get('avoided_prefix_token_evaluations', 0),
                     'prefix_state_bytes': reuse.get('prefix_state_bytes', 0),
                     'fork_seconds': reuse.get('fork_seconds', 0.0)})
    before = sum(row['reference_rerank_seconds'] for row in rows)
    after = sum(row['shared_rerank_seconds'] for row in rows)
    passed = bool(rows) and all(identity_checks.values()) and all(all(row['checks'].values()) for row in rows)
    report = {'same_workload_checks_pass': passed, 'identity_checks': identity_checks, 'rows': rows,
              'questions': len(rows), 'reference_rerank_total_seconds': before,
              'shared_rerank_total_seconds': after,
              'descriptive_rerank_time_ratio': before / after if after else None,
              'descriptive_median_rerank_delta_seconds': statistics.median(
                  row['reference_rerank_seconds'] - row['shared_rerank_seconds'] for row in rows) if rows else None,
              'avoided_prefix_token_evaluations': sum(row['avoided_prefix_token_evaluations'] for row in rows),
              'max_probability_error': max((row['max_probability_error'] for row in rows
                                            if row['max_probability_error'] is not None), default=None),
              'provenance': {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                             for path in (args.reference, args.result, Path(__file__))},
              'boundary': 'ex-post checks against same-input recorded FP32 reference, not online fallback or a global numerical bound; answer strings/counts are compared, not unavailable greedy token IDs; old/new wall times are descriptive non-interleaved development runs, not a significance claim'}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({key: report[key] for key in ('same_workload_checks_pass', 'questions',
                     'reference_rerank_total_seconds', 'shared_rerank_total_seconds',
                     'descriptive_rerank_time_ratio', 'max_probability_error')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
