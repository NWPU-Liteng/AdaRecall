"""Independent whole-prefix audit of frozen bank dispatch and label-side quality."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from build_omni_smoke_judge_requests import build_requests
from evaluate_omni_service_quality import summarize_quality
from qualify_omni_functional_pim import checked, verify_registration
from audit_omni_smoke_closure import digest
from score_mobilemem import load_pinned_protocol


def matched_quality(original, replacement):
    identities = [row['question_id'] for row in replacement]
    if not identities or len(set(identities)) != len(identities):
        raise ValueError('unique nonempty replacement prefix required')
    if [row['question_id'] for row in original] != identities:
        raise ValueError('same whole-prefix identities and order required')
    rows = []
    for before, after in zip(original, replacement):
        rows.append({'question_id': after['question_id'],
            'original_success': before['strict_local_proxy_success'],
            'replacement_success': after['strict_local_proxy_success']})
    original_successes = sum(row['original_success'] for row in rows)
    replacement_successes = sum(row['replacement_success'] for row in rows)
    return {'requests': len(rows), 'rows': rows,
        'original_successes': original_successes, 'replacement_successes': replacement_successes,
        'original_rate': original_successes / len(rows),
        'replacement_rate': replacement_successes / len(rows),
        'lost_successes': sum(row['original_success'] and not row['replacement_success'] for row in rows),
        'gained_successes': sum(not row['original_success'] and row['replacement_success'] for row in rows),
        'observed_count_nonregression': replacement_successes >= original_successes,
        'statistical_noninferiority_established': False}


def audit(run_root, prepared, baseline_path):
    inputs = {}

    def load(path, expected=None):
        path = Path(path)
        if expected is not None and digest(path) != expected:
            raise ValueError('input provenance differs: ' + str(path))
        inputs[str(path)] = digest(path)
        return json.loads(path.read_text())

    registration_path = run_root / 'registration.json'
    registration = load(registration_path)
    verify_registration(registration)
    summary = load(run_root / 'summary.json')
    if not summary['complete'] or summary['registration_sha256'] != digest(registration_path):
        raise ValueError('complete registered execution required')
    plan = load(registration['execution_plan'], registration['execution_plan_sha256'])
    routes = load(registration['routes'], registration['routes_sha256'])
    if prepared.resolve() != Path(plan['prepared']).resolve():
        raise ValueError('prepared serving cohort differs')
    prepared_audit = load(prepared / 'audit.json', plan['prepared_audit_sha256'])
    tables = {}
    for name in ('labels.jsonl', 'manifests.jsonl'):
        path = prepared / name
        if digest(path) != prepared_audit['artifacts_sha256'][name]:
            raise ValueError('prepared evaluation input changed')
        inputs[str(path)] = digest(path)
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if len({row['question_id'] for row in rows}) != len(rows):
            raise ValueError('duplicate evaluation identity')
        tables[name] = {row['question_id']: row for row in rows}
    _, _, template, _ = load_pinned_protocol()
    predictions, requests, execution_rows, dispatches = [], [], [], []
    for job in plan['jobs']:
        original = load(job['smoke'], job['smoke_sha256'])
        originals = original['rows'][:registration['requests_per_user']]
        expected_ids = [row['question_id'] for row in originals]
        result = load(run_root / ('user_' + job['person_id']) / 'result.json')
        if (not result['complete'] or result['registration_sha256'] != digest(registration_path)
                or result['person_id'] != job['person_id'] or result['evaluation_label_files_read']):
            raise ValueError('incomplete or label-contaminated functional execution')
        for mode in registration['modes']:
            rows = [row for row in result['rows'] if row['mode'] == mode]
            if [row['question_id'] for row in rows] != expected_ids:
                raise ValueError('registered full prefix required in each mode')
            for row, frozen in zip(rows, originals):
                if (row['input_contract']['processor_inputs_sha256'] != frozen['processor_inputs_sha256']
                        or row['evaluation_label_files_read']):
                    raise ValueError('serving operands or label isolation differ')
                checks = row['full_matrix_checks']
                if ({check['parameter_identity'] for check in checks} != set(routes['placement'])
                        or len(checks) != len(routes['placement'])):
                    raise ValueError('one complete check per selected matrix required')
                if mode == 'shadow' and (row['answer_token_ids'] != frozen['answer_token_ids']
                                         or row['stop_reason'] != frozen['stop_reason']):
                    raise ValueError('shadow changed frozen complete generation')
                execution_rows.append(row)
                if mode == 'replace':
                    predictions.append({'question_id': row['question_id'], 'person_id': job['person_id'],
                        'answer': row['answer'], 'answer_sha256': hashlib.sha256(row['answer'].encode()).hexdigest(),
                        'question_sha256': frozen['question_sha256'], 'stop_reason': row['stop_reason']})
            if mode == 'replace':
                dispatch = load(run_root / ('user_' + job['person_id']) / 'replace_dispatch.json')
                if (dispatch['selected_calls'] != dispatch['host_calls'] + dispatch['modeled_admitted_calls']
                        or dispatch['bank_function_replacement_calls'] != dispatch['modeled_admitted_calls']
                        or dispatch['selected_calls'] != sum(row['decode_invocations'] for row in rows)
                            * len(routes['placement'])):
                    raise ValueError('dispatch call accounting differs')
                dispatches.append(dispatch)
    requests = build_requests({'complete': True, 'rows': predictions}, tables['labels.jsonl'],
                              tables['manifests.jsonl'], template)
    request_path = run_root / 'replace_judge_requests.json'
    if load(request_path)['judge_requests'] != requests:
        raise ValueError('judge prompts differ from independently rebuilt full prefix')
    judge = load(run_root / 'replace_judge.json')
    if judge['provenance']['scores_sha256'] != digest(request_path):
        raise ValueError('judge request producer differs')
    quality = summarize_quality(predictions, requests, judge)
    baseline = load(baseline_path)
    if not baseline['complete']:
        raise ValueError('complete frozen original quality required')
    for path, expected in baseline['inputs_sha256'].items():
        source = Path(path)
        if digest(source) != expected:
            raise ValueError('baseline provenance differs: ' + str(source))
        inputs[str(source)] = expected
    baseline_rows = {row['question_id']: row for row in baseline['rows']}
    comparison = matched_quality([baseline_rows[row['question_id']] for row in predictions], quality['rows'])
    return {'schema_version': 'omni-functional-quality-audit-v1', 'complete': True,
        'execution_integrity': 'pass', 'requests_per_mode': len(predictions),
        'shadow_complete_token_matches': sum(row['mode'] == 'shadow' and
            row['complete_greedy_tokens_equal_frozen'] for row in execution_rows),
        'replacement_complete_token_matches': sum(row['mode'] == 'replace' and
            row['complete_greedy_tokens_equal_frozen'] for row in execution_rows),
        'full_matrix_checks': sum(len(row['full_matrix_checks']) for row in execution_rows),
        'replacement_calls': sum(row['bank_function_replacement_calls'] for row in dispatches),
        'host_calls': sum(row['host_calls'] for row in dispatches), 'quality': quality,
        'matched_quality': comparison, 'replacement_expansion_gate_pass': False,
        'inputs_sha256': inputs, 'source_files_sha256': {str(Path(__file__)): digest(Path(__file__))},
        'evaluation_label_files_read': [str(prepared / 'labels.jsonl')],
        'actual_physical_pim_executed': False, 'performance_comparable': False,
        'end_to_end_speedup': None, 'main_experiment_closed': False, 'publication_ready': False,
        'boundary': 'fixed small development prefix, same-family local judge; integrity pass is not '
                    'quality qualification, statistical noninferiority, task success or PIM speedup'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--baseline-quality', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('fresh independent audit required')
    result = audit(args.run_root, args.prepared, args.baseline_quality)
    with args.output.open('x') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result['matched_quality']))


if __name__ == '__main__':
    main()
