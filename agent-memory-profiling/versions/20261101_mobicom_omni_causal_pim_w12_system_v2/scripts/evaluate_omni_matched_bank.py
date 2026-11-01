"""Separate label-side preparation and closure for matched bank qualification."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from audit_omni_functional_quality import matched_quality
from audit_omni_smoke_closure import digest
from build_omni_smoke_judge_requests import build_requests
from evaluate_omni_service_quality import summarize_quality
from qualify_omni_functional_pim import checked, verify_registration
from qualify_omni_matched_bank import verify_sources
from score_mobilemem import load_pinned_protocol

MODES = ('matched_host', 'matched_replace')


def prepare(run_root, output_root):
    inputs = {}

    def load(path, expected=None):
        path = Path(path)
        inputs[str(path)] = digest(path)
        if expected is not None and inputs[str(path)] != expected:
            raise ValueError('input identity differs: ' + str(path))
        return json.loads(path.read_text())

    registration_path = run_root / 'registration.json'
    registration = load(registration_path)
    verify_sources(registration)
    summary = load(run_root / 'summary.json')
    if not summary['complete'] or summary['registration_sha256'] != digest(registration_path):
        raise ValueError('complete whole-prefix execution required')
    base = load(registration['base_registration'], registration['base_registration_sha256'])
    verify_registration(base)
    plan = load(base['execution_plan'], base['execution_plan_sha256'])
    routes = load(base['routes'], base['routes_sha256'])
    prepared = Path(plan['prepared'])
    prepared_audit = load(prepared / 'audit.json', plan['prepared_audit_sha256'])
    tables = {}
    for name in ('labels.jsonl', 'manifests.jsonl'):
        path = prepared / name
        inputs[str(path)] = digest(path)
        if inputs[str(path)] != prepared_audit['artifacts_sha256'][name]:
            raise ValueError('prepared evaluator artifact changed')
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if len({row['question_id'] for row in rows}) != len(rows):
            raise ValueError('duplicate evaluation identity')
        tables[name] = {row['question_id']: row for row in rows}
    predictions = {mode: [] for mode in MODES}
    calls, host_calls, token_matches, stop_matches, checks = 0, 0, 0, 0, 0
    for job in plan['jobs']:
        smoke = load(job['smoke'], job['smoke_sha256'])
        originals = smoke['rows'][:registration['requests_per_user']]
        prefix = [row['question_id'] for row in originals]
        result = load(run_root / ('user_' + job['person_id']) / 'result.json')
        if (not result['complete'] or result['registration_sha256'] != digest(registration_path)
                or result['evaluation_label_files_read']
                or len(result['rows']) != len(MODES) * len(prefix)):
            raise ValueError('whole registered execution and label isolation required')
        host_rows = None
        for mode in MODES:
            rows = [row for row in result['rows'] if row['mode'] == mode]
            if [row['question_id'] for row in rows] != prefix:
                raise ValueError('fixed complete original prefix required')
            for row, original in zip(rows, originals):
                if (row['input_contract']['processor_inputs_sha256'] != original['processor_inputs_sha256']
                        or row['evaluation_label_files_read'] or row['question_sha256'] != original['question_sha256']
                        or {check['parameter_identity'] for check in row['full_matrix_checks']} != set(routes['placement'])
                        or len(row['full_matrix_checks']) != len(routes['placement'])):
                    raise ValueError('canonical operands or matrix coverage differs')
                checks += len(row['full_matrix_checks'])
                predictions[mode].append({key: row[key] for key in ('question_id', 'person_id',
                    'answer', 'answer_sha256', 'question_sha256', 'stop_reason')})
            dispatch = load(run_root / ('user_' + job['person_id']) / (mode + '_dispatch.json'))
            if (dispatch['selected_calls'] != sum(row['decode_invocations'] for row in rows) * len(routes['placement'])
                    or dispatch['selected_calls'] != dispatch['host_calls'] + dispatch['modeled_admitted_calls']):
                raise ValueError('dispatch accounting differs')
            if mode == 'matched_host':
                if dispatch['bank_function_replacement_calls']:
                    raise ValueError('matched host must never return bank outputs')
                host_rows = rows
            else:
                if dispatch['bank_function_replacement_calls'] != dispatch['modeled_admitted_calls']:
                    raise ValueError('all admitted replacements must execute')
                calls += dispatch['bank_function_replacement_calls']
                host_calls += dispatch['host_calls']
                token_matches += sum(row['answer_token_ids'] == host['answer_token_ids']
                                     for row, host in zip(rows, host_rows))
                stop_matches += sum(row['stop_reason'] == host['stop_reason']
                                    for row, host in zip(rows, host_rows))
    _, _, template, _ = load_pinned_protocol()
    combined_predictions, combined_requests, identities = [], [], []
    for mode in MODES:
        requests = build_requests({'complete': True, 'rows': predictions[mode]},
                                  tables['labels.jsonl'], tables['manifests.jsonl'], template)
        for prediction, request in zip(predictions[mode], requests):
            identity = prediction['question_id']
            evaluation_id = mode + '/' + identity
            identities.append({'evaluation_id': evaluation_id, 'mode': mode, 'question_id': identity})
            combined_predictions.append(dict(prediction, question_id=evaluation_id))
            combined_requests.append(dict(request, question_id=evaluation_id))
    output_root.mkdir(parents=True, exist_ok=False)
    with (output_root / 'judge_requests.json').open('x') as handle:
        json.dump({'judge_requests': combined_requests,
            'boundary': 'mode-prefixed evaluation IDs only disambiguate repeated serving question IDs; '
                        'prompts use unmodified original questions/labels'}, handle, ensure_ascii=False, indent=2)
    with (output_root / 'evaluation_plan.json').open('x') as handle:
        json.dump({'predictions': combined_predictions, 'identities': identities, 'inputs_sha256': inputs,
            'evaluator_sha256': digest(Path(__file__)),
            'judge_requests_sha256': digest(output_root / 'judge_requests.json'),
            'functional': {'replacement_calls': calls, 'host_calls': host_calls,
                'full_matrix_checks': checks, 'complete_token_matches': token_matches,
                'stop_matches': stop_matches, 'requests_per_mode': len(predictions[MODES[0]])}},
            handle, ensure_ascii=False, indent=2)


def close(output_root):
    plan_path = output_root / 'evaluation_plan.json'
    plan = json.loads(plan_path.read_text())
    if digest(Path(__file__)) != plan['evaluator_sha256']:
        raise ValueError('registered evaluator source changed')
    for path, expected in plan['inputs_sha256'].items():
        if digest(Path(path)) != expected:
            raise ValueError('execution/evaluation artifact changed: ' + path)
    requests_path = output_root / 'judge_requests.json'
    if digest(requests_path) != plan['judge_requests_sha256']:
        raise ValueError('judge requests changed')
    requests = json.loads(requests_path.read_text())['judge_requests']
    judge_path = output_root / 'judge.json'
    judge = json.loads(judge_path.read_text())
    if judge['provenance']['scores_sha256'] != digest(requests_path):
        raise ValueError('judge request provenance differs')
    quality = summarize_quality(plan['predictions'], requests, judge)
    identities = plan['identities']
    per_mode = {mode: [] for mode in MODES}
    for row, identity in zip(quality['rows'], identities):
        if row['question_id'] != identity['evaluation_id']:
            raise ValueError('quality row and registered evaluation identity differ')
        per_mode[identity['mode']].append(dict(row, question_id=identity['question_id']))
    comparison = matched_quality(per_mode[MODES[0]], per_mode[MODES[1]])
    result = {'schema_version': 'omni-matched-bank-quality-v1', 'complete': True,
        'functional': plan['functional'], 'quality_by_mode': per_mode, 'matched_quality': comparison,
        'inputs_sha256': {str(plan_path): digest(plan_path), str(requests_path): digest(requests_path),
                          str(judge_path): digest(judge_path)},
        'quality_boundary': quality['boundary'], 'replacement_expansion_authorized': False,
        'actual_physical_pim_executed': False, 'end_to_end_speedup': None,
        'performance_comparable': False, 'main_experiment_closed': False, 'publication_ready': False}
    with (output_root / 'summary.json').open('x') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps({'functional': result['functional'], 'quality': comparison}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', type=Path)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--close', action='store_true')
    args = parser.parse_args()
    if args.close:
        close(args.output_root)
    elif args.run_root:
        prepare(args.run_root, args.output_root)
    else:
        parser.error('run root required when preparing')


if __name__ == '__main__':
    main()
