"""Freeze and revalidate only the query contract on the registered 12 dev queries."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from audit_frozen_query_semantics import EXPECTED
from omni_proof_query_contract import (QUERY_CONTRACT_PROMPT, QUERY_CONTRACT_SCHEMA,
                                       parse_query_contract)
from omni_proof_runtime import file_digest


VERSION = Path(__file__).resolve().parents[1]
PILOT = VERSION / 'artifacts/proof_direction_pilot12_v1_20261025'
DEFAULT_ROOT = VERSION / 'artifacts/proof_direction_semantic_revalidation_v2_20261025'


def write_new(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def environment(gpu):
    return {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu), 'OMP_NUM_THREADS': '2',
            'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2',
            'TOKENIZERS_PARALLELISM': 'false', 'HF_HUB_OFFLINE': '1'}


def register(root):
    if root.exists():
        raise ValueError('fresh registered query-only diagnostic required')
    pilot_protocol = json.loads((PILOT / 'protocol.json').read_text())
    split_path = VERSION / 'artifacts/omni_paper_scope_v3_20261021/prepared/split.json'
    split = json.loads(split_path.read_text())
    cases = pilot_protocol['cases']
    identities = [case['question_id'] for case in cases]
    by_user = {user: [case['question_id'] for case in cases if case['person_id'] == user]
               for user in ('0', '1', '2')}
    if (len(identities) != 12 or set(identities) != set(EXPECTED)
            or any(case['split'] != 'development' or case['person_id'] not in split['development_people']
                   for case in cases)):
        raise ValueError('exact frozen development-only 12-query denominator required')
    model = pilot_protocol['models']['reader']
    for filename, expected in model['files_sha256'].items():
        if file_digest(Path(model['path']) / filename) != expected:
            raise ValueError('pinned Reader model identity changed')
    root.mkdir(parents=True)
    snapshot = root / 'runtime/scripts'
    snapshot.mkdir(parents=True)
    files = ['revalidate_query_contract.py', 'audit_frozen_query_semantics.py',
             'omni_proof_query_contract.py', 'omni_proof_planner.py',
             'omni_proof_runtime.py', 'omni_patch_projection.py']
    for name in files:
        shutil.copyfile(Path(__file__).resolve().parent / name, snapshot / name)
    users = {}
    for index, user in enumerate(by_user):
        output = root / f'user_{user}.json'
        users[user] = {'question_ids': by_user[user], 'gpu': index,
                       'output': str(output)}
    write_new(root / 'protocol.json', {
        'registered_before_generation': True,
        'kind': 'query_only_semantic_contract_revalidation',
        'candidate': root.name,
        'parent_pilot_protocol_sha256': file_digest(PILOT / 'protocol.json'),
        'parent_pilot_run_outputs_unchanged': True,
        'reader': model,
        'cases': cases,
        'question_ids_by_user': by_user,
        'users': users,
        'source_files_sha256': {name: file_digest(snapshot / name) for name in files},
        'prompt_sha256': hashlib.sha256(QUERY_CONTRACT_PROMPT.encode()).hexdigest(),
        'schema_sha256': hashlib.sha256(json.dumps(QUERY_CONTRACT_SCHEMA, sort_keys=True).encode()).hexdigest(),
        'parser_max_new_tokens': 1024, 'input_budget': 8192, 'image_max_pixels': 262144,
        'answer_labels_read': False, 'history_read': False, 'judge_launched': False,
        'validation_accessed': False, 'holdout_accessed': False,
        'baseline_or_ablation_launched': False, 'full_192_run_launched': False,
        'selection_uses_answer_or_judge': False, 'method_effectiveness_established': False,
        'boundary': 'query-only development diagnostic, no retrieval/evidence loading, answer generation, Judge, baseline or ablation'})
    print(json.dumps({'registered': str(root), 'requests': len(identities),
                      'question_ids_by_user': by_user}, ensure_ascii=False), flush=True)


def run_user(root, user):
    protocol = json.loads((root / 'protocol.json').read_text())
    registration = protocol['users'][user]
    output = Path(registration['output'])
    if output.exists():
        raise ValueError('fresh user result required')
    for name, expected in protocol['source_files_sha256'].items():
        if file_digest(root / 'runtime/scripts' / name) != expected:
            raise ValueError('frozen query-only runtime changed: ' + name)
    sys.path.insert(0, str(root / 'runtime/scripts'))
    from omni_proof_query_contract import QUERY_CONTRACT_PROMPT, QUERY_CONTRACT_SCHEMA, parse_query_contract
    from omni_proof_runtime import load_reader, structured_call

    cases = [case for case in protocol['cases'] if case['person_id'] == user]
    if [case['question_id'] for case in cases] != registration['question_ids']:
        raise ValueError('registered question order/denominator mismatch')
    started = time.perf_counter()
    processor, reader = load_reader(Path(protocol['reader']['path']), protocol['image_max_pixels'])
    startup = time.perf_counter() - started
    rows = []
    for case in cases:
        generated = structured_call(reader, processor, QUERY_CONTRACT_PROMPT,
            {'question': case['question'], 'output_contract': 'JSON only'},
            max_tokens=protocol['parser_max_new_tokens'], budget=protocol['input_budget'],
            json_schema=QUERY_CONTRACT_SCHEMA)
        contract = parse_query_contract(generated['parsed'], case['question'])
        rows.append({'question_id': case['question_id'], 'question': case['question'],
            'raw': generated['answer'], 'parsed': generated['parsed'], 'contract': contract,
            'reader_seconds': generated['reader_seconds'], 'stop_reason': generated['stop_reason'],
            'output_tokens': len(generated['answer_token_ids'])})
        partial = root / f'user_{user}.partial.json'
        partial.write_text(json.dumps({'complete': False, 'rows': rows}, ensure_ascii=False, indent=2))
        print(json.dumps({'question_id': case['question_id'], 'status': contract['status'],
            'mode': contract['answer_mode'], 'facts': len(contract['source_facts']),
            'operations': [item['kind'] for item in contract['operations']]}, ensure_ascii=False), flush=True)
    write_new(output, {'complete': True, 'user': user, 'question_ids': registration['question_ids'],
        'model_startup_seconds': startup, 'rows': rows, 'answer_labels_read': False})


def summarize(root):
    protocol = json.loads((root / 'protocol.json').read_text())
    rows = []
    for name, expected in protocol['source_files_sha256'].items():
        if file_digest(root / 'runtime/scripts' / name) != expected:
            raise ValueError('query-only runtime identity mismatch')
    for user, registration in protocol['users'].items():
        run = json.loads(Path(registration['output']).read_text())
        if not run['complete'] or run['question_ids'] != registration['question_ids']:
            raise ValueError('incomplete or changed question denominator')
        rows.extend(run['rows'])
    if len(rows) != 12 or len({row['question_id'] for row in rows}) != 12:
        raise ValueError('complete unique 12-query denominator required')
    details = []
    for row in rows:
        identity = row['question_id']
        contract = row['contract']
        expected = EXPECTED[identity]
        observed = {'mode': contract['answer_mode'], 'facts': len(contract['source_facts']),
                    'operations': [item['kind'] for item in contract['operations']]}
        expected_shape = {key: expected[key] for key in ('mode', 'facts', 'operations')}
        scope_ok = True
        if identity == '2_q_747':
            roles = [fact['text'] for fact in contract['source_facts']]
            scope_ok = (len(roles) == 4 and sum('晚饭' in role for role in roles) == 2
                        and any('外卖' in role for role in roles)
                        and any('水果酸奶' in role for role in roles))
        details.append({'question_id': identity, 'contract_status': contract['status'],
            'observed': observed, 'expected_review_shape': expected_shape,
            'matches_review_shape': contract['status'] == 'ok' and observed == expected_shape and scope_ok,
            'scoped_reference_roles_ok': scope_ok,
            'stop_reason': row['stop_reason'], 'review_rationale': expected['rationale']})
    outputs = {user: file_digest(Path(registration['output']))
               for user, registration in protocol['users'].items()}
    result = {'complete': True, 'requests': len(details),
        'review_shape_matches': sum(row['matches_review_shape'] for row in details),
        'review_shape_fraction': sum(row['matches_review_shape'] for row in details) / len(details),
        'contract_status': dict(Counter(row['contract_status'] for row in details)),
        'stop_reason': dict(Counter(row['stop_reason'] for row in details)),
        'total_query_reader_seconds': sum(row['reader_seconds'] for row in rows),
        'user_output_sha256': outputs,
        'parent_pilot_run_modified': False, 'answer_labels_read': False,
        'history_read': False, 'validation_accessed': False, 'holdout_accessed': False,
        'judge_launched': False, 'baseline_or_ablation_launched': False,
        'full_192_run_launched': False, 'annotations_registered_before_generation': False,
        'independent_semantic_gold': False, 'method_effectiveness_established': False,
        'main_experiment_closed': False,
        'boundary': 'post-revision development query-only smoke; transparent diagnostic, not independent semantic accuracy or method-effectiveness estimate',
        'rows': details}
    write_new(root / 'summary.json', result)
    print(json.dumps({key: value for key, value in result.items() if key != 'rows'},
                     ensure_ascii=False, indent=2), flush=True)


def run_all(root):
    protocol = json.loads((root / 'protocol.json').read_text())
    if (root / 'worker_status.json').exists():
        raise ValueError('query revalidation already launched')
    jobs, processes = [], []
    started = time.perf_counter()
    for user, registration in protocol['users'].items():
        command = [sys.executable, str(root / 'runtime/scripts/revalidate_query_contract.py'),
                   '--mode', 'run-user', '--root', str(root), '--user', user]
        log_path = root / f'user_{user}.log'
        log = log_path.open('x')
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   env=environment(registration['gpu']))
        jobs.append((user, process, log, log_path))
    for user, process, log, log_path in jobs:
        status = process.wait()
        log.close()
        processes.append({'user': user, 'exit_code': status, 'log': str(log_path)})
    write_new(root / 'worker_status.json', {'workers': processes,
        'wall_seconds': time.perf_counter() - started})
    if any(worker['exit_code'] for worker in processes):
        raise RuntimeError('query-only revalidation incomplete')
    summarize(root)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['register', 'run-user', 'run-all', 'summarize'], required=True)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--user', choices=['0', '1', '2'])
    args = parser.parse_args()
    root = args.root.resolve()
    if args.mode == 'register':
        register(root)
    elif args.mode == 'run-user':
        if args.user is None:
            parser.error('--user required for run-user')
        run_user(root, args.user)
    elif args.mode == 'run-all':
        run_all(root)
    else:
        summarize(root)


if __name__ == '__main__':
    main()
