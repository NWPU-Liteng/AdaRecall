"""Frozen query-only diagnostic; no evidence, gold answers, Judge or QA scores."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import time

from audit_proof_direction import file_hash, selected_jsonl
from omni_proof_query_contract import (QUERY_CONTRACT_PROMPT, QUERY_CONTRACT_SCHEMA,
                                      parse_query_contract, retrieval_views)
from omni_proof_runtime import load_reader, structured_call, write_json


def register(root, previous, prepared, reader):
    if root.exists():
        raise ValueError('fresh diagnostic directory required')
    protocol = json.loads((previous / 'protocol.json').read_text())
    if file_hash(prepared / 'audit.json') != protocol['prepared_audit_sha256']:
        raise ValueError('previous prepared identity changed')
    split = json.loads((prepared / 'split.json').read_text())
    cohort = json.loads((previous / 'cohort.json').read_text())
    identities, selections = [], {}
    for user, full_ids in cohort['question_ids_by_user'].items():
        if user not in split['development_people']:
            raise ValueError('development users only')
        run = json.loads((previous / f'proof_user_{user}.json').read_text())
        selected = set(full_ids[:2]) | {row['question_id'] for row in run['rows']
                                      if len(row['proof_plan']['demands']) > 1}
        selections[user] = [identity for identity in full_ids if identity in selected]
        identities.extend(selections[user])
    manifests = selected_jsonl(prepared / 'manifests.jsonl', identities)
    model_identity = protocol['models']['reader']
    if Path(model_identity['path']).resolve() != reader.resolve():
        raise ValueError('unchanged Reader required')
    for filename, expected in model_identity['files_sha256'].items():
        if file_hash(reader / filename) != expected:
            raise ValueError('Reader identity changed: ' + filename)
    root.mkdir()
    snapshot = root / 'scripts'
    snapshot.mkdir()
    files = ['run_query_contract_diagnostic.py', 'omni_proof_query_contract.py',
             'omni_proof_planner.py', 'omni_proof_runtime.py', 'omni_patch_projection.py',
             'audit_proof_direction.py']
    source = Path(__file__).resolve().parent
    for filename in files:
        shutil.copyfile(source / filename, snapshot / filename)
    write_json(root / 'protocol.json', {
        'registered_before_generation': True, 'kind': 'query_contract_correctness_diagnostic',
        'question_ids_by_user': selections,
        'cases': [manifests[identity] for identity in identities],
        'selection': 'all previous validated multi-demand queries plus original first-two per user; no answer/label selection',
        'previous_protocol_sha256': file_hash(previous / 'protocol.json'),
        'prepared_audit_sha256': file_hash(prepared / 'audit.json'),
        'reader': model_identity, 'bf16_sdpa': True, 'parser_max_new_tokens': 1024,
        'input_budget': 8192, 'image_max_pixels': protocol['image_max_pixels'],
        'source_files_sha256': {filename: file_hash(snapshot / filename) for filename in files},
        'prompt_sha256': __import__('hashlib').sha256(QUERY_CONTRACT_PROMPT.encode()).hexdigest(),
        'schema_sha256': __import__('hashlib').sha256(json.dumps(QUERY_CONTRACT_SCHEMA, sort_keys=True).encode()).hexdigest(),
        'evaluation_label_files_read': [], 'history_or_gold_loaded': False,
        'judge_launched': False, 'baseline_or_ablation_launched': False,
        'validation_accessed': False, 'holdout_accessed': False,
        'parameter_selection_from_diagnostic': False,
        'main_experiment_closed': False,
        'boundary': 'mechanism diagnosis only; query-only checks cannot establish retrieval sufficiency, QA improvement or system speedup'})
    print(json.dumps({'registered': str(root), 'requests': len(identities), 'question_ids_by_user': selections}))


def run_partition(root, user):
    protocol = json.loads((root / 'protocol.json').read_text())
    output = root / f'user_{user}.json'
    if output.exists():
        raise ValueError('fresh partition output required')
    for filename, expected in protocol['source_files_sha256'].items():
        if file_hash(root / 'scripts' / filename) != expected:
            raise ValueError('frozen diagnostic source changed')
    cases = [case for case in protocol['cases'] if case['person_id'] == user]
    if [case['question_id'] for case in cases] != protocol['question_ids_by_user'][user]:
        raise ValueError('partition denominator mismatch')
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
        views = retrieval_views(case['question'], contract)
        row = {'question_id': case['question_id'], 'question': case['question'],
               'raw': generated['answer'], 'parsed': generated['parsed'], 'contract': contract,
               'retrieval_views': views,
               'original_question_in_every_retrieval_view': all(
                   json.loads(view)['original_question'] == case['question'] for view in views),
               'reader_seconds': generated['reader_seconds'], 'stop_reason': generated['stop_reason'],
               'output_tokens': len(generated['answer_token_ids'])}
        rows.append(row)
        write_json(root / f'user_{user}.partial.json', {'complete': False, 'rows': rows})
        print(json.dumps({'question_id': case['question_id'], 'status': contract['status'],
                          'facts': len(contract['source_facts']), 'operations': len(contract['operations']),
                          'stop_reason': generated['stop_reason']}, ensure_ascii=False), flush=True)
    write_json(output, {'complete': True, 'protocol_sha256': file_hash(root / 'protocol.json'),
                        'person_id': user, 'model_startup_seconds': startup, 'rows': rows,
                        'evaluation_label_files_read': [], 'main_experiment_closed': False})


def summarize(root):
    output = root / 'summary.json'
    if output.exists():
        raise ValueError('fresh summary required')
    protocol = json.loads((root / 'protocol.json').read_text())
    rows = []
    for user, identities in protocol['question_ids_by_user'].items():
        run = json.loads((root / f'user_{user}.json').read_text())
        if (not run['complete'] or run['protocol_sha256'] != file_hash(root / 'protocol.json')
                or [row['question_id'] for row in run['rows']] != identities):
            raise ValueError('partition registration/denominator mismatch')
        rows.extend(run['rows'])
    result = {'complete': True, 'requests': len(rows),
              'status': dict(Counter(row['contract']['status'] for row in rows)),
              'stop_reason': dict(Counter(row['stop_reason'] for row in rows)),
              'answer_mode': dict(Counter(row['contract']['answer_mode'] for row in rows)),
              'multi_fact_queries': sum(len(row['contract']['source_facts']) > 1 for row in rows),
              'operation_queries': sum(bool(row['contract']['operations']) for row in rows),
              'relation_queries': sum(bool(row['contract']['relations']) for row in rows),
              'original_question_in_every_retrieval_view': all(
                  row['original_question_in_every_retrieval_view'] for row in rows),
              'model_startup_excluded_reader_seconds': sum(row['reader_seconds'] for row in rows),
              'rows': rows, 'evaluation_label_files_read': [],
              'qa_accuracy_measured': False, 'semantic_effectiveness_established': False,
              'main_experiment_closed': False,
              'boundary': 'schema and query plan checks only; no retrieval, source sufficiency or Reader answer quality evaluated'}
    write_json(output, result)
    print(json.dumps({key: value for key, value in result.items() if key != 'rows'}, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--mode', choices=['register', 'run', 'summarize'], required=True)
    parser.add_argument('--user', choices=['0', '1', '2'])
    parser.add_argument('--previous', type=Path)
    parser.add_argument('--prepared', type=Path)
    parser.add_argument('--reader', type=Path)
    args = parser.parse_args()
    if args.mode == 'register':
        register(args.root, args.previous, args.prepared, args.reader)
    elif args.mode == 'run':
        if args.user is None:
            parser.error('--user is required for run')
        run_partition(args.root, args.user)
    else:
        summarize(args.root)


if __name__ == '__main__':
    main()
