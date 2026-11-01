"""Audit denominator, source identity, budget and same-run judge provenance."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--cohort', type=Path, required=True)
    args = parser.parse_args()
    cohort = json.loads(args.cohort.read_text())
    expected = {question_id for values in cohort['question_ids_by_user'].values() for question_id in values}
    rows, per_user = [], {}
    for user, question_ids in cohort['question_ids_by_user'].items():
        path = args.root / f'proof_user_{user}.json'
        run = json.loads(path.read_text())
        if not run.get('complete') or run.get('question_ids') != question_ids:
            raise ValueError(f'complete original-order run required for user {user}')
        actual = [row['question_id'] for row in run['rows']]
        if actual != question_ids or any(row['evaluation_label_files_read'] for row in run['rows']):
            raise ValueError(f'denominator/order/evaluation isolation failed for user {user}')
        for row in run['rows']:
            if row['proof_plan']['final_render']['input_tokens'] > 8192:
                raise ValueError(f'final processor budget exceeded: {row["question_id"]}')
            plan = row['proof_plan']
            if plan.get('selection_mode') == 'source_located_proof_join':
                if (plan.get('semantic_proof') is not False
                        or {witness['demand_id'] for witness in plan['selected_witnesses']}
                        != {demand['demand_id'] for demand in plan['demands']}):
                    raise ValueError(f'incomplete or overstated witness join: {row["question_id"]}')
                witness_objects = {identity for witness in plan['selected_witnesses']
                                   for identity in witness['object_ids']}
                if witness_objects != set(plan['accepted_object_ids']):
                    raise ValueError(f'witness/materialization identity mismatch: {row["question_id"]}')
            elif plan.get('selection_mode') == 'nonproof_retrieval_package':
                if plan.get('selected_witnesses') or plan.get('semantic_proof') is not False:
                    raise ValueError(f'nonproof package mislabeled as a proof: {row["question_id"]}')
            else:
                raise ValueError(f'unknown evidence selection mode: {row["question_id"]}')
            if (row['materialized_object_ids'] != plan['accepted_object_ids']
                    or plan['final_render']['input_tokens'] != row['input_tokens']
                    or plan['final_render']['image_count'] > 4
                    or plan['final_render']['text_tokens'] > 3600):
                raise ValueError(f'materialization or registered budget mismatch: {row["question_id"]}')
            if row['stop_reason'] not in {'eos', 'max_new_tokens'}:
                raise ValueError('unknown stop reason')
            events = row['execution_events']
            if [event['phase'] for event in events] != ['query_parse_and_retrieval', 'source_grounding', 'proof_materialize',
                                                         'reader_prefill', 'reader_decode', 'release']:
                raise ValueError('incomplete lifecycle event chain')
            if row['backend_plan']['physical_execution']:
                raise ValueError('unqualified physical PIM execution entered mainline')
        rows.extend(run['rows'])
        per_user[user] = {'requests': len(run['rows']),
            'proof_fallbacks': sum(row['proof_plan'].get('strict_proof_failure', False) for row in run['rows']),
            'nonproof_packages': sum(row['proof_plan']['selection_mode'] == 'nonproof_retrieval_package'
                                     for row in run['rows']),
            'multi_demand_queries': sum(len(row['proof_plan']['demands']) > 1 for row in run['rows']),
            'eos': sum(row['stop_reason'] == 'eos' for row in run['rows']),
            'service_seconds': sum(row['service_seconds'] for row in run['rows']),
            'parse_valid': sum(row['proof_plan']['query_parse']['status'] == 'ok' for row in run['rows']),
            'grounded_witnesses': sum(row['proof_plan']['grounding']['status'] == 'complete' for row in run['rows'])}
    if {row['question_id'] for row in rows} != expected or len(rows) != len(expected):
        raise ValueError('full registered denominator mismatch')
    judge = json.loads((args.root / 'judge.json').read_text())
    if not judge.get('complete') or len(judge['rows']) != len(rows):
        raise ValueError('same-run complete fixed judge required')
    judge_by_key = {(row['run'], row['question_id']): row for row in judge['rows']}
    correct = invalid = 0
    for row in rows:
        key = (str(args.root / f'proof_user_{row["person_id"]}.json'), row['question_id'])
        judgment = judge_by_key.get(key)
        if judgment is None or judgment['prediction_sha256'] != row['answer_sha256']:
            raise ValueError('judge does not bind exact generated answer')
        correct += judgment.get('judge_label') == 'CORRECT' and judgment.get('parse_valid', False)
        invalid += not judgment.get('parse_valid', False)
    summary = {'complete': True, 'requests': len(rows), 'correct': int(correct),
        'invalid_judgments': int(invalid), 'full_denominator_accuracy': correct / len(rows),
        'per_user': per_user, 'judge_sha256': digest(args.root / 'judge.json'),
        'main_experiment_closed': False,
        'boundary': 'single frozen candidate development run and fixed proxy judge; no baseline, ablation, independent validation, mobile/PIM speedup or QA ground truth claim'}
    (args.root / 'closure.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
