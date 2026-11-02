#!/usr/bin/env python3
"""Compare two proof-mainline run directories on the frozen regression fields."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


COMPARE_FIELDS = (
    'answer',
    'accepted_object_ids',
    'proof_status',
    'selection_mode',
    'demand_candidates',
    'query_contract',
)
IGNORED_FIELDS = (
    'extensions',
)


def _load_rows(run_dir: Path) -> dict[str, dict]:
    if not run_dir.is_dir():
        raise ValueError(f'run directory does not exist: {run_dir}')
    rows: dict[str, dict] = {}
    for path in sorted(run_dir.glob('user_*_shard_*.json')):
        payload = json.loads(path.read_text())
        if not payload.get('complete'):
            continue
        person_id = str(payload.get('person_id'))
        for row in payload.get('rows', []):
            question_id = str(row.get('question_id'))
            if question_id in rows:
                raise ValueError(f'duplicate question_id across run files: {question_id}')
            rows[question_id] = {**row, 'person_id': person_id}
    if not rows:
        raise ValueError(f'no complete rows found in: {run_dir}')
    return rows


def _candidate_lists(row: dict) -> dict[str, list[str]]:
    frontier = row.get('grounding_frontier') or {}
    records = frontier.get('records')
    if records is None:
        return {}
    if not isinstance(records, list):
        raise ValueError('grounding_frontier.records must be a list')
    result: dict[str, list[str]] = {}
    for record in records:
        demand_id = str(record.get('demand_id'))
        record_id = record.get('base_record_id')
        if record_id is None:
            raise ValueError('grounding frontier record has no base_record_id')
        result.setdefault(demand_id, []).append(str(record_id))
    return result


def _comparable(row: dict) -> dict:
    plan = row.get('proof_plan')
    if not isinstance(plan, dict):
        raise ValueError('row.proof_plan must be a mapping')
    accepted = plan.get('accepted_object_ids')
    if not isinstance(accepted, list):
        raise ValueError('proof_plan.accepted_object_ids must be a list')
    return {
        'answer': row.get('answer'),
        'accepted_object_ids': sorted(str(value) for value in accepted),
        'proof_status': plan.get('proof_status'),
        'selection_mode': plan.get('selection_mode'),
        'demand_candidates': _candidate_lists(row),
        'query_contract': plan.get('query_contract'),
    }


def _canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def _load_nondeterministic(run_dir: Path) -> set[str]:
    path = run_dir.parent / 'NONDETERMINISTIC_FIELDS.json'
    if not path.exists():
        return set()
    payload = json.loads(path.read_text())
    fields = payload.get('fields', payload)
    if not isinstance(fields, list) or not all(isinstance(value, str) for value in fields):
        raise ValueError('NONDETERMINISTIC_FIELDS.json must contain a string list')
    return set(fields)


def compare(dir_a: Path, dir_b: Path) -> tuple[list[str], list[str]]:
    rows_a = _load_rows(dir_a)
    rows_b = _load_rows(dir_b)
    allowed = _load_nondeterministic(dir_a) | _load_nondeterministic(dir_b)
    failures: list[str] = []
    reported: list[str] = []
    if set(rows_a) != set(rows_b):
        failures.append(
            'question set differs: '
            f'only_a={sorted(set(rows_a) - set(rows_b))} '
            f'only_b={sorted(set(rows_b) - set(rows_a))}')
    for question_id in sorted(set(rows_a) & set(rows_b)):
        value_a = _comparable(rows_a[question_id])
        value_b = _comparable(rows_b[question_id])
        for field in COMPARE_FIELDS:
            if field in IGNORED_FIELDS:
                continue
            left, right = value_a[field], value_b[field]
            if _canonical(left) == _canonical(right):
                continue
            message = f'{question_id}: {field} differs'
            if field in allowed:
                reported.append(message)
            else:
                failures.append(message)
    return failures, reported


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('dir_a', type=Path)
    parser.add_argument('dir_b', type=Path)
    args = parser.parse_args()
    failures, reported = compare(args.dir_a, args.dir_b)
    payload = {
        'status': 'fail' if failures else 'pass',
        'dir_a': str(args.dir_a),
        'dir_b': str(args.dir_b),
        'failures': failures,
        'nondeterministic_differences': reported,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
