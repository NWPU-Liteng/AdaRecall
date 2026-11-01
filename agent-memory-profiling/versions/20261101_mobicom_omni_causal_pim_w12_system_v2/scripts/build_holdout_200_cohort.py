#!/usr/bin/env python3
"""Freeze a label-free, per-user holdout cohort before any serving run."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def select(prepared: Path, users: list[str], per_user: int,
           namespace: str) -> dict:
    if not users or len(users) != len(set(users)):
        raise ValueError('distinct holdout users required')
    if per_user <= 0:
        raise ValueError('positive per-user count required')
    if not namespace:
        raise ValueError('nonempty selection namespace required')
    audit = json.loads((prepared / 'audit.json').read_text())
    manifests_path = prepared / 'manifests.jsonl'
    if digest(manifests_path) != audit['artifacts_sha256']['manifests.jsonl']:
        raise ValueError('serving manifest does not match preparation audit')
    split = json.loads((prepared / 'split.json').read_text())
    if not set(users).issubset(split['holdout_people']):
        raise ValueError('all requested users must be sealed holdout users')
    by_user: dict[str, list[dict]] = {user: [] for user in users}
    with manifests_path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            user = str(row['person_id'])
            if user in by_user and row['split'] == 'holdout':
                by_user[user].append({key: row[key] for key in (
                    'question_id', 'person_id', 'split', 'effective_timestamp',
                    'question_index')})
    question_ids_by_user = {}
    for user in users:
        rows = by_user[user]
        if len(rows) < per_user:
            raise ValueError(f'holdout user {user} has fewer than {per_user} questions')
        ranked = sorted(rows, key=lambda row: hashlib.sha256(
            f'{namespace}\0{user}\0{row["question_id"]}'.encode()).hexdigest())
        chosen = ranked[:per_user]
        chosen.sort(key=lambda row: (row['effective_timestamp'], row['question_index'],
                                     row['question_id']))
        question_ids_by_user[user] = [row['question_id'] for row in chosen]
    return {
        'schema': 'mobilemem-omni-holdout-cohort-v1',
        'selection': 'per-user lowest SHA256(namespace,user,question_id); manifests only',
        'namespace': namespace,
        'prepared_audit_sha256': digest(prepared / 'audit.json'),
        'manifests_sha256': digest(manifests_path),
        'split_reference_sha256': digest(prepared / 'split.json'),
        'serving_splits': ['holdout'],
        'users': users,
        'per_user': per_user,
        'requests': per_user * len(users),
        'question_ids_by_user': question_ids_by_user,
        'question_count_by_user': {user: len(ids) for user, ids in question_ids_by_user.items()},
        'evaluation_label_files_read': [],
        'boundary': 'question identities and split metadata only; no answer, type, evidence or label fields are copied',
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--users', nargs='+', required=True)
    parser.add_argument('--per-user', type=int, default=50)
    parser.add_argument('--namespace', required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('refusing to overwrite frozen cohort')
    payload = select(args.prepared, [str(user) for user in args.users],
                     args.per_user, args.namespace)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: payload[key] for key in (
        'users', 'per_user', 'requests', 'question_count_by_user')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
