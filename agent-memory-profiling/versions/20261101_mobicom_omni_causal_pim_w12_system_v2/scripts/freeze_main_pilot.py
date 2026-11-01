"""Freeze a deterministic, label-free development subset for a bounded pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def freeze(manifests: list[dict], per_user: int, namespace: str) -> dict:
    if per_user < 1 or not namespace:
        raise ValueError('positive per-user count and fixed namespace required')
    by_user: dict[str, list[dict]] = {}
    for row in manifests:
        if row['split'] == 'development':
            by_user.setdefault(str(row['person_id']), []).append(row)
    eligible = sorted((user for user, rows in by_user.items() if len(rows) >= per_user),
                      key=int)
    if not eligible:
        raise ValueError('no development user has enough questions')
    selected = {}
    for user in eligible:
        pool = by_user[user]
        if len({row['question_id'] for row in pool}) != len(pool):
            raise ValueError(f'duplicate development question for user {user}')
        ranked = sorted(pool, key=lambda row: hashlib.sha256(
            f'{namespace}\0{user}\0{row["question_id"]}'.encode()).hexdigest())
        identities = {row['question_id'] for row in ranked[:per_user]}
        selected[user] = [row['question_id'] for row in pool
                          if row['question_id'] in identities]
    return {
        'schema': 'mobilemem-omni-frozen-pilot-cohort-v1',
        'selection': 'per-user lowest SHA256(namespace,user,question_id); manifests only',
        'namespace': namespace,
        'per_user': per_user,
        'question_ids_by_user': selected,
        'question_count_by_user': {user: len(ids) for user, ids in selected.items()},
        'questions': sum(map(len, selected.values())),
        'serving_splits': ['development'],
        'evaluation_label_files_read': [],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifests', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--per-user', type=int, default=50)
    parser.add_argument('--namespace', default='mobicom-main-200-pilot-v1-20261026')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('refusing to overwrite a frozen pilot cohort')
    manifests = [json.loads(line) for line in args.manifests.read_text().splitlines() if line]
    result = freeze(manifests, args.per_user, args.namespace)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print(json.dumps({key: result[key] for key in
                      ('schema', 'namespace', 'question_count_by_user', 'questions',
                       'evaluation_label_files_read')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
