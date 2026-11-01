"""Deterministic label-independent development sampling, preserving source order."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def freeze(manifests, users, per_user, namespace):
    if per_user < 1 or len(users) != len(set(users)) or not users or not namespace:
        raise ValueError('positive sample count, unique users and fixed namespace required')
    selected = {}
    for person in users:
        pool = [row for row in manifests if row['person_id'] == person and row['split'] == 'development']
        if len(pool) < per_user or len({row['question_id'] for row in pool}) != len(pool):
            raise ValueError('insufficient unique authorized development requests')
        ranks = sorted(pool, key=lambda row: hashlib.sha256(
            (namespace + '\0' + person + '\0' + row['question_id']).encode()).hexdigest())
        identities = {row['question_id'] for row in ranks[:per_user]}
        selected[person] = [row['question_id'] for row in pool if row['question_id'] in identities]
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--users', nargs='+', default=['0', '1', '2'])
    parser.add_argument('--per-user', type=int, default=16)
    parser.add_argument('--namespace', default='omni-closure-label-independent-v1')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('fresh immutable cohort required')
    manifest_path = args.prepared / 'manifests.jsonl'
    audit = json.loads((args.prepared / 'audit.json').read_text())
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != audit['artifacts_sha256']['manifests.jsonl']:
        raise ValueError('prepared serving manifests changed')
    manifests = list(map(json.loads, manifest_path.read_text().splitlines()))
    selected = freeze(manifests, args.users, args.per_user, args.namespace)
    result = {'complete': True, 'selection': 'smallest SHA256(namespace,user,question_id); preserve source order',
              'namespace': args.namespace, 'per_user': args.per_user, 'question_ids_by_user': selected,
              'prepared_audit_sha256': hashlib.sha256((args.prepared / 'audit.json').read_bytes()).hexdigest(),
              'evaluation_label_files_read': [], 'request_arrival_boundary': 'source-order replay, not real QA release timestamps',
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
