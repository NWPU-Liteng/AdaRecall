"""Freeze a label-free question cohort for the main serving run."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def build(prepared: Path, output: Path, splits: set[str]) -> dict:
    if output.exists():
        raise ValueError(f'refusing to overwrite frozen cohort: {output}')
    audit = json.loads((prepared / 'audit.json').read_text())
    manifests_path = prepared / 'manifests.jsonl'
    if digest(manifests_path) != audit['artifacts_sha256']['manifests.jsonl']:
        raise ValueError('serving manifest does not match preparation audit')
    by_user: dict[str, list[dict]] = defaultdict(list)
    with manifests_path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row['split'] in splits:
                by_user[str(row['person_id'])].append(row)
    question_ids_by_user = {}
    for person_id, rows in sorted(by_user.items(), key=lambda item: int(item[0])):
        rows.sort(key=lambda row: (row['effective_timestamp'], row['question_id']))
        question_ids_by_user[person_id] = [row['question_id'] for row in rows]
    payload = {
        'schema': 'mobilemem-omni-main-cohort-v1',
        'prepared_audit_sha256': digest(prepared / 'audit.json'),
        'manifests_sha256': digest(manifests_path),
        'serving_splits': sorted(splits),
        'question_ids_by_user': question_ids_by_user,
        'question_count_by_user': {key: len(value) for key, value in question_ids_by_user.items()},
        'questions': sum(map(len, question_ids_by_user.values())),
        'evaluation_label_files_read': [],
        'boundary': 'question IDs and timestamps only; no answer, type, evidence, or label fields are copied',
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--splits', nargs='+', default=['development', 'validation', 'holdout'],
                        choices=['development', 'validation', 'holdout'])
    args = parser.parse_args()
    print(json.dumps(build(args.prepared, args.output, set(args.splits)),
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
