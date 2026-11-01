"""Select a label-free, time-spread replay subset for Reader capacity profiling."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def select(prepared: Path, replay: Path, pilot: Path, output: Path,
           users: list[str], per_user: int) -> dict:
    manifest_path = prepared / 'manifests.jsonl'
    prepared_audit = json.loads((prepared / 'audit.json').read_text())
    if digest(manifest_path) != prepared_audit['artifacts_sha256']['manifests.jsonl']:
        raise ValueError('prepared serving manifest does not match its audit')
    replay_manifest = json.loads((replay / 'manifest.json').read_text())
    for name, expected in replay_manifest['artifacts_sha256'].items():
        if digest(replay / name) != expected:
            raise ValueError(f'persistent replay artifact changed: {name}')
    if replay_manifest.get('runtime_labels_read') is not False:
        raise ValueError('persistent replay is not label-separated')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'refusing to overwrite non-empty output: {output}')

    dev_by_id = {}
    for row in load_jsonl(manifest_path):
        if row.get('split') == 'development' and str(row.get('person_id')) in users:
            dev_by_id[str(row['question_id'])] = row
    eligible_by_id = {}
    for row in load_jsonl(replay / 'schedule_audit.jsonl'):
        if (str(row.get('person_id')) in users
                and row.get('temporal_replay_eligible') is True):
            eligible_by_id[str(row['question_id'])] = row
    replay_by_id = {}
    for row in load_jsonl(replay / 'queries.jsonl'):
        if str(row.get('person_id')) in users:
            if row.get('runtime_labels_read') is not False:
                raise ValueError(f'replay query is not label-free: {row["question_id"]}')
            replay_by_id[str(row['question_id'])] = row
    pilot_payload = json.loads(pilot.read_text())
    pilot_ids = {identity for values in pilot_payload['question_ids_by_user'].values()
                 for identity in values}

    output.mkdir(parents=True, exist_ok=True)
    selected_by_user = {}
    details_by_user = {}
    for user in users:
        candidates = [
            (identity, replay_by_id[identity], eligible_by_id[identity])
            for identity, manifest in dev_by_id.items()
            if str(manifest['person_id']) == user
            and identity in replay_by_id and identity in eligible_by_id
            and identity not in pilot_ids
        ]
        candidates.sort(key=lambda item: (item[1]['checkpoint_time'], item[0]))
        if len(candidates) < per_user:
            raise ValueError(f'user {user} has only {len(candidates)} eligible candidates')
        indices = [round(index * (len(candidates) - 1) / (per_user - 1))
                   for index in range(per_user)] if per_user > 1 else [len(candidates) // 2]
        chosen = [candidates[index] for index in indices]
        identities = [item[0] for item in chosen]
        if len(set(identities)) != per_user:
            raise ValueError(f'user {user} selection contains duplicate IDs')
        selected_by_user[user] = identities
        details_by_user[user] = {
            'eligible_development_candidates_after_pilot_exclusion': len(candidates),
            'selected_count': len(identities),
            'selected_checkpoints': [item[1]['checkpoint_time'] for item in chosen],
            'selected_ids': identities,
        }
        ids_path = output / f'user_{user}_question_ids.json'
        if ids_path.exists():
            raise FileExistsError(ids_path)
        ids_path.write_text(json.dumps(identities, ensure_ascii=False, indent=2) + '\n')

    payload = {
        'schema': 'mobilemem-host-capacity-replay-selection-v1',
        'prepared_audit_sha256': digest(prepared / 'audit.json'),
        'prepared_manifests_sha256': digest(manifest_path),
        'persistent_replay_manifest_sha256': digest(replay / 'manifest.json'),
        'persistent_replay_queries_sha256': digest(replay / 'queries.jsonl'),
        'schedule_audit_sha256': digest(replay / 'schedule_audit.jsonl'),
        'prior_200_pilot_sha256': digest(pilot),
        'selection_policy': 'development split intersect temporal_replay_eligible; exclude all IDs in frozen 200-question pilot; select uniformly by checkpoint-time rank',
        'users': users,
        'questions': sum(map(len, selected_by_user.values())),
        'evaluation_label_files_read': [],
        'user_details': details_by_user,
    }
    manifest_path_out = output / 'selection_manifest.json'
    if manifest_path_out.exists():
        raise FileExistsError(manifest_path_out)
    manifest_path_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--replay', type=Path, required=True)
    parser.add_argument('--pilot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--users', nargs='+', default=['0', '1', '2', '3'])
    parser.add_argument('--per-user', type=int, default=16)
    args = parser.parse_args()
    if args.per_user < 1 or len(set(args.users)) != len(args.users):
        raise ValueError('per-user count must be positive and users unique')
    print(json.dumps(select(args.prepared, args.replay, args.pilot, args.output,
                            args.users, args.per_user), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
