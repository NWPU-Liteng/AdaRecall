"""Seal mutually exclusive visual-locator shards without repairing failed outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from omni_proof_runtime import file_digest, write_json
from extract_omni_visual_locators import valid_regions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('fresh merged locator output required')
    shards, rows, jobs, hashes = [], [], [], {}
    for directory in sorted(args.root.glob('visual_shard_*')):
        if not directory.is_dir():
            continue
        status = json.loads((directory / 'status.json').read_text())
        if not status.get('complete'):
            raise ValueError(f'incomplete visual locator shard: {directory}')
        registration = json.loads((directory / 'registration.json').read_text())
        shard_jobs = json.loads((directory / 'jobs.json').read_text())
        shard_hashes = json.loads((directory / 'image_hashes.json').read_text())
        shards.append({'directory': directory.name, 'status': status, 'registration': registration,
                       'records_sha256': file_digest(directory / 'records.jsonl'),
                       'jobs_sha256': file_digest(directory / 'jobs.json')})
        jobs.extend(shard_jobs)
        for member, identity in shard_hashes.items():
            if member in hashes and hashes[member] != identity:
                raise ValueError('content hash changed across locator shards')
            hashes[member] = identity
        with (directory / 'records.jsonl').open() as stream:
            for line in stream:
                row = json.loads(line)
                row['original_parse_valid'] = row['parse_valid']
                row['regions'] = valid_regions({'regions': row.get('regions', [])},
                                               image_size=row.get('image_size'))
                row['parse_valid'] = bool(row['regions'])
                rows.append(row)
    if not shards:
        raise ValueError('no locator shards found')
    row_by_identity = {}
    for row in rows:
        old = row_by_identity.get(row['image_sha256'])
        if old is not None and old != row:
            raise ValueError('duplicate locator output differs')
        row_by_identity[row['image_sha256']] = row
    expected = set(hashes.values())
    if set(row_by_identity) != expected:
        raise ValueError(f'locator denominator mismatch: expected {len(expected)}, got {len(row_by_identity)}')
    args.output.mkdir(parents=True)
    with (args.output / 'records.jsonl').open('x') as stream:
        for identity in sorted(row_by_identity):
            stream.write(json.dumps(row_by_identity[identity], ensure_ascii=False) + '\n')
    write_json(args.output / 'image_hashes.json', hashes)
    payload = {'complete': True, 'schema': 'merged-visual-locator-v1',
        'shards': shards, 'shard_count': len(shards), 'images': len(row_by_identity),
        'invalid_records': sum(not row['parse_valid'] for row in row_by_identity.values()),
        'records_sha256': file_digest(args.output / 'records.jsonl'),
        'image_hashes_sha256': file_digest(args.output / 'image_hashes.json'),
        'boundary': 'failed/truncated locators remain in denominator; no QA labels, retries, or manual completion'}
    write_json(args.output / 'status.json', payload)
    print(json.dumps({key: payload[key] for key in ('complete', 'images', 'invalid_records',
        'records_sha256', 'boundary')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
