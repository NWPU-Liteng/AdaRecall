"""Bind newly built atom/region indexes and reused sealed locator sources."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from omni_proof_runtime import file_digest, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--locator-root', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    args = parser.parse_args()
    entries = {}
    for user in ('0', '1', '2'):
        index = args.root / f'proof_index_user_{user}'
        metadata = json.loads((index / 'index.json').read_text())
        if metadata.get('schema') != 'proof-index-v2' or not metadata.get('complete'):
            raise ValueError(f'complete atom/region index required for user {user}')
        entries[user] = {'index_sha256': file_digest(index / 'index.json'),
            'records_sha256': file_digest(index / 'records.json'),
            'vectors_sha256': file_digest(index / 'vectors.npy'),
            'source_sha256': file_digest(index / 'source.json'),
            'image_costs_sha256': file_digest(index / 'image_costs.json'),
            'record_count': metadata['records'], 'source_object_count': len(
                json.loads((index / 'source.json').read_text())['objects'])}
    locator_status = json.loads((args.locator_root / 'status.json').read_text())
    payload = {'complete': True, 'indexes': entries,
        'reused_locator_root': str(args.locator_root.resolve()),
        'locator_records_sha256': file_digest(args.locator_root / 'records.jsonl'),
        'locator_image_hashes_sha256': file_digest(args.locator_root / 'image_hashes.json'),
        'locator_status_sha256': file_digest(args.locator_root / 'status.json'),
        'locator_extraction_manifest': locator_status,
        'image_archive_sha256': file_digest(args.archive),
        'cost_boundary': 'new text vectors are rebuilt; offline visual locator extraction is reused and its prior construction is reported separately'}
    write_json(args.root / 'index_manifest.json', payload)
    print(json.dumps({'users': len(entries), 'locator_images': locator_status['images'],
                      'archive_sha256': payload['image_archive_sha256']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
