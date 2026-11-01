"""Build a label-free exact evidence-state sidecar for a proof index."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SCHEMA = 'evidence-state-index-v1'
EXACT_FIELDS = (
    'record_id', 'kind', 'session_id', 'container_id', 'primary_object_id',
    'object_ids', 'source_atom_ids', 'source_span')


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def build(index: Path, output: Path) -> dict:
    meta = json.loads((index / 'index.json').read_text())
    records_path = index / 'records.json'
    records = json.loads(records_path.read_text())
    if not meta.get('complete') or meta.get('schema') != 'proof-index-v2':
        raise ValueError('complete proof-index-v2 is required')
    if meta.get('records_sha256') != file_digest(records_path):
        raise ValueError('proof index records identity changed')
    sidecar_records = []
    seen = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError('proof index record must be an object')
        missing = [field for field in EXACT_FIELDS
                   if field not in record
                   and not (field == 'source_span'
                            and record.get('kind') == 'image_locator')]
        if missing:
            raise ValueError(f"record {record.get('record_id')} missing exact fields: {missing}")
        record_id = record['record_id']
        if not isinstance(record_id, str) or not record_id or record_id in seen:
            raise ValueError('record identities must be unique non-empty strings')
        if record['kind'] not in {'source_text', 'image_locator'}:
            raise ValueError(f'unsupported exact record kind: {record["kind"]}')
        if not isinstance(record['object_ids'], list) or not record['object_ids']:
            raise ValueError(f'record {record_id} object_ids must be nonempty')
        if not isinstance(record['source_atom_ids'], list) or not record['source_atom_ids']:
            raise ValueError(f'record {record_id} source_atom_ids must be nonempty')
        source_span = record.get('source_span')
        if record['kind'] == 'source_text':
            if (not isinstance(source_span, list) or len(source_span) != 2
                    or any(type(value) is not int for value in source_span)):
                raise ValueError(f'record {record_id} source_span must be [int, int]')
            start, end = source_span
            if start < 0 or end < start:
                raise ValueError(f'record {record_id} source_span is invalid')
        elif source_span is not None:
            raise ValueError(f'record {record_id} image source_span must be null')
        sidecar_records.append({
            field: record.get(field) if field == 'source_span' else record[field]
            for field in EXACT_FIELDS})
        seen.add(record_id)
    result = {
        'schema': SCHEMA,
        'person_id': meta['person_id'],
        'source_index_sha256': file_digest(index / 'index.json'),
        'records_sha256': file_digest(records_path),
        'label_free': True,
        'semantic_claims': False,
        'record_count': len(sidecar_records),
        'records': sidecar_records,
        'boundary': 'exact source identity metadata only; no semantic entailment or answer labels',
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = build(args.index, args.output)
    print(json.dumps({key: result[key] for key in
                      ('schema', 'person_id', 'record_count', 'label_free', 'semantic_claims')},
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
