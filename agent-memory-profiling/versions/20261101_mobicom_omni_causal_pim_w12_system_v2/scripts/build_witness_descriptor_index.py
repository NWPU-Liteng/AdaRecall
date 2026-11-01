"""Build the label-free witness-descriptor sidecar for one proof index."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from omni_proof_runtime import file_digest, write_json
from omni_witness_descriptor import SCHEMA, compile_descriptor_index


def build(index: Path, output: Path) -> dict:
    meta = json.loads((index / 'index.json').read_text())
    if meta.get('complete') is not True or meta.get('schema') != 'proof-index-v2':
        raise ValueError('complete proof-index-v2 is required')
    for name, field in (('records.json', 'records_sha256'),
                        ('source.json', 'source_sha256'),
                        ('image_costs.json', 'image_costs_sha256')):
        if file_digest(index / name) != meta[field]:
            raise ValueError(f'proof index payload identity changed: {name}')
    records = json.loads((index / 'records.json').read_text())
    image_costs = {key: int(value) for key, value in
                   json.loads((index / 'image_costs.json').read_text()).items()}
    result = compile_descriptor_index(
        records, image_costs,
        person_id=meta['person_id'],
        source_index_sha256=file_digest(index / 'index.json'),
        records_sha256=file_digest(index / 'records.json'))
    if result['schema'] != SCHEMA:
        raise ValueError('unexpected witness descriptor schema')
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('fresh witness descriptor output required')
    result = build(args.index, args.output)
    print(json.dumps({key: result[key] for key in (
        'schema', 'person_id', 'record_count', 'source_family_count',
        'kind_counts', 'label_free', 'semantic_claims')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
