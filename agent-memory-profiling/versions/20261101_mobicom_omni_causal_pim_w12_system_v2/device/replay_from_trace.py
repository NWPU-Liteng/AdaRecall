#!/usr/bin/env python3
"""Build a label-free llama.cpp replay manifest from T1.1 workload traces."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import string
import sys


TRACE_SCHEMA = 'omni-serving-workload-trace-v1'
MANIFEST_SCHEMA = 'device-replay-manifest-v1'
STAGE_NAMES = (
    'contract_parse',
    'query_embed',
    'dense_retrieval',
    'rerank',
    'witness_construct',
    'join',
    'materialize',
    'answer_prefill',
    'answer_decode',
)
READER_STAGE_NAMES = (
    'contract_parse',
    'witness_construct',
    'answer_prefill',
    'answer_decode',
)
WORKLOAD_FIELDS = ('prefill_tokens', 'decode_tokens', 'n_images')


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def _nonempty_string(value, path):
    if not isinstance(value, str) or not value:
        raise ValueError(f'{path} must be a non-empty string')
    return value


def _nonnegative_integer(value, path):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f'{path} must be a non-negative integer')
    return value


def _positive_integer(value, path):
    value = _nonnegative_integer(value, path)
    if value == 0:
        raise ValueError(f'{path} must be a positive integer')
    return value


def _sha256(value, path):
    value = _nonempty_string(value, path)
    if len(value) != 64 or any(character not in string.hexdigits for character in value):
        raise ValueError(f'{path} must be a sha256 hex digest')
    return value.lower()


def _person_sort_key(value):
    return (0, int(value)) if value.isdigit() else (1, value)


def build_manifest(trace_dir):
    trace_dir = Path(trace_dir)
    if not trace_dir.is_dir():
        raise ValueError(f'trace directory does not exist: {trace_dir}')
    paths = sorted(trace_dir.rglob('*.json'))
    if not paths:
        raise ValueError(f'no workload traces found in {trace_dir}')

    identities = set()
    workloads = []
    trace_hashes = {}
    for path in paths:
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict) or payload.get('schema') != TRACE_SCHEMA:
            raise ValueError(f'unsupported workload trace schema: {path}')
        person_id = _nonempty_string(payload.get('person_id'), f'{path}:person_id')
        question_id = _nonempty_string(
            payload.get('question_id'), f'{path}:question_id')
        question_sha256 = _sha256(
            payload.get('question_sha256'), f'{path}:question_sha256')
        _nonnegative_integer(payload.get('image_count'), f'{path}:image_count')
        identity = (person_id, question_id)
        if identity in identities:
            raise ValueError(f'duplicate trace identity: {identity}')
        identities.add(identity)
        stages = payload.get('stages')
        if not isinstance(stages, dict) or not stages:
            raise ValueError(f'trace stages must be a non-empty object: {path}')
        unknown_stages = sorted(set(stages) - set(STAGE_NAMES))
        if unknown_stages:
            raise ValueError(f'unknown trace stages in {path}: {unknown_stages}')
        for stage_name, stage in stages.items():
            if not isinstance(stage, dict):
                raise ValueError(f'trace stage must be an object: {path}:{stage_name}')
            values = {name: stage.get(name) for name in WORKLOAD_FIELDS}
            if stage_name not in READER_STAGE_NAMES:
                unexpected = [name for name, value in values.items() if value is not None]
                if unexpected:
                    raise ValueError(
                        f'non-Reader stage has workload fields: {path}:{stage_name}:'
                        f'{unexpected}')
                continue
            missing = [name for name, value in values.items() if value is None]
            if missing:
                raise ValueError(
                    f'Reader stage is missing workload fields: {path}:{stage_name}:'
                    f'{missing}')
            workloads.append({
                'person_id': person_id,
                'question_id': question_id,
                'question_sha256': question_sha256,
                'stage': stage_name,
                'prefill_tokens': _positive_integer(
                    values['prefill_tokens'], f'{path}:{stage_name}:prefill_tokens'),
                'decode_tokens': _nonnegative_integer(
                    values['decode_tokens'], f'{path}:{stage_name}:decode_tokens'),
                'n_images': _nonnegative_integer(
                    values['n_images'], f'{path}:{stage_name}:n_images'),
            })
        trace_hashes[str(path.relative_to(trace_dir))] = file_digest(path)

    if not workloads:
        raise ValueError(f'no Reader stage workloads found in {trace_dir}')
    workloads.sort(key=lambda item: (
        _person_sort_key(item['person_id']),
        item['question_id'],
        STAGE_NAMES.index(item['stage']),
    ))

    combinations = {}
    for workload in workloads:
        key = (workload['prefill_tokens'], workload['decode_tokens'],
               workload['n_images'])
        combination = combinations.get(key)
        if combination is None:
            combination = {
                'prefill_tokens': key[0],
                'decode_tokens': key[1],
                'n_images': key[2],
                'occurrences': 0,
                'sources': [],
            }
            combinations[key] = combination
        combination['occurrences'] += 1
        combination['sources'].append({
            'person_id': workload['person_id'],
            'question_id': workload['question_id'],
            'stage': workload['stage'],
        })

    return {
        'schema': MANIFEST_SCHEMA,
        'trace_schema': TRACE_SCHEMA,
        'trace_dir': str(trace_dir.resolve()),
        'reader_stage_names': list(READER_STAGE_NAMES),
        'non_reader_stage_policy': (
            'non-Reader stages have no prefill/decode/image workload and are omitted'),
        'trace_count': len(trace_hashes),
        'stage_entry_count': len(workloads),
        'unique_combination_count': len(combinations),
        'image_combination_count': sum(
            value['n_images'] > 0 for value in combinations.values()),
        'stage_workloads': workloads,
        'combinations': list(combinations.values()),
        'trace_files_sha256': trace_hashes,
        'script_sha256': file_digest(Path(__file__)),
        'measurements': [],
        'image_replay_boundary': (
            'llama-bench text length combinations do not execute image preprocessing or '
            'the visual encoder; n_images is metadata only'),
        'boundary': (
            'replay input manifest only; no device command is run and no latency, energy, '
            'throughput, or speedup value is produced or implied'),
    }


def main():
    parser = argparse.ArgumentParser(
        description='Build a device replay manifest from T1.1 workload traces.')
    parser.add_argument('trace_dir', type=Path)
    parser.add_argument('--output', type=Path,
                        help='write manifest here instead of stdout; existing files are rejected')
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        raise ValueError(f'refusing to overwrite replay manifest: {args.output}')
    manifest = build_manifest(args.trace_dir)
    if args.output is None:
        json.dump(manifest, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write('\n')
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print(json.dumps({
        'traces': len(manifest['trace_files_sha256']),
        'stage_workloads': manifest['stage_entry_count'],
        'unique_combinations': manifest['unique_combination_count'],
        'image_combinations': manifest['image_combination_count'],
        'output': str(args.output),
    }))


if __name__ == '__main__':
    main()
