"""Build and merge resumable, label-independent joint producer indices."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from omni_joint_index import JointEncoder, digest, model_files
from omni_provenance_optimizer import build_provenance_hyperedges
from omni_evidence_planner import images_for_atoms
from run_omni_raw_image_smoke import image_members, load_image


def source_state(prepared, image_archive, user_id, image_only):
    audit = json.loads((prepared / 'audit.json').read_text())
    manifests = [json.loads(line) for line in (prepared / 'manifests.jsonl').read_text().splitlines()]
    if not any(row['person_id'] == user_id and row['split'] == 'development' for row in manifests):
        raise ValueError('development user required')
    for name in ('omni_graph.json', 'manifests.jsonl', 'split.json'):
        if digest(prepared / name) != audit['artifacts_sha256'][name]:
            raise ValueError('observable serving source changed')
    graph = next(item for item in json.loads((prepared / 'omni_graph.json').read_text())
                 if item['person_id'] == user_id)
    by_session, by_id = {}, {}
    for atom in graph['atoms']:
        by_session.setdefault(atom['session_id'], []).append(atom)
        by_id[atom['atom_id']] = atom
    edges = build_provenance_hyperedges([session['session_id'] for session in graph['sessions']], by_session)
    valid, rejected = [], []
    for edge in edges:
        has_image = any(by_id[identity].get('image_refs') for identity in edge['atom_ids'])
        if image_only and not has_image:
            continue
        if any(ref.get('status') not in ('exact', 'unique_user_basename')
               for identity in edge['atom_ids'] for ref in by_id[identity].get('image_refs', [])):
            rejected.append({'edge_id': edge['edge_id'], 'reason': 'unresolved_observable_image'})
        else:
            valid.append(edge)
    return audit, valid, rejected, by_id, digest(image_archive)


def registration(args, valid, rejected, archive_identity):
    return {'complete': False, 'person_id': args.user_id,
        'prepared_audit_sha256': digest(args.prepared / 'audit.json'),
        'model_files_sha256': model_files(args.model), 'archive_sha256': archive_identity,
        'image_max_pixels': args.image_max_pixels, 'max_input_tokens': args.max_input_tokens,
        'batch_size': args.batch_size, 'candidate_edges': len(valid),
        'rejected_edges': rejected, 'evaluation_label_files_read': [],
        'script_sha256': digest(Path(__file__)),
        'module_sha256': digest(Path(__file__).with_name('omni_joint_index.py')),
        'visibility_protocol': 'development_user_own_complete_history_end',
        'pooling': 'official lasttoken; FP32 L2 normalization; no vector quantization',
        'multi_image_extension': 'all original images in exchange, source order; no image pooling or caption substitution',
        'edge_scope': 'image_bearing_complete_exchange' if args.image_only else 'all_complete_exchange'}


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def validate_batch(record, vectors, edges, batch_index, batch_size):
    begin = batch_index * batch_size
    end = min(begin + batch_size, len(edges))
    expected_ids = [edge['edge_id'] for edge in edges[begin:end]]
    if (record['batch_index'] != batch_index or record['start'] != begin or record['end'] != end
            or record['edge_ids'] != expected_ids
            or [producer['edge_id'] for producer in record['producers']] != expected_ids):
        raise ValueError('checkpoint producer identity mismatch')
    if (vectors.ndim != 2 or vectors.shape[0] != end - begin or vectors.shape[1] < 1
            or vectors.dtype != np.float32 or not np.isfinite(vectors).all()
            or not np.allclose(np.linalg.norm(vectors, axis=1), 1., atol=1e-5, rtol=1e-5)):
        raise ValueError('invalid producer vectors')
    if not np.isfinite(record['construction_seconds']) or record['construction_seconds'] < 0:
        raise ValueError('invalid batch duration')


def build(args):
    invocation_started = time.perf_counter()
    if args.batch_size < 1 or args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError('invalid batch or shard configuration')
    if args.output.exists() and not args.resume:
        raise ValueError('fresh output required unless --resume is set')
    args.output.mkdir(parents=True, exist_ok=True)
    batches = args.output / 'batches'
    batches.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(0)
    audit, valid, rejected, by_id, archive_identity = source_state(
        args.prepared, args.image_archive, args.user_id, args.image_only)
    identity = registration(args, valid, rejected, archive_identity)
    identity.update({'shard_index': args.shard_index, 'shard_count': args.shard_count,
                     'total_batches': (len(valid) + args.batch_size - 1) // args.batch_size,
                     'batch_indices': list(range(args.shard_index,
                         (len(valid) + args.batch_size - 1) // args.batch_size, args.shard_count))})
    registration_path = args.output / 'registration.json'
    if registration_path.exists():
        old = json.loads(registration_path.read_text())
        if old != identity:
            raise ValueError('resume registration mismatch')
    else:
        atomic_json(registration_path, identity)
    encoder = JointEncoder(args.model, args.device, args.image_max_pixels, args.max_input_tokens)
    members = image_members(args.image_archive)
    started = time.perf_counter()
    attempt_path = args.output / f'attempt_{time.time_ns()}.json'
    attempt = {'complete': False, 'started_unix_seconds': time.time(),
               'initialization_seconds': started - invocation_started,
               'prior_completed_batches': len(list(batches.glob('*.json')))}
    atomic_json(attempt_path, attempt)
    processed = 0
    for batch_index in identity['batch_indices']:
        begin = batch_index * args.batch_size
        end = min(begin + args.batch_size, len(valid))
        vector_path = batches / f'batch_{batch_index:06d}.npy'
        record_path = batches / f'batch_{batch_index:06d}.json'
        if vector_path.exists() and record_path.exists():
            vectors = np.load(vector_path, allow_pickle=False)
            record = json.loads(record_path.read_text())
            validate_batch(record, vectors, valid, batch_index, args.batch_size)
            if record['vectors_sha256'] != digest(vector_path):
                raise ValueError('checkpoint checksum mismatch')
            processed += end - begin
            continue
        batch_started = time.perf_counter()
        documents, records = [], []
        for edge in valid[begin:end]:
            refs = images_for_atoms([by_id[atom_id] for atom_id in edge['atom_ids']])
            loaded = [load_image(args.image_archive, ref, members) for ref in refs]
            documents.append({'text': edge['embedding_text'], 'images': [image for image, _ in loaded]})
            records.append({'edge_id': edge['edge_id'], 'image_sha256': [checksum for _, checksum in loaded]})
        encoded, input_records = encoder.encode(documents)
        vectors = encoded.cpu().numpy()
        if len(input_records) != len(records):
            raise ValueError('producer metadata cardinality mismatch')
        npy_tmp = vector_path.with_suffix('.npy.tmp')
        with npy_tmp.open('wb') as handle:
            np.save(handle, vectors, allow_pickle=False)
        npy_tmp.replace(vector_path)
        batch_record = {'batch_index': batch_index, 'start': begin, 'end': end,
                        'edge_ids': [edge['edge_id'] for edge in valid[begin:end]],
                        'producers': [{**record, **input_record}
                                      for record, input_record in zip(records, input_records)],
                        'construction_seconds': time.perf_counter() - batch_started,
                        'vectors_sha256': digest(vector_path)}
        validate_batch(batch_record, vectors, valid, batch_index, args.batch_size)
        atomic_json(record_path, batch_record)
        processed += end - begin
        print(json.dumps({'shard': args.shard_index, 'batch': batch_index,
                          'completed_edges': processed, 'assigned_batches': len(identity['batch_indices']),
                          'construction_seconds': time.perf_counter() - started}), flush=True)
    if args.device.startswith('cuda'):
        torch.cuda.synchronize(args.device)
    if identity != json.loads(registration_path.read_text()):
        raise ValueError('registration changed during construction')
    current_identity = registration(args, valid, rejected, digest(args.image_archive))
    if any(current_identity[key] != identity[key] for key in current_identity):
        raise ValueError('producer model, archive, or source changed during construction')
    source_state(args.prepared, args.image_archive, args.user_id, args.image_only)
    attempt.update({'complete': True, 'ended_unix_seconds': time.time(),
                    'invocation_seconds': time.perf_counter() - invocation_started})
    atomic_json(attempt_path, attempt)
    completed_records = [json.loads((batches / f'batch_{index:06d}.json').read_text())
                         for index in identity['batch_indices']]
    attempts = [json.loads(path.read_text()) for path in args.output.glob('attempt_*.json')]
    result = {**identity, 'complete': True,
              'construction_seconds': sum(record['construction_seconds'] for record in completed_records),
              'initialization_seconds_sum': sum(item['initialization_seconds'] for item in attempts),
              'incomplete_attempts': sum(not item['complete'] for item in attempts),
              'attempts': attempts,
              'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(args.device)
              if args.device.startswith('cuda') else 0}
    atomic_json(args.output / 'shard.json', result)
    print(json.dumps({'complete': True, 'shard': args.shard_index,
                      'batches': len(identity['batch_indices']),
                      'seconds': result['construction_seconds']}), flush=True)


def merge(args):
    if args.output.exists():
        raise ValueError('fresh merged output required')
    audit, valid, rejected, _, archive_identity = source_state(
        args.prepared, args.image_archive, args.user_id, args.image_only)
    expected_batches = (len(valid) + args.batch_size - 1) // args.batch_size
    shard_states = [json.loads((path / 'shard.json').read_text()) for path in args.shards]
    if len(shard_states) != args.shard_count or sorted(state['shard_index'] for state in shard_states) != list(range(args.shard_count)):
        raise ValueError('incomplete shard set')
    common = ('person_id', 'prepared_audit_sha256', 'archive_sha256', 'model_files_sha256',
              'image_max_pixels', 'max_input_tokens', 'batch_size', 'candidate_edges',
              'module_sha256', 'script_sha256', 'rejected_edges', 'evaluation_label_files_read',
              'edge_scope', 'pooling', 'multi_image_extension', 'visibility_protocol')
    for state in shard_states:
        if not state.get('complete') or state['shard_count'] != args.shard_count or state['total_batches'] != expected_batches:
            raise ValueError('invalid shard registration')
        if any(state[key] != shard_states[0][key] for key in common):
            raise ValueError('shard identity mismatch')
        if state['person_id'] != args.user_id or state['archive_sha256'] != archive_identity:
            raise ValueError('source identity mismatch')
        if (state['prepared_audit_sha256'] != digest(args.prepared / 'audit.json')
                or state['batch_size'] != args.batch_size
                or state['image_max_pixels'] != args.image_max_pixels
                or state['max_input_tokens'] != args.max_input_tokens
                or state['candidate_edges'] != len(valid) or state['rejected_edges'] != rejected
                or state['evaluation_label_files_read']
                or state['edge_scope'] != ('image_bearing_complete_exchange' if args.image_only else 'all_complete_exchange')
                or state['batch_indices'] != list(range(state['shard_index'], expected_batches, args.shard_count))):
            raise ValueError('merge protocol mismatch')
    batches = {}
    for path in args.shards:
        state = json.loads((path / 'shard.json').read_text())
        for batch_index in state['batch_indices']:
            if batch_index in batches:
                raise ValueError('duplicate batch')
            record_path = path / 'batches' / f'batch_{batch_index:06d}.json'
            vector_path = path / 'batches' / f'batch_{batch_index:06d}.npy'
            record = json.loads(record_path.read_text())
            vectors = np.load(vector_path, allow_pickle=False)
            if record['vectors_sha256'] != digest(vector_path):
                raise ValueError('checkpoint checksum mismatch')
            batches[batch_index] = (vectors, record)
    if sorted(batches) != list(range(expected_batches)):
        raise ValueError('missing batches')
    payloads, producers = [], []
    for batch_index in range(expected_batches):
        vectors, record = batches[batch_index]
        validate_batch(record, vectors, valid, batch_index, args.batch_size)
        payloads.append(vectors)
        producers.extend(record['producers'])
    payload = np.concatenate(payloads)
    args.output.mkdir(parents=True)
    np.save(args.output / 'vectors.npy', payload, allow_pickle=False)
    base = {**shard_states[0], 'complete': True, 'shard_count': args.shard_count,
            'edges': valid, 'producers': producers, 'dimension': int(payload.shape[1]),
            'vectors_sha256': digest(args.output / 'vectors.npy'), 'vector_bytes': int(payload.nbytes),
            'construction_seconds': sum(state['construction_seconds'] for state in shard_states),
            'construction_seconds_sum': sum(state['construction_seconds'] for state in shard_states),
            'initialization_seconds_sum': sum(state['initialization_seconds_sum'] for state in shard_states),
            'incomplete_attempts': sum(state['incomplete_attempts'] for state in shard_states),
            'construction_wall_seconds': max(attempt['ended_unix_seconds'] for state in shard_states
                                             for attempt in state['attempts'] if attempt['complete'])
                - min(attempt['started_unix_seconds'] - attempt['initialization_seconds']
                      for state in shard_states for attempt in state['attempts']),
            'peak_cuda_allocated_bytes': max(state['peak_cuda_allocated_bytes'] for state in shard_states),
            'merged_from': [str(path) for path in args.shards]}
    base.pop('batch_indices', None)
    base.pop('shard_index', None)
    (args.output / 'index.json').write_text(json.dumps(base, ensure_ascii=False, indent=2))
    print(json.dumps({'complete': True, 'edges': len(valid),
                      'construction_seconds_sum': base['construction_seconds_sum'],
                      'construction_wall_seconds': base['construction_wall_seconds']}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command', required=True)
    build_parser = subparsers.add_parser('build')
    for name, kwargs in {
        'prepared': {'type': Path, 'required': True}, 'image-archive': {'type': Path, 'required': True},
        'model': {'type': Path, 'required': True}, 'output': {'type': Path, 'required': True},
        'user-id': {'required': True}, 'device': {'default': 'cuda:0'},
        'batch-size': {'type': int, 'default': 8}, 'image-max-pixels': {'type': int, 'default': 65536},
        'max-input-tokens': {'type': int, 'default': 8192}, 'shard-index': {'type': int, 'required': True},
        'shard-count': {'type': int, 'required': True}}.items():
        build_parser.add_argument('--' + name, **kwargs)
    build_parser.add_argument('--image-only', action='store_true')
    build_parser.add_argument('--resume', action='store_true')
    merge_parser = subparsers.add_parser('merge')
    for name, kwargs in {
        'prepared': {'type': Path, 'required': True}, 'image-archive': {'type': Path, 'required': True},
        'output': {'type': Path, 'required': True}, 'user-id': {'required': True},
        'batch-size': {'type': int, 'default': 8}, 'image-max-pixels': {'type': int, 'default': 65536},
        'max-input-tokens': {'type': int, 'default': 8192}, 'shard-count': {'type': int, 'required': True}}.items():
        merge_parser.add_argument('--' + name, **kwargs)
    merge_parser.add_argument('--image-only', action='store_true')
    merge_parser.add_argument('--shards', type=Path, nargs='+', required=True)
    args = parser.parse_args()
    (build if args.command == 'build' else merge)(args)


if __name__ == '__main__':
    main()
