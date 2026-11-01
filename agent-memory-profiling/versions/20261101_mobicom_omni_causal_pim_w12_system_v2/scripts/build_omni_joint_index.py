"""Build development-only, observable-history joint multimodal producer indices."""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--image-archive', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--user-id', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--image-max-pixels', type=int, default=262144)
    parser.add_argument('--max-input-tokens', type=int, default=8192)
    parser.add_argument('--image-only', action='store_true')
    args = parser.parse_args()
    if args.output.exists() or args.batch_size < 1:
        raise ValueError('fresh index output and positive batch size required')
    torch.set_num_threads(2)
    torch.manual_seed(0)
    audit = json.loads((args.prepared / 'audit.json').read_text())
    split = json.loads((args.prepared / 'split.json').read_text())
    manifests = [json.loads(line) for line in (args.prepared / 'manifests.jsonl').read_text().splitlines()]
    if not any(row['person_id'] == args.user_id and row['split'] == 'development' for row in manifests):
        raise ValueError('development user required; no validation/holdout producers in this experiment')
    for name in ('omni_graph.json', 'manifests.jsonl', 'split.json'):
        if digest(args.prepared / name) != audit['artifacts_sha256'][name]:
            raise ValueError('observable serving source changed')
    graph = next(item for item in json.loads((args.prepared / 'omni_graph.json').read_text())
                 if item['person_id'] == args.user_id)
    by_session, by_id = {}, {}
    for atom in graph['atoms']:
        by_session.setdefault(atom['session_id'], []).append(atom)
        by_id[atom['atom_id']] = atom
    edges = build_provenance_hyperedges([session['session_id'] for session in graph['sessions']], by_session)
    valid, rejected = [], []
    for edge in edges:
        has_image = any(by_id[identity].get('image_refs') for identity in edge['atom_ids'])
        if args.image_only and not has_image:
            continue
        if any(ref.get('status') not in ('exact', 'unique_user_basename')
               for identity in edge['atom_ids'] for ref in by_id[identity].get('image_refs', [])):
            rejected.append({'edge_id': edge['edge_id'], 'reason': 'unresolved_observable_image'})
        else:
            valid.append(edge)
    args.output.mkdir(parents=True)
    identity = model_files(args.model)
    archive_identity = digest(args.image_archive)
    registration = {'complete': False, 'person_id': args.user_id,
        'prepared_audit_sha256': digest(args.prepared / 'audit.json'),
        'model_files_sha256': identity, 'archive_sha256': archive_identity,
        'image_max_pixels': args.image_max_pixels, 'max_input_tokens': args.max_input_tokens,
        'batch_size': args.batch_size, 'candidate_edges': len(valid),
        'rejected_edges': rejected, 'evaluation_label_files_read': [],
        'script_sha256': digest(Path(__file__)), 'module_sha256': digest(Path(__file__).with_name('omni_joint_index.py')),
        'visibility_protocol': 'development_user_own_complete_history_end',
        'pooling': 'official lasttoken; FP32 L2 normalization; no vector quantization',
        'multi_image_extension': 'all original images in exchange, source order; no image pooling or caption substitution'}
    registration['edge_scope'] = 'image_bearing_complete_exchange' if args.image_only else 'all_complete_exchange'
    (args.output / 'registration.json').write_text(json.dumps(registration, indent=2))
    encoder = JointEncoder(args.model, args.device, args.image_max_pixels, args.max_input_tokens)
    members = image_members(args.image_archive)
    vectors, producers = [], []
    started = time.perf_counter()
    for begin in range(0, len(valid), args.batch_size):
        documents, records = [], []
        for edge in valid[begin:begin + args.batch_size]:
            refs = images_for_atoms([by_id[identity] for identity in edge['atom_ids']])
            loaded = [load_image(args.image_archive, ref, members) for ref in refs]
            documents.append({'text': edge['embedding_text'], 'images': [image for image, _ in loaded]})
            records.append({'edge_id': edge['edge_id'], 'image_sha256': [checksum for _, checksum in loaded]})
        encoded, input_records = encoder.encode(documents)
        vectors.append(encoded.cpu().numpy())
        producers.extend({**record, **input_record} for record, input_record in zip(records, input_records))
        if begin % (args.batch_size * 32) == 0:
            print(json.dumps({'completed': begin + len(documents), 'total': len(valid),
                              'construction_seconds': time.perf_counter() - started}), flush=True)
    torch.cuda.synchronize(args.device)
    construction_seconds = time.perf_counter() - started
    payload = np.concatenate(vectors)
    np.save(args.output / 'vectors.npy', payload, allow_pickle=False)
    result = {**registration, 'complete': True, 'edges': valid, 'producers': producers,
        'dimension': payload.shape[1], 'vectors_sha256': digest(args.output / 'vectors.npy'),
        'vector_bytes': payload.nbytes, 'construction_seconds': construction_seconds,
        'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(args.device)}
    if identity != model_files(args.model):
        raise ValueError('encoder checkpoint changed during producer construction')
    (args.output / 'index.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({'complete': True, 'edges': len(valid), 'seconds': construction_seconds}), flush=True)


if __name__ == '__main__':
    main()
