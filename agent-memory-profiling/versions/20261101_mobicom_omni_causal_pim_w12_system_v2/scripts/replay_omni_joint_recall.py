"""Label-free whole-development query replay against existing visual exchange indices."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from omni_joint_index import JointEncoder, JointIndex, digest, model_files


def ordered_queries(manifests, split, reference):
    people = split['development_people']
    selected = [row for row in manifests if row['person_id'] in people and row['split'] == 'development']
    identities = [row['question_id'] for row in selected]
    if (not reference.get('complete') or reference.get('evaluation_label_files_read') != []
            or identities != reference['question_ids'] or len(identities) != len(set(identities))
            or any(row['visibility_protocol'] != 'history_end' for row in selected)):
        raise ValueError('entire authorized unique source-order development replay required')
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--reference-replay', type=Path, required=True)
    parser.add_argument('--index-root', type=Path, required=True)
    parser.add_argument('--joint-model', type=Path, required=True)
    parser.add_argument('--image-archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix('.registration.json').exists():
        raise ValueError('fresh output required')
    audit_path = args.prepared / 'audit.json'
    audit = json.loads(audit_path.read_text())
    for name in ('manifests.jsonl', 'split.json'):
        if digest(args.prepared / name) != audit['artifacts_sha256'][name]:
            raise ValueError('prepared source changed')
    reference = json.loads(args.reference_replay.read_text())
    if reference['prepared_audit_sha256'] != digest(audit_path):
        raise ValueError('reference replay uses a different prepared release')
    manifests = list(map(json.loads, (args.prepared / 'manifests.jsonl').read_text().splitlines()))
    split = json.loads((args.prepared / 'split.json').read_text())
    selected = ordered_queries(manifests, split, reference)
    identity = model_files(args.joint_model)
    archive_hash = digest(args.image_archive)
    indices = {person: JointIndex(args.index_root / ('index_user_' + person),
               digest(audit_path), person, identity, 65536, archive_hash)
               for person in split['development_people']}
    registration = {'complete': False, 'question_ids': [row['question_id'] for row in selected],
        'reference_replay_sha256': digest(args.reference_replay),
        'prepared_audit_sha256': digest(audit_path), 'model_files_sha256': identity,
        'archive_sha256': archive_hash,
        'indices_sha256': {person: digest(args.index_root / ('index_user_' + person) / 'index.json')
                           for person in indices},
        'sources_sha256': {name: digest(Path(__file__).with_name(name)) for name in
                           ('replay_omni_joint_recall.py', 'omni_joint_index.py', 'omni_patch_projection.py')},
        'selection': 'all development queries in original source order, not true arrival timestamps',
        'recall': 'top16 existing image-bearing observed exchanges; no reranker or selection utility',
        'evaluation_label_files_read': [],
        'boundary': 'retrieval candidate opportunity only; not answer quality, free index construction, PIM, or an online tuned policy'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix('.registration.json').write_text(json.dumps(registration, indent=2))
    torch.set_num_threads(2)
    torch.manual_seed(0)
    started = time.perf_counter()
    encoder = JointEncoder(args.joint_model, args.device, 65536)
    torch.cuda.synchronize(args.device)
    initialization_seconds = time.perf_counter() - started
    rows = []
    for request_index, query in enumerate(selected):
        started = time.perf_counter()
        vector, producer = encoder.encode([{'text': query['question'], 'images': []}])
        edges, scores = indices[query['person_id']].search(vector[0].detach().cpu().numpy(), 16)
        torch.cuda.synchronize(args.device)
        row = {'question_id': query['question_id'], 'person_id': query['person_id'],
               'query_sha256': hashlib.sha256(query['question'].encode()).hexdigest(),
               'query_producer': producer, 'query_and_recall_seconds': time.perf_counter() - started,
               'edge_ids': [edge['edge_id'] for edge in edges], 'scores': scores,
               'session_ids': [edge['session_id'] for edge in edges],
               'candidate_atom_ids': sorted({identity for edge in edges for identity in edge['atom_ids']})}
        rows.append(row)
        with args.output.with_suffix('.rows.jsonl').open('a') as handle:
            handle.write(json.dumps(row) + '\n')
        if request_index % 100 == 0:
            print(json.dumps({'completed_requests': len(rows), 'person_id': query['person_id']}), flush=True)
    if model_files(args.joint_model) != identity:
        raise ValueError('joint model changed during replay')
    if any(digest(Path(__file__).with_name(name)) != expected
           for name, expected in registration['sources_sha256'].items()):
        raise ValueError('serving source changed during replay')
    result = {**registration, 'complete': True, 'rows': rows,
              'model_initialization_seconds': initialization_seconds,
              'existing_index_construction_seconds': {person: index.metadata['construction_seconds']
                                                       for person, index in indices.items()},
              'actual_index_vector_bytes': {person: index.vectors.nbytes for person, index in indices.items()},
              'runtime': {'torch': torch.__version__, 'device': torch.cuda.get_device_name(args.device)}}
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps({'complete': True, 'requests': len(rows)}), flush=True)


if __name__ == '__main__':
    main()
