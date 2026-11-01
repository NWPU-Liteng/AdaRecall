"""Extend pinned exchange producers with optional evidence views and Reader costs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from transformers import AutoConfig, AutoProcessor

from omni_evidence_planner import images_for_atoms
from omni_joint_index import JointEncoder, JointIndex, digest, model_files
from omni_provenance_optimizer import build_provenance_hyperedges
from omni_evidence_views import produce_evidence_views
from run_omni_raw_image_smoke import DEFAULT_READER, image_members, load_image


def processor_identity(path):
    names = ('config.json', 'preprocessor_config.json', 'tokenizer.json',
             'tokenizer_config.json', 'chat_template.jinja', 'vocab.json', 'merges.txt')
    return {name: digest(path / name) for name in names if (path / name).is_file()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--image-archive', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--visual-index', '--source-index', dest='visual_index', type=Path, required=True)
    parser.add_argument('--reader', type=Path, default=DEFAULT_READER)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--user-id', required=True)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--evidence-views', action='store_true')
    args = parser.parse_args()
    if args.output.exists() or args.batch_size < 1:
        raise ValueError('fresh output and positive batch size required')
    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.manual_seed(0)
    audit = json.loads((args.prepared / 'audit.json').read_text())
    manifests = [json.loads(line) for line in (args.prepared / 'manifests.jsonl').read_text().splitlines()]
    if not any(row['person_id'] == args.user_id and row['split'] == 'development' for row in manifests):
        raise ValueError('authorized development history only')
    for name in ('omni_graph.json', 'manifests.jsonl', 'split.json'):
        if digest(args.prepared / name) != audit['artifacts_sha256'][name]:
            raise ValueError('serving data identity differs')
    identity = model_files(args.model)
    archive_hash = digest(args.image_archive)
    pinned = JointIndex(args.visual_index, digest(args.prepared / 'audit.json'), args.user_id,
                        identity, 65536, archive_hash)
    graph = next(item for item in json.loads((args.prepared / 'omni_graph.json').read_text())
                 if item['person_id'] == args.user_id)
    by_id = {atom['atom_id']: atom for atom in graph['atoms']}
    by_session = {}
    for atom in graph['atoms']:
        by_session.setdefault(atom['session_id'], []).append(atom)
    edges = build_provenance_hyperedges([session['session_id'] for session in graph['sessions']], by_session)
    valid, rejected = [], []
    for edge in edges:
        if any(ref.get('status') not in {'exact', 'unique_user_basename'}
               for atom_id in edge['atom_ids'] for ref in by_id[atom_id].get('image_refs', [])):
            rejected.append({'edge_id': edge['edge_id'], 'reason': 'unresolved_observable_image'})
        else:
            valid.append(edge)
    args.output.mkdir(parents=True, exist_ok=False)
    registration = {'complete': False, 'person_id': args.user_id, 'edge_scope': 'all_complete_exchange',
        'prepared_audit_sha256': digest(args.prepared / 'audit.json'),
        'model_files_sha256': identity, 'archive_sha256': archive_hash, 'image_max_pixels': 65536,
        'visual_index_sha256': digest(args.visual_index / 'index.json'),
        'visual_index': str(args.visual_index.resolve()), 'evaluation_label_files_read': [],
        'inherited_index_scope': pinned.edge_scope,
        'source_files_sha256': {name: digest(Path(__file__).with_name(name)) for name in
            ('build_omni_compiler_index.py', 'omni_joint_index.py', 'omni_provenance_optimizer.py',
             'omni_evidence_views.py')},
        'evidence_views_requested': args.evidence_views,
        'reader_processor_files_sha256': processor_identity(args.reader),
        'reader_image_max_pixels': 262144, 'batch_size': args.batch_size,
        'visibility_protocol': 'development_user_own_complete_history_end'}
    (args.output / 'registration.json').write_text(json.dumps(registration, indent=2))
    encoder = JointEncoder(args.model, 'cuda:0', 65536, 8192)
    members = image_members(args.image_archive)
    inherited = {edge['edge_id']: (edge, pinned.vectors[index], pinned.metadata['producers'][index])
                 for index, edge in enumerate(pinned.edges)}
    new_edges = [edge for edge in valid if edge['edge_id'] not in inherited]
    produced = {}
    encoding_started = time.perf_counter()
    for begin in range(0, len(new_edges), args.batch_size):
        batch = new_edges[begin:begin + args.batch_size]
        documents = []
        for edge in batch:
            refs = images_for_atoms([by_id[identity] for identity in edge['atom_ids']])
            documents.append({'text': edge['embedding_text'], 'images': [
                load_image(args.image_archive, ref, members)[0] for ref in refs]})
        encoded, records = encoder.encode(documents)
        for edge, vector, record in zip(batch, encoded.detach().cpu().numpy(), records):
            produced[edge['edge_id']] = (vector, {'edge_id': edge['edge_id'], **record})
        if begin % (args.batch_size * 32) == 0:
            print(json.dumps({'new_producers': len(produced), 'total_new': len(new_edges),
                              'seconds': time.perf_counter() - encoding_started}), flush=True)
    torch.cuda.synchronize()
    encoding_seconds = time.perf_counter() - encoding_started
    vectors, producers = [], []
    for edge in valid:
        if edge['edge_id'] in inherited:
            old, vector, record = inherited[edge['edge_id']]
            if old != edge:
                raise ValueError('pinned producer no longer matches observable exchange')
            record = {**record, 'origin': 'pinned_visual_producer'}
        else:
            vector, record = produced[edge['edge_id']]
            record = {**record, 'origin': 'new_same_encoder_exchange'}
        vectors.append(vector)
        producers.append(record)
    view_metadata = {}
    if args.evidence_views:
        views, evidence_payload, view_stats = produce_evidence_views(
            valid, by_id, np.stack(vectors), encoder.encode,
            lambda ref: load_image(args.image_archive, ref, members), batch_size=args.batch_size,
            progress=lambda record: print(json.dumps(record), flush=True))
        np.save(args.output / 'evidence_view_vectors.npy', evidence_payload, allow_pickle=False)
        view_metadata = {
            'retrieval_unit': 'independent_text_exchange_or_single_image_max_pooled_by_parent_exchange',
            'evidence_views': views, 'evidence_view_count': len(views),
            'evidence_view_vector_bytes': evidence_payload.nbytes,
            'evidence_view_vectors_sha256': digest(args.output / 'evidence_view_vectors.npy'),
            'evidence_view_construction_seconds': view_stats['construction_seconds'],
            'encoded_evidence_view_producers': view_stats['encoded_producers']}
    costs_started = time.perf_counter()
    image_costs = {}
    refs = images_for_atoms([by_id[identity] for identity in sorted(
        {identity for edge in valid for identity in edge['atom_ids']}, key=lambda identity: by_id[identity]['ordinal'])])
    reusable_costs = (pinned.metadata.get('reader_image_token_costs', {})
        if pinned.metadata.get('reader_processor_files_sha256') == registration['reader_processor_files_sha256']
        and pinned.metadata.get('reader_image_max_pixels') == registration['reader_image_max_pixels'] else {})
    processor = None
    inherited_image_costs = 0
    for ref in refs:
        identity = ref.get('resolved', ref['normalized'])
        if identity in reusable_costs:
            image_costs[identity] = reusable_costs[identity]
            inherited_image_costs += 1
            continue
        if processor is None:
            processor = AutoProcessor.from_pretrained(str(args.reader), local_files_only=True)
            processor.image_processor.size['longest_edge'] = 262144
            merge = int(AutoConfig.from_pretrained(str(args.reader), local_files_only=True).vision_config.spatial_merge_size)
        image, _ = load_image(args.image_archive, ref, members)
        processed = processor.image_processor(images=[image], return_tensors='pt')
        patches = int(processed['image_grid_thw'].prod(dim=-1).sum().item())
        if patches < 1 or patches % (merge * merge):
            raise ValueError('invalid actual Reader processor grid')
        image_costs[identity] = patches // (merge * merge)
    payload = np.stack(vectors)
    np.save(args.output / 'vectors.npy', payload, allow_pickle=False)
    incremental_seconds = time.perf_counter() - started
    result = {**registration, **view_metadata, 'complete': True, 'edges': valid, 'rejected_edges': rejected,
        'producers': producers, 'dimension': payload.shape[1], 'vector_bytes': payload.nbytes,
        'vectors_sha256': digest(args.output / 'vectors.npy'),
        'reader_image_token_costs': image_costs,
        'inherited_reader_image_costs': inherited_image_costs,
        'inherited_index_construction_seconds': pinned.metadata['construction_seconds'],
        'new_encoding_seconds': encoding_seconds,
        'image_cost_preparation_seconds': time.perf_counter() - costs_started,
        'inherited_visual_construction_seconds': pinned.metadata['construction_seconds'],
        'incremental_construction_seconds': incremental_seconds,
        'construction_seconds': incremental_seconds + pinned.metadata['construction_seconds'],
        'boundary': 'offline same-encoder derived memory; inherited visual production charged separately, not free Reader features; not new exact cache-hit or causal latency evidence'}
    (args.output / 'index.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({'complete': True, 'edges': len(valid),
                      'evidence_views': view_metadata.get('evidence_view_count', 0),
                      'encoded_evidence_views': view_metadata.get('encoded_evidence_view_producers', 0),
                      'new_producers': len(produced),
                      'incremental_seconds': result['incremental_construction_seconds']}), flush=True)


if __name__ == '__main__':
    main()
