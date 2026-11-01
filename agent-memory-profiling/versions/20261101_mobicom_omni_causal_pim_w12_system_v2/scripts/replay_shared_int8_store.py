#!/usr/bin/env python3
"""Materialize one immutable INT8 vector table and replay registered scans."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from run_mobilemem_text import (EMBEDDER, READER, FeatureCache, atom_text, atomize,
                                file_hash, model_hashes)
from mobilemem_runtime import hybrid_candidates
from model_identity import require_model_roles
from pim_backend.lpddr_bank_pim import bank_int8_topk
from qualify_lpddr_materialization import POOL_BYTES, TRANSFER_BYTES, rd16_coordinates


def physical_bank_scores(values, query, ordinals):
    if (values.ndim != 2 or query.ndim != 1 or values.shape[1] != len(query)
            or not len(values) or len(values) != len(ordinals) or values.dtype != np.int8
            or query.dtype != np.int8 or len(query) % TRANSFER_BYTES
            or len(query) * 128**2 > np.iinfo(np.int32).max):
        raise ValueError('aligned nonempty INT8 operands with bounded INT32 accumulation required')
    if len(set(ordinals)) != len(ordinals) or any(
            ordinal < 0 or (ordinal + 1) * len(query) > POOL_BYTES for ordinal in ordinals):
        raise ValueError('distinct in-capacity physical ordinals required')
    offsets = np.arange(len(query) // TRANSFER_BYTES) * TRANSFER_BYTES
    addresses = np.asarray(ordinals, dtype=np.int64)[:, None] * len(query) + offsets[None, :]
    transactions = addresses // TRANSFER_BYTES
    banks = (transactions & 1) * 16 + ((transactions >> 7) & 3) * 4 + ((transactions >> 9) & 3)
    chunks = values.reshape(len(values), -1, TRANSFER_BYTES)
    query_chunks = query.reshape(-1, TRANSFER_BYTES)
    scores = np.zeros(len(values), dtype=np.int32)
    bank_reads = {}
    for bank in range(32):
        positions, chunk_ids = np.where(banks == bank)
        if not len(positions):
            continue
        partial = np.sum(chunks[positions, chunk_ids].astype(np.int32)
                         * query_chunks[chunk_ids].astype(np.int32), axis=1, dtype=np.int32)
        np.add.at(scores, positions, partial)
        bank_reads[f'{bank // 16}/{(bank % 16) // 4}/{bank % 4}'] = len(positions)
    return scores, bank_reads


def count_range(counter, address, size):
    if address % TRANSFER_BYTES or size < 0 or size % TRANSFER_BYTES:
        raise ValueError('aligned physical transaction range required')
    for current in range(address, address + size, TRANSFER_BYTES):
        channel, _, bankgroup, bank, _, _ = rd16_coordinates(current)
        counter[f'{channel}/{bankgroup}/{bank}'] += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--qualified-trace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('fresh replay root required')
    source = json.loads(args.result.read_text())
    if not source.get('complete') or source['configuration']['shard_count'] != 1:
        raise ValueError('complete single-stream source result required')
    provenance = source['provenance']
    qualification = json.loads((args.qualified_trace / 'qualification.json').read_text())
    traffic = json.loads((args.qualified_trace / 'traffic.json').read_text())
    native_registration = json.loads((args.qualified_trace / 'registration.json').read_text())
    native_inputs = native_registration['source_input_sha256']
    for name in ('input.trace', 'traffic.json'):
        path = args.qualified_trace / name
        expected = native_inputs.get(str(path.resolve())) if name == 'traffic.json' else qualification['trace_sha256']
        if not expected or file_hash(path) != expected:
            raise ValueError('qualified traffic artifact changed')
    if (not qualification['host_trace_qualified']
            or qualification['source_result_sha256'] != file_hash(args.result)
            or qualification['source_question_ids'] != source['selected_question_ids']
            or [row['question_id'] for row in source['rows']] != source['selected_question_ids']):
        raise ValueError('complete aligned qualified trace required')
    if provenance['prepared_audit_sha256'] != file_hash(args.prepared / 'audit.json'):
        raise ValueError('prepared audit differs from source run')
    audit = json.loads((args.prepared / 'audit.json').read_text())
    for name in ('mobilemem_graph.json', 'manifests.jsonl'):
        if file_hash(args.prepared / name) != audit['artifacts_sha256'][name]:
            raise ValueError('prepared content differs from source audit')
    for name in ('run_mobilemem_text.py', 'mobilemem_runtime.py', 'lpddr_bank_pim.py'):
        source_path = ROOT / ('pim_backend' if name == 'lpddr_bank_pim.py' else 'scripts') / name
        if file_hash(source_path) != provenance['code_sha256'][name]:
            raise ValueError(f'source code changed: {name}')
    if model_hashes(EMBEDDER) != provenance['embedding_weights']:
        raise ValueError('embedding weights differ from source run')
    for name in ('tokenizer.json', 'tokenizer_config.json', 'merges.txt', 'vocab.json'):
        if file_hash(READER / name) != provenance['reader_weights'][name]:
            raise ValueError('atomization tokenizer differs from source run')
    if file_hash(EMBEDDER / '1_Pooling/config.json') != provenance['embedding_pooling_sha256']:
        raise ValueError('embedding pooling contract changed')
    require_model_roles(provenance)
    graph_data = json.loads((args.prepared / 'mobilemem_graph.json').read_text())
    people = {row['person_id'] for row in source['rows']}
    if len(people) != 1:
        raise ValueError('one persistent feature store required')
    graph = next(item for item in graph_data if item['person_id'] == next(iter(people)))
    from transformers import AutoProcessor
    tokenizer = AutoProcessor.from_pretrained(str(READER), local_files_only=True).tokenizer
    if file_hash(READER / 'tokenizer.json') != provenance['reader_weights']['tokenizer.json']:
        raise ValueError('tokenizer changed after source execution')
    atoms, _ = atomize(graph, tokenizer)
    if len({atom['identity'] for atom in atoms}) != len(atoms):
        raise ValueError('immutable atom identities must be unique')
    for left, right in zip(atoms, atoms[1:]):
        if (left['timestamp'], left['ordinal']) > (right['timestamp'], right['ordinal']):
            raise ValueError('atomization order is not canonical')

    questions = {row['question_id']: row for row in map(json.loads,
                 (args.prepared / 'manifests.jsonl').read_text().splitlines())}
    for row in source['rows']:
        question = questions[row['question_id']]
        if question['split'] != 'development' or any(row[key] != question[key]
                for key in ('person_id', 'effective_timestamp', 'split')):
            raise ValueError('only registered development requests are allowed')
    dimension = json.loads((EMBEDDER / 'config.json').read_text())['hidden_size']
    expected_inventory = {row['identity']: row for row in traffic['inventory']}
    if len(expected_inventory) != traffic['unique_vectors']:
        raise ValueError('qualified inventory has duplicate identities')
    for index, atom in enumerate(atoms):
        if atom['identity'] in expected_inventory:
            entry = expected_inventory[atom['identity']]
            if entry['address'] != index * dimension or entry['bytes'] != dimension:
                raise ValueError('atom/address mapping differs from qualified native trace')
    vector_store = args.output / 'shared_int8_vectors.bin'
    args.output.mkdir(parents=True)
    paths = [Path(__file__), ROOT / 'scripts/qualify_lpddr_materialization.py',
             ROOT / 'scripts/run_mobilemem_text.py', ROOT / 'scripts/mobilemem_runtime.py',
             ROOT / 'scripts/model_identity.py', ROOT / 'pim_backend/lpddr_bank_pim.py',
             ROOT / 'reports/SHARED_INT8_VALUE_REPLAY_CONTRACT_20261019_ZH.md']
    inputs = [args.result, args.qualified_trace / 'qualification.json', args.qualified_trace / 'traffic.json',
              args.prepared / 'audit.json', args.prepared / 'mobilemem_graph.json', args.prepared / 'manifests.jsonl']
    frozen = {str(path.resolve()): file_hash(path) for path in paths + inputs}
    for path in paths:
        target = args.output / 'source_snapshot' / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    (args.output / 'registration.json').write_text(json.dumps({
        'complete': False, 'source_question_ids': source['selected_question_ids'],
        'source_sha256': frozen, 'embedding_weights': provenance['embedding_weights'],
        'allows_development128': False, 'pim_backend_qualified': False, 'device_calibrated': False}, indent=2))
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    features = FeatureCache(EMBEDDER, args.device,
                            source['configuration']['feature_capacity_mib'] * 1024**2,
                            json.dumps(provenance['embedding_weights'], sort_keys=True))
    int_vectors = np.memmap(vector_store, dtype=np.int8, mode='w+', shape=(len(atoms), dimension))
    committed, producer_bytes, all_bank_requests = set(), 0, Counter()
    profile = json.loads((ROOT / 'pim_backend/mobile_lpddr5_bank_profile.json').read_text())
    banks = profile['config']['channels'] * profile['config']['banks_per_channel']
    rows, replay_matches = [], []
    for source_row in source['rows']:
        question = questions[source_row['question_id']]
        visible_indices = [index for index, atom in enumerate(atoms)
                           if atom['timestamp'] <= question['effective_timestamp']]
        if len(visible_indices) != source_row['visible_atoms']:
            raise ValueError('visible atom inventory differs from registered query')
        visible_atoms = [atoms[index] for index in visible_indices]
        vectors, _ = features.materialize(visible_atoms)
        missing = [position for position, index in enumerate(visible_indices) if index not in committed]
        if missing:
            new_indices = [visible_indices[position] for position in missing]
            int_vectors[new_indices] = np.clip(np.rint(vectors[missing] * 127), -127, 127).astype(np.int8)
            int_vectors.flush()
            committed.update(new_indices)
            producer_bytes += len(new_indices) * dimension
            for index in new_indices:
                count_range(all_bank_requests, index * dimension, dimension)
        if any(index not in committed for index in visible_indices):
            raise ValueError('attempted read before immutable producer commit')
        query = features.encode(['Instruct: Given a personal memory question, retrieve supporting historical messages.\nQuery: '
                                + question['question']])[0]
        int_query = np.clip(np.rint(query * 127), -127, 127).astype(np.int8)
        visible_vectors = int_vectors[visible_indices]
        expected_vectors = np.clip(np.rint(vectors * 127), -127, 127).astype(np.int8)
        values_equal = np.array_equal(visible_vectors, expected_vectors)
        top_k = min(128, len(visible_indices))
        selected, scores = bank_int8_topk(visible_vectors, int_query, banks=banks, top_k=top_k)
        reference_scores = visible_vectors.astype(np.int32) @ int_query.astype(np.int32)
        physical_scores, bank_requests = physical_bank_scores(visible_vectors, int_query, visible_indices)
        physical_exact = np.array_equal(physical_scores, reference_scores)
        all_bank_requests.update(bank_requests)
        count_range(all_bank_requests, POOL_BYTES, dimension)
        count_range(all_bank_requests, POOL_BYTES, dimension)
        result_bytes = ((top_k * 8 + TRANSFER_BYTES - 1) // TRANSFER_BYTES) * TRANSFER_BYTES
        count_range(all_bank_requests, POOL_BYTES + dimension, result_bytes)
        reference = np.lexsort((np.arange(len(visible_indices)), -reference_scores))[:top_k]
        if not np.array_equal(selected, reference) or not np.array_equal(scores, reference_scores[reference]):
            raise ValueError(f'bank-local int32 merge differs for {question["question_id"]}')
        cosine = vectors @ query
        dense_order = sorted(selected.tolist(), key=lambda index: (-float(cosine[index]), index))
        candidate_indices, _ = hybrid_candidates(
            [atoms[index] for index in visible_indices], question['question'], dense_order,
            count=source['configuration']['dense_event_count'])
        candidate_ids = [atoms[visible_indices[index]]['event_id'] for index in candidate_indices]
        recorded_ids = source_row['retrieval_audit']['candidate_events']
        equal = candidate_ids == recorded_ids
        phases = [phase for phase in traffic['phases'] if phase['question_id'] == question['question_id']]
        checks = {'shared_values_exact': values_equal, 'physical_bank_int32_exact': physical_exact,
                  'source_candidates_equal': equal,
                  'producer_byte_accounting': phases[0]['write_requests'] * TRANSFER_BYTES == (len(missing) + 1) * dimension,
                  'scan_byte_accounting': phases[1]['read_requests'] * TRANSFER_BYTES == visible_vectors.nbytes + dimension,
                  'source_feature_misses_equal': features.metrics()['misses'] == source_row['feature_cache']['misses'],
                  'closure_selected_identities_present': {atom['identity'] for atom in source_row['selected_atoms']}
                       <= {atom['identity'] for atom in visible_atoms}}
        replay_matches.append(equal)
        rows.append({'question_id': question['question_id'], 'visible_vectors': len(visible_indices),
                     'checks': checks, 'producer_vectors': len(missing),
                     'top_k': top_k, 'int32_reference_exact': True,
                     'candidate_event_sequence_equal': equal,
                     'candidate_event_sequence_sha256': hashlib.sha256(
                         json.dumps(candidate_ids, separators=(',', ':')).encode()).hexdigest(),
                     'recorded_candidate_event_sequence_sha256': hashlib.sha256(
                         json.dumps(recorded_ids, separators=(',', ':')).encode()).hexdigest(),
                     'source_row_sha256': hashlib.sha256(json.dumps(
                         source_row, sort_keys=True, ensure_ascii=False).encode()).hexdigest()})
        if len(rows) % 8 == 0:
            (args.output / 'replay.partial.json').write_text(json.dumps({'complete': False, 'rows': rows}, indent=2))
            print(json.dumps({'completed': len(rows), 'total': len(source['rows']),
                              'candidate_exact': sum(replay_matches)}, ensure_ascii=False), flush=True)
    if len(rows) != len(source['rows']):
        raise ValueError('query/result alignment mismatch')
    inventory = [{'identity': atoms[index]['identity'], 'event_id': atoms[index]['event_id'],
                  'address': index * dimension, 'bytes': dimension,
                  'int8_sha256': hashlib.sha256(int_vectors[index].tobytes()).hexdigest()}
                 for index in sorted(committed)]
    (args.output / 'vector_inventory.json').write_text(json.dumps(inventory, indent=2))
    mismatches = [path for path, expected in frozen.items() if file_hash(Path(path)) != expected]
    final_checks = {
        'all_request_checks': all(all(row['checks'].values()) for row in rows),
        'native_bank_access_accounting': dict(all_bank_requests) == traffic['bank_requests'],
        'native_producer_accounting': producer_bytes == traffic['producer_vector_bytes'],
        'native_inventory_exact': {row['identity']: (row['address'], row['bytes']) for row in inventory}
            == {identity: (row['address'], row['bytes']) for identity, row in expected_inventory.items()},
        'frozen_sources_unchanged': not mismatches,
        'embedding_weights_unchanged': model_hashes(EMBEDDER) == provenance['embedding_weights'],
        'pooling_unchanged': file_hash(EMBEDDER / '1_Pooling/config.json') == provenance['embedding_pooling_sha256'],
    }
    replay = {'complete': True, 'source_result_sha256': file_hash(args.result),
              'shared_vector_file_sha256': file_hash(vector_store), 'vector_inventory_sha256': file_hash(args.output / 'vector_inventory.json'),
              'source_question_ids': source['selected_question_ids'], 'rows': rows,
              'checks': final_checks, 'frozen_source_mismatches': mismatches,
              'runtime': {'torch': torch.__version__, 'numpy': np.__version__,
                          'device': torch.cuda.get_device_name(args.device),
                          'cuda_runtime': torch.version.cuda, 'torch_threads': torch.get_num_threads()},
              'primary_vector_producer_bytes': producer_bytes, 'unique_vectors': len(committed),
              'summary': {'questions': len(rows), 'all_int32_references_exact': True,
                          'identical_recorded_candidate_event_sequences': sum(replay_matches),
                          'total_candidate_event_sequences': len(rows)},
              'boundary': 'chronological embedding values materialized once in one primary INT8 image, sharing registered identities/addresses with native-qualified host trace; exact physical-bank partial reduction and integer functionality checked. This is not current-service integration or LPDDR-PIM command/timing execution; no calibrated device-time or E2E speedup is claimed.'}
    replay['functional_store_contract_pass'] = all(final_checks.values())
    (args.output / 'replay.json').write_text(json.dumps(replay, ensure_ascii=False, indent=2))
    (args.output / 'completion.json').write_text(json.dumps({
        'complete': True, 'source_question_ids': replay['source_question_ids'],
        'functional_store_contract_pass': replay['functional_store_contract_pass'],
        'source_result_sha256': replay['source_result_sha256'],
        'vector_store_sha256': replay['shared_vector_file_sha256'],
        'allows_development128': False, 'pim_backend_qualified': False,
        'device_calibrated': False, 'publication_ready': False}, indent=2))
    print(json.dumps({'functional_store_contract_pass': replay['functional_store_contract_pass'],
                      **replay['summary']}, ensure_ascii=False))
    if not replay['functional_store_contract_pass']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
