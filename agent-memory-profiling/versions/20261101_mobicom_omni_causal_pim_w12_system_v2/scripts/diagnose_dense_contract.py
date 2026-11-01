"""Replay embedding candidates against an exact FP32 reference, without labels."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from pim_backend import bank_int8_topk
from mobilemem_runtime import hybrid_candidates
from run_mobilemem_text import EMBEDDER, READER, FeatureCache, atomize, model_hashes
from model_identity import require_model_roles


def event_ranks(atoms, ordered_indices):
    ranks = {}
    for atom_rank, index in enumerate(ordered_indices, start=1):
        event = atoms[int(index)]['event_id']
        if event not in ranks:
            ranks[event] = {'event_rank': len(ranks) + 1, 'best_atom_rank': atom_rank}
    return ranks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('fresh numerical diagnosis required')
    if not (EMBEDDER / 'config_sentence_transformers.json').is_file():
        raise ValueError('embedding model identity is not the frozen sentence-transformer model')
    payload = json.loads(args.result.read_text())
    if not payload.get('complete'):
        raise ValueError('complete source run required')
    if [row['question_id'] for row in payload['rows']] != payload['selected_question_ids']:
        raise ValueError('source rows do not match registration')
    require_model_roles(payload['provenance'])
    if model_hashes(EMBEDDER) != payload['provenance']['embedding_weights']:
        raise ValueError('replay embedding weights differ from recorded model')
    prepared_audit = args.prepared / 'audit.json'
    if hashlib.sha256(prepared_audit.read_bytes()).hexdigest() != payload['provenance']['prepared_audit_sha256']:
        raise ValueError('replay prepared audit differs from recorded data')
    for name, expected in json.loads(prepared_audit.read_text())['artifacts_sha256'].items():
        if hashlib.sha256((args.prepared / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'replay prepared source changed: {name}')
    questions = {item['question_id']: item for item in map(json.loads,
                 (args.prepared / 'manifests.jsonl').read_text().splitlines())}
    selected = [questions[identity] for identity in payload['selected_question_ids']]
    if any(question['split'] != 'development' for question in selected):
        raise ValueError('development-only numerical replay')
    for name in ('run_mobilemem_text.py', 'mobilemem_runtime.py'):
        if hashlib.sha256((ROOT / 'scripts' / name).read_bytes()).hexdigest() != payload['provenance']['code_sha256'][name]:
            raise ValueError('numerical replay source does not match recorded candidate code')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from transformers import AutoProcessor
    tokenizer = AutoProcessor.from_pretrained(str(READER), local_files_only=True).tokenizer
    graph = next(graph for graph in json.loads((args.prepared / 'mobilemem_graph.json').read_text())
                 if graph['person_id'] == selected[0]['person_id'])
    atoms, _ = atomize(graph, tokenizer)
    provenance = payload['provenance']
    features = FeatureCache(EMBEDDER, args.device, payload['configuration']['feature_capacity_mib'] * 1024**2,
                            json.dumps(provenance['embedding_weights'], sort_keys=True))
    banks = json.loads((ROOT / 'pim_backend/mobile_lpddr5_bank_profile.json').read_text())['config']
    bank_count = banks['channels'] * banks['banks_per_channel']
    rows = []
    for question, source in zip(selected, payload['rows']):
        visible = [atom for atom in atoms if atom['timestamp'] <= question['effective_timestamp']]
        vectors, _ = features.materialize(visible)
        query = features.encode(['Instruct: Given a personal memory question, retrieve supporting historical messages.\nQuery: '
                                 + question['question']])[0]
        int_vectors = np.clip(np.rint(vectors * 127), -127, 127).astype(np.int8)
        int_query = np.clip(np.rint(query * 127), -127, 127).astype(np.int8)
        top_k = min(128, len(visible))
        approximate, _ = bank_int8_topk(int_vectors, int_query, banks=bank_count, top_k=top_k)
        scores = vectors @ query
        exact_order = np.lexsort((np.arange(len(visible)), -scores)).tolist()
        approximate_order = sorted(approximate.tolist(), key=lambda index: (-float(scores[index]), index))
        integer_scores = int_vectors.astype(np.int32) @ int_query.astype(np.int32)
        integer_order = np.lexsort((np.arange(len(visible)), -integer_scores)).tolist()
        if approximate.tolist() != integer_order[:top_k]:
            raise ValueError('replay bank scan differs from full integer reference')
        event_count = payload['configuration'].get('dense_event_count', 64)
        approximate_indices, _ = hybrid_candidates(visible, question['question'], approximate_order, count=event_count)
        exact_indices, _ = hybrid_candidates(visible, question['question'], exact_order[:top_k], count=event_count)
        approximate_events = {visible[index]['event_id'] for index in approximate_indices}
        exact_events = {visible[index]['event_id'] for index in exact_indices}
        recorded_events = set(source.get('retrieval_audit', {}).get('candidate_events', []))
        rows.append({'question_id': question['question_id'], 'visible_atoms': len(visible),
                     'fp32_event_ranks': event_ranks(visible, exact_order),
                     'int8_event_ranks': event_ranks(visible, integer_order),
                     'candidate_event_count': len(approximate_events),
                     'requested_event_budget': event_count,
                     'atom_scan_top_k': top_k,
                     'recorded_candidates_reproduced': approximate_events == recorded_events,
                     'int8_candidates': sorted(approximate_events),
                     'fp32_candidates': sorted(exact_events),
                     'fp32_events_missing_after_int8': sorted(exact_events - approximate_events),
                     'int8_only_events': sorted(approximate_events - exact_events),
                     'fp32_top128_atom_recall_in_int8_top128': len(set(exact_order[:top_k]) & set(approximate.tolist())) / top_k})
        print(json.dumps({key: value for key, value in rows[-1].items()
                          if key in {'question_id', 'recorded_candidates_reproduced',
                                     'fp32_top128_atom_recall_in_int8_top128'}}, ensure_ascii=False), flush=True)
    output = {'complete': True, 'rows': rows,
              'provenance': {'result_sha256': hashlib.sha256(args.result.read_bytes()).hexdigest(),
                             'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                             'embedding_weights': provenance['embedding_weights'],
                             'prepared_audit_sha256': provenance['prepared_audit_sha256'],
                             'source_checks_passed': True},
              'summary': {'requests': len(rows),
                          'reproduced_requests': sum(row['recorded_candidates_reproduced'] for row in rows),
                          'fp32_events_missing_after_int8': sum(len(row['fp32_events_missing_after_int8']) for row in rows)},
              'boundary': 'label-free numerical candidate-contract check, not an answer-quality comparison or serving change; no model, budget, or ranking optimization'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
