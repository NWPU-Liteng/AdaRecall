"""Measure exact embedding producers over every authorized development query."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

import torch

from omni_evidence_planner import build_local_hyperedges
from omni_embedding_producer import resolve_producers
from omni_embedding_store import EdgeEmbeddingStore
from omni_visual_feature_store import VisualFeatureStore
from run_omni_raw_image_smoke import DEFAULT_EMBEDDER, token_vector


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--embedder', type=Path, default=DEFAULT_EMBEDDER)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--users', nargs='+', default=['0', '1', '2'])
    parser.add_argument('--cache-bytes', type=int, default=64 * 1024**2)
    parser.add_argument('--candidate-sessions', type=int, default=4)
    args = parser.parse_args()
    if (args.output.exists() or len(args.users) != len(set(args.users))
            or not args.users or args.cache_bytes < 1 or args.candidate_sessions < 1):
        raise ValueError('fresh output, unique users and positive budgets required')
    audit = json.loads((args.prepared / 'audit.json').read_text())
    for name in ('omni_graph.json', 'manifests.jsonl', 'split.json'):
        if digest(args.prepared / name) != audit['artifacts_sha256'][name]:
            raise ValueError('prepared input digest changed')
    split = json.loads((args.prepared / 'split.json').read_text())
    if not set(args.users) <= set(split['development_people']):
        raise ValueError('only authorized development users may be replayed')
    manifests = [json.loads(line) for line in (args.prepared / 'manifests.jsonl').read_text().splitlines()]
    selected = [row for row in manifests if row['person_id'] in args.users and row['split'] == 'development']
    if not selected or len({row['question_id'] for row in selected}) != len(selected):
        raise ValueError('complete unique source-order development stream required')
    files = sorted(path for path in args.embedder.iterdir()
                   if path.is_file() and path.suffix in ('.json', '.safetensors', '.txt'))
    model_identity = {path.name: digest(path) for path in files}
    execution_identity = hashlib.sha256(json.dumps(model_identity, sort_keys=True).encode()).hexdigest()
    graphs = {row['person_id']: row for row in json.loads((args.prepared / 'omni_graph.json').read_text())
              if row['person_id'] in args.users}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    protocol = {
        'complete': False, 'status': 'registered_before_execution',
        'question_ids': [row['question_id'] for row in selected],
        'prepared_audit_sha256': digest(args.prepared / 'audit.json'),
        'model_files_sha256': model_identity, 'script_sha256': digest(Path(__file__)),
        'cache_bytes_per_user': args.cache_bytes, 'candidate_sessions': args.candidate_sessions,
        'selection': 'all authorized development questions in original source order; each user starts cold',
        'visibility': 'history_end; source-order QA replay is not real query arrival time',
        'evaluation_label_files_read': [],
        'boundary': 'embedding producer opportunity and measured stage costs only; no Reader or PIM speedup',
    }
    args.output.with_suffix('.registration.json').write_text(json.dumps(protocol, indent=2))
    from transformers import AutoModel, AutoTokenizer
    import transformers

    torch.set_num_threads(2)
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(str(args.embedder), local_files_only=True, padding_side='left')
    model = AutoModel.from_pretrained(str(args.embedder), local_files_only=True,
                                    dtype=torch.bfloat16, attn_implementation='sdpa').to(args.device).eval()
    users = []
    for person in args.users:
        graph = graphs[person]
        atoms_by_session = defaultdict(list)
        atoms_by_id = {atom['atom_id']: atom for atom in graph['atoms']}
        for atom in graph['atoms']:
            atoms_by_session[atom['session_id']].append(atom)
        sessions = graph['sessions']
        chunks, owners = [], []
        for session_index, session in enumerate(sessions):
            text = '\n'.join(f"{atom['role']}: {atom['content']}"
                             for atom in atoms_by_session[session['session_id']]
                             if atom.get('content_type') == 'text' and atom.get('content'))
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            for begin in range(0, max(1, len(token_ids)), 384):
                chunks.append(tokenizer.decode(token_ids[begin:begin + 384]))
                owners.append(session_index)
        construction_started = time.perf_counter()
        vectors = torch.cat([token_vector(chunks[begin:begin + 8], tokenizer, model, args.device)
                             for begin in range(0, len(chunks), 8)])
        torch.cuda.synchronize(args.device)
        construction_seconds = time.perf_counter() - construction_started
        store = EdgeEmbeddingStore(args.cache_bytes)
        rows = []
        for request_index, manifest in enumerate(row for row in selected if row['person_id'] == person):
            started = time.perf_counter()
            query_vector = token_vector([manifest['question']], tokenizer, model, args.device)[0]
            similarities = torch.mv(vectors, query_vector).tolist()
            session_scores = torch.full((len(sessions),), -float('inf'), device=args.device)
            for chunk_index, session_index in enumerate(owners):
                session_scores[session_index] = max(float(session_scores[session_index]), similarities[chunk_index])
            indices = torch.topk(session_scores, k=min(args.candidate_sessions, len(sessions)),
                                 largest=True, sorted=True).indices.tolist()
            ranked = [sessions[index]['session_id'] for index in indices]
            edges = [edge for edge in build_local_hyperedges(ranked, atoms_by_session, radius=2)
                     if len(tokenizer.encode(edge['embedding_text'], add_special_tokens=True)) <= 2048]
            torch.cuda.synchronize(args.device)
            retrieval_seconds = time.perf_counter() - started
            started = time.perf_counter()
            _, records = resolve_producers(
                edges, atoms_by_id, store,
                lambda texts: list(token_vector(texts, tokenizer, model, args.device).detach()),
                execution_identity, output_digest=lambda values: VisualFeatureStore.tensor_digest(torch.stack(values)),
                request_index=request_index)
            torch.cuda.synchronize(args.device)
            producer_seconds = time.perf_counter() - started
            row = {'question_id': manifest['question_id'], 'person_id': person,
                   'request_index': request_index, 'retrieved_session_ids': ranked,
                   'retrieval_seconds': retrieval_seconds, 'producer_seconds': producer_seconds,
                   'embedding_producers': records, 'store': store.summary()}
            rows.append(row)
            with args.output.with_suffix('.rows.jsonl').open('a') as handle:
                handle.write(json.dumps(row) + '\n')
            if request_index % 32 == 0:
                print(json.dumps({'person_id': person, 'requests': len(rows), 'store': store.summary()}), flush=True)
        users.append({'person_id': person, 'requests': len(rows), 'construction_seconds': construction_seconds,
                      'store': store.summary(), 'rows': rows})
    result = {**protocol, 'complete': True, 'status': 'completed', 'users': users,
              'runtime': {'torch': torch.__version__, 'transformers': transformers.__version__,
                          'device_name': torch.cuda.get_device_name(args.device)}}
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps({'complete': True, 'requests': sum(user['requests'] for user in users)}), flush=True)


if __name__ == '__main__':
    main()
