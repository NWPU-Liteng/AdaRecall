"""Rebuild frozen embedding candidates before a label-free image-quota intervention."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

from audit_omni_smoke_closure import digest, verify_source_snapshot
from omni_embedding_producer import resolve_producers
from omni_embedding_store import EdgeEmbeddingStore
from omni_evidence_planner import bound_hyperedges, build_local_hyperedges, select_hyperedges


PLAN_FIELDS = ('text', 'images', 'accepted_atom_ids', 'selected_hyperedges',
               'rejected_hyperedges', 'text_tokens', 'objective')


def candidate_summary(edges, atoms):
    images = set()
    sessions = set()
    for edge in edges:
        for identity in edge['atom_ids']:
            atom = atoms[identity]
            sessions.add(atom['session_id'])
            images.update(ref.get('resolved', ref['normalized']) for ref in atom.get('image_refs', [])
                          if ref.get('status') in ('exact', 'unique_user_basename'))
    return {'edge_ids': [edge['edge_id'] for edge in edges],
            'image_ids': sorted(images), 'session_ids': sorted(sessions)}


def replay_bounds(edges, scores, atoms, expected, tokenizer):
    control, control_scores, pruned = bound_hyperedges(
        edges, scores, atoms, expected['candidate_hyperedge_limit'],
        image_edges_per_session=expected['image_candidate_quota_per_session'])
    expected_candidates = expected['candidate_replay']
    mismatches = []
    if [edge['edge_id'] for edge in control] != [edge['edge_id'] for edge in expected_candidates]:
        mismatches.append('control_candidate_order')
    if set(pruned) != set(expected['candidate_pruned_edge_ids']):
        mismatches.append('control_pruned_candidates')
    if control_scores != [edge['similarity'] for edge in expected_candidates]:
        mismatches.append('control_similarity_values')
    parameters = {'text_budget': expected['text_budget'], 'max_images': expected['max_images'],
                  'require_image': True, 'image_novelty_weight': expected['image_novelty_weight']}
    control_plan = select_hyperedges(control, control_scores, atoms, tokenizer, **parameters)
    mismatches.extend('control_plan_' + field for field in PLAN_FIELDS
                      if control_plan[field] != expected[field])
    if mismatches:
        return {'control_mismatches': mismatches, 'counterfactual_executed': False}
    intervention, intervention_scores, _ = bound_hyperedges(
        edges, scores, atoms, expected['candidate_hyperedge_limit'], image_edges_per_session=1)
    started = time.perf_counter()
    plan = select_hyperedges(intervention, intervention_scores, atoms, tokenizer, **parameters)
    return {'control_mismatches': [], 'counterfactual_executed': True,
            'pre_bound_candidates': candidate_summary(edges, atoms),
            'control_candidates': candidate_summary(control, atoms),
            'quota_candidates': candidate_summary(intervention, atoms),
            'quota_plan': plan, 'quota_planner_seconds': time.perf_counter() - started,
            'control_selected_image_ids': sorted(ref.get('resolved', ref['normalized'])
                                                 for ref in control_plan['images']),
            'changed_plan_fields': [field for field in PLAN_FIELDS if plan[field] != control_plan[field]]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix('.registration.json').exists():
        raise ValueError('fresh replay output required')
    run = json.loads(args.run.read_text())
    registered = json.loads(args.run.with_suffix('.registration.json').read_text())
    if (not run.get('complete') or run.get('evaluation_label_files_read') != []
            or run.get('evidence_policy_name') != 'local_hyperedge'
            or not verify_source_snapshot(run, registered)):
        raise ValueError('complete immutable label-isolated local-hyperedge run required')
    audit_path = args.prepared / 'audit.json'
    audit = json.loads(audit_path.read_text())
    if digest(audit_path) != run['prepared_audit_sha256']:
        raise ValueError('different prepared release')
    for name in ('omni_graph.json', 'manifests.jsonl', 'split.json'):
        if digest(args.prepared / name) != audit['artifacts_sha256'][name]:
            raise ValueError('prepared input changed')
    split = json.loads((args.prepared / 'split.json').read_text())
    people = {row['person_id'] for row in run['rows']}
    if len(people) != 1 or not people <= set(split['development_people']):
        raise ValueError('one authorized development user per replay required')
    person = next(iter(people))
    manifests = [json.loads(line) for line in (args.prepared / 'manifests.jsonl').read_text().splitlines()]
    request_ids = [row['question_id'] for row in run['rows']]
    queries = [row for row in manifests if row['question_id'] in set(request_ids)]
    if (len(request_ids) != len(set(request_ids)) or [row['question_id'] for row in queries] != request_ids
            or any(row['split'] != 'development' or row['person_id'] != person for row in queries)):
        raise ValueError('unique source-order development requests required')
    graph = next(graph for graph in json.loads((args.prepared / 'omni_graph.json').read_text())
                 if graph['person_id'] == person)
    atoms = {atom['atom_id']: atom for atom in graph['atoms']}
    by_session = defaultdict(list)
    for atom in graph['atoms']:
        by_session[atom['session_id']].append(atom)
    model_path = Path(run['embedder'])
    model_hashes = {path.name: digest(path) for path in sorted(model_path.iterdir())
                    if path.is_file() and path.suffix in ('.json', '.safetensors', '.txt')}
    if model_hashes != run['embedder_identity_files_sha256']:
        raise ValueError('embedding model/processor changed')
    source_paths = [Path(__file__), Path(__file__).with_name('omni_evidence_planner.py'),
                    Path(__file__).with_name('omni_embedding_producer.py'),
                    Path(__file__).with_name('run_omni_raw_image_smoke.py')]
    sources = {str(path): digest(path) for path in source_paths}
    registration = {'complete': False, 'run': str(args.run), 'run_sha256': digest(args.run),
        'prepared_audit_sha256': digest(audit_path), 'model_files_sha256': model_hashes,
        'sources_sha256': sources, 'question_ids': request_ids,
        'intervention': 'existing bound_hyperedges image_edges_per_session=1; all other decisions fixed',
        'scope': 'reconstructed full pre-bound scores; frozen baseline must match exactly per request',
        'evaluation_label_files_read': [],
        'boundary': 'candidate/selection diagnostic only; no answers, no E2E acceleration, no label-selected policy'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix('.registration.json').write_text(json.dumps(registration, indent=2))
    import torch
    from transformers import AutoModel, AutoTokenizer
    from run_omni_raw_image_smoke import token_vector
    from omni_visual_feature_store import VisualFeatureStore
    torch.set_num_threads(2)
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, padding_side='left')
    reader_tokenizer = AutoTokenizer.from_pretrained(run['reader'], local_files_only=True)
    model = AutoModel.from_pretrained(str(model_path), local_files_only=True,
                                     dtype=torch.bfloat16, attn_implementation='sdpa').to(args.device).eval()
    store = EdgeEmbeddingStore(64 * 1024**2)
    rows = []
    for index, (served, query) in enumerate(zip(run['rows'], queries)):
        started = time.perf_counter()
        expected = served['hypergraph_evidence_plan']
        edges = build_local_hyperedges(served['retrieved_session_ids'], by_session, radius=2)
        eligible = [edge for edge in edges if len(tokenizer.encode(
            edge['embedding_text'], add_special_tokens=True)) <= 2048]
        query_vector = token_vector([query['question']], tokenizer, model, args.device)[0]
        vectors, records = resolve_producers(
            eligible, atoms, store, lambda texts: list(token_vector(texts, tokenizer, model, args.device).detach()),
            run['embedding_execution_identity'], batch_size=32,
            output_digest=lambda values: VisualFeatureStore.tensor_digest(torch.stack(values)), request_index=index)
        scores = torch.mv(torch.stack([vectors[edge['edge_id']] for edge in eligible]), query_vector).tolist()
        producer_fields = ('producer_sha256', 'edge_ids', 'output_sha256')
        control_records = expected['embedding_producers']
        producers_match = [{key: row[key] for key in producer_fields} for row in records] == [
            {key: row[key] for key in producer_fields} for row in control_records]
        if producers_match:
            result = replay_bounds(eligible, scores, atoms, expected, reader_tokenizer)
        else:
            result = {'control_mismatches': ['embedding_producer_identity_or_output'],
                      'counterfactual_executed': False}
        row = {'question_id': served['question_id'], 'person_id': person,
               'producer_outputs_exact': producers_match, 'embedding_rejected_edge_ids': sorted(
                   {edge['edge_id'] for edge in edges} - {edge['edge_id'] for edge in eligible}),
               **result, 'diagnostic_seconds': time.perf_counter() - started}
        rows.append(row)
        with args.output.with_suffix('.rows.jsonl').open('a') as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
        print(json.dumps({'request': index + 1, 'question_id': row['question_id'],
                          'control_mismatches': row['control_mismatches']}), flush=True)
    if any(digest(path) != expected_hash for path, expected_hash in (
            (Path(name), expected_hash) for name, expected_hash in sources.items())):
        raise ValueError('diagnostic source changed during replay')
    result = {**registration, 'complete': True, 'rows': rows,
              'all_controls_exact': all(not row['control_mismatches'] for row in rows),
              'counterfactual_requests': sum(row['counterfactual_executed'] for row in rows)}
    with args.output.open('x') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
