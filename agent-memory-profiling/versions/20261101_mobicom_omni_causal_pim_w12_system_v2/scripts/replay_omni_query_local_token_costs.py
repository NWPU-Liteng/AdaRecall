"""Exact saved-candidate plan replay with query-local atom token costs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from audit_omni_smoke_closure import digest, verify_source_snapshot
from omni_atom_token_cost import AtomUnionTokenCosts
from omni_evidence_planner import select_hyperedges


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repetitions', type=int, default=2)
    args = parser.parse_args()
    if args.output.exists() or args.repetitions < 1:
        raise ValueError('fresh output and positive repetitions required')
    run = json.loads(args.run.read_text())
    registration = json.loads(args.run.with_suffix('.registration.json').read_text())
    if (not run.get('complete') or run.get('evaluation_label_files_read') != []
            or run.get('evidence_policy_name') != 'local_hyperedge'
            or not verify_source_snapshot(run, registration)):
        raise ValueError('complete immutable label-isolated local-hyperedge run required')
    prepared_audit = json.loads((args.prepared / 'audit.json').read_text())
    if digest(args.prepared / 'audit.json') != run['prepared_audit_sha256']:
        raise ValueError('prepared release differs')
    graph_path = args.prepared / 'omni_graph.json'
    if digest(graph_path) != prepared_audit['artifacts_sha256']['omni_graph.json']:
        raise ValueError('observable graph changed')
    person = run['rows'][0]['person_id']
    graph = next(graph for graph in json.loads(graph_path.read_text()) if graph['person_id'] == person)
    atoms = {atom['atom_id']: atom for atom in graph['atoms']}
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(run['reader'], local_files_only=True)
    source_files = [Path(__file__), Path(__file__).with_name('omni_atom_token_cost.py'),
                    Path(__file__).with_name('omni_evidence_planner.py')]
    sources = {str(path): digest(path) for path in source_files}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix('.registration.json').open('x') as handle:
        json.dump({'run_sha256': digest(args.run), 'sources_sha256': sources,
                   'repetitions': args.repetitions, 'evaluation_label_files_read': [],
                   'policy': 'fresh atom-cost state for every query and every repetition; alternating arm order'}, handle)
    rows = []
    states = []
    for request_index, row in enumerate(run['rows']):
        expected = row['hypergraph_evidence_plan']
        candidates = expected['candidate_replay']
        edges = [{'edge_id': edge['edge_id'], 'atom_ids': edge['atom_ids']} for edge in candidates]
        scores = [edge['similarity'] for edge in candidates]
        times = {'baseline': [], 'fragment_cost': []}
        plans = {}
        members = {identity: atoms[identity] for edge in edges for identity in edge['atom_ids']}
        for repetition in range(args.repetitions):
            order = ('baseline', 'fragment_cost') if (request_index + repetition) % 2 == 0 else ('fragment_cost', 'baseline')
            for name in order:
                started = time.perf_counter()
                costs = AtomUnionTokenCosts(tokenizer, members) if name == 'fragment_cost' else None
                plans[name] = select_hyperedges(
                    edges, scores, atoms, tokenizer, text_budget=expected['text_budget'],
                    max_images=expected['max_images'], require_image=True,
                    image_novelty_weight=expected['image_novelty_weight'], atom_token_costs=costs)
                times[name].append(time.perf_counter() - started)
                if costs is not None:
                    states.append(costs.summary())
        fields = ('text', 'images', 'accepted_atom_ids', 'selected_hyperedges', 'rejected_hyperedges',
                  'text_tokens', 'objective', 'atom_coverage_objective', 'image_coverage_objective')
        mismatches = [name + ':' + field for name in plans for field in fields
                      if plans[name][field] != expected[field]]
        rows.append({'question_id': row['question_id'], 'plan_mismatches': mismatches,
                     'baseline_seconds': times['baseline'], 'fragment_cost_seconds': times['fragment_cost'],
                     'saved_service_planning_seconds': row['evidence_planning_seconds'],
                     'saved_plan_fields_exact': not mismatches})
        print(json.dumps({'question_id': row['question_id'], 'plan_mismatches': mismatches,
                          'state': states[-1]}), flush=True)
    if any(digest(path) != sources[str(path)] for path in source_files):
        raise ValueError('source changed during replay')
    result = {'complete': True, 'rows': rows,
        'all_plans_exact': all(not row['plan_mismatches'] for row in rows),
        'token_cost_states': states, 'repetitions': args.repetitions,
        'inputs_sha256': {'run': digest(args.run), 'registration': digest(args.run.with_suffix('.registration.json')),
                          'prepared_audit': digest(args.prepared / 'audit.json'),
                          'graph': digest(graph_path)},
        'script_sha256': digest(Path(__file__)),
        'sources_sha256': sources,
        'evaluation_label_files_read': [], 'full_service_speedup': None,
        'boundary': 'shared-host CPU planner function replay, exact saved plans required; alternates arm order; fresh query-local state, preparation inside timed bracket; excludes model initialization/retrieval; not isolated causal full-service, quality or PIM results'}
    with args.output.open('x') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps({'requests': len(rows), 'all_plans_exact': result['all_plans_exact'],
                      'baseline_seconds': sum(sum(row['baseline_seconds']) for row in rows),
                      'fragment_cost_seconds': sum(sum(row['fragment_cost_seconds']) for row in rows)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
