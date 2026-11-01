"""Serving-artifact replay of exact query-local evidence union state reuse."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import statistics
import time

from audit_omni_smoke_closure import digest, verify_source_snapshot
from omni_evidence_planner import select_hyperedges


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=12)
    parser.add_argument('--frozen-planner', type=Path, required=True)
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.limit < 1 or args.repetitions < 1:
        raise ValueError('fresh output, positive prefix and repetitions required')
    from transformers import AutoProcessor
    run = json.loads(args.run.read_text())
    registration = json.loads(args.run.with_suffix('.registration.json').read_text())
    if (not run.get('complete') or run.get('evidence_policy_name') != 'local_hyperedge'
            or run.get('evaluation_label_files_read') != [] or not verify_source_snapshot(run, registration)):
        raise ValueError('complete immutable local-hyperedge run required')
    frozen_spec = importlib.util.spec_from_file_location('frozen_omni_evidence_planner', args.frozen_planner)
    frozen_module = importlib.util.module_from_spec(frozen_spec)
    frozen_spec.loader.exec_module(frozen_module)
    audit = json.loads((args.prepared / 'audit.json').read_text())
    graph_path = args.prepared / 'omni_graph.json'
    if digest(graph_path) != audit['artifacts_sha256']['omni_graph.json']:
        raise ValueError('observable graph changed')
    person_id = run['rows'][0]['person_id']
    graph = next(graph for graph in json.loads(graph_path.read_text()) if graph['person_id'] == person_id)
    atoms = {atom['atom_id']: atom for atom in graph['atoms']}
    processor = AutoProcessor.from_pretrained(run['reader'], local_files_only=True)
    rows = []
    for request_index, row in enumerate(run['rows'][:args.limit]):
        expected = row['hypergraph_evidence_plan']
        candidates = expected['candidate_replay']
        scores = [edge['similarity'] for edge in candidates]
        values = {}
        measurements = {'frozen': [], 'cached': []}
        for repetition in range(args.repetitions):
            order = ('frozen', 'cached') if (repetition + request_index) % 2 == 0 else ('cached', 'frozen')
            for label in order:
                planner = frozen_module.select_hyperedges if label == 'frozen' else select_hyperedges
                started = time.perf_counter()
                actual = planner(candidates, scores, atoms, processor.tokenizer,
                    text_budget=expected['text_budget'], max_images=expected['max_images'],
                    require_image=True, image_novelty_weight=expected['image_novelty_weight'])
                measurements[label].append(time.perf_counter() - started)
                values[label] = actual
        old, new = values['frozen'], values['cached']
        fields = ('text', 'images', 'accepted_atom_ids', 'selected_hyperedges', 'text_tokens', 'objective',
                  'atom_coverage_objective', 'image_coverage_objective', 'rejected_hyperedges')
        old_mismatches = [name for name in fields if old[name] != expected[name]]
        new_mismatches = [name for name in fields if new[name] != expected[name]]
        rows.append({'question_id': row['question_id'],
                     'frozen_plan_mismatches': old_mismatches, 'cached_plan_mismatches': new_mismatches,
                     'frozen_seconds': measurements['frozen'], 'cached_seconds': measurements['cached'],
                     'frozen_median_seconds': statistics.median(measurements['frozen']),
                     'cached_median_seconds': statistics.median(measurements['cached']),
                     'original_concurrent_planning_seconds': row['evidence_planning_seconds'],
                     'union_state': new['query_local_union_state_reuse']})
        print(json.dumps(rows[-1]), flush=True)
    result = {'complete': True, 'rows': rows,
              'requested_limit': args.limit, 'source_requests': len(run['rows']),
              'repetitions': args.repetitions,
              'execution_order': 'alternates frozen/cached order across source requests and repetitions',
              'all_frozen_exact_plans_match': all(not row['frozen_plan_mismatches'] for row in rows),
              'all_cached_exact_plans_match': all(not row['cached_plan_mismatches'] for row in rows),
              'run_sha256': digest(args.run), 'graph_sha256': digest(graph_path),
              'script_sha256': digest(Path(__file__)),
              'planner_sha256': digest(Path(__file__).with_name('omni_evidence_planner.py')),
              'frozen_planner_sha256': digest(args.frozen_planner),
              'evaluation_label_files_read': [],
              'boundary': 'same exact candidate operands and tokenizer; CPU replay versus old concurrent service is not isolated speedup; input/answer equality inferred only when all selected evidence fields match; cache state counters are not semantic-plan changes'}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
