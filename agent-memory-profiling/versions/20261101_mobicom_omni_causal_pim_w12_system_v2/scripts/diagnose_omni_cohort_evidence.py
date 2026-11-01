"""Label-side coverage diagnostics, never an oracle for serving evidence selection."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from audit_omni_smoke_closure import digest, empirical_latency


def image_members(atoms):
    return {ref.get('resolved', ref['normalized']) for atom in atoms
            for ref in atom.get('image_refs', [])
            if ref.get('status') in {'exact', 'unique_user_basename'}}


def recall(expected, selected):
    return len(expected & selected) / len(expected) if expected else None


def diagnose_row(row, label, atoms_by_id):
    plan = row['hypergraph_evidence_plan']
    accepted = [atoms_by_id[identity] for identity in plan['accepted_atom_ids']]
    retrieved_ids = set(row['retrieved_session_ids'])
    retrieved = [atom for atom in atoms_by_id.values() if atom['session_id'] in retrieved_ids]
    candidate_ids = {identity for edge in plan['candidate_replay'] for identity in edge['atom_ids']}
    candidates = [atoms_by_id[identity] for identity in candidate_ids]
    expected_images = {ref.get('resolved', ref['normalized']) for ref in label['evidence_image_refs']}
    expected_sessions = set(label['source_session_ids'])
    selected_images = {image['archive_member'] for image in row['images']}
    candidate_images = image_members(candidates)
    retrieved_images = image_members(retrieved)
    if not expected_images:
        gap = 'no_label_image_reference'
    elif not expected_images <= retrieved_images:
        gap = 'image_reference_missing_from_retrieved_dialogue'
    elif not expected_images <= candidate_images:
        gap = 'image_reference_missing_from_bounded_candidates'
    elif not expected_images <= selected_images:
        gap = 'image_reference_not_selected_within_budget'
    else:
        gap = 'all_label_image_references_selected'
    return {'question_id': row['question_id'],
            'source_session_recall_retrieved': recall(expected_sessions, retrieved_ids),
            'source_session_recall_candidates': recall(expected_sessions, {atom['session_id'] for atom in candidates}),
            'source_session_recall_selected': recall(expected_sessions, {atom['session_id'] for atom in accepted}),
            'label_image_recall_retrieved': recall(expected_images, retrieved_images),
            'label_image_recall_candidates': recall(expected_images, candidate_images),
            'label_image_recall_selected': recall(expected_images, selected_images),
            'image_reference_gap_category': gap}


def main():
    parser = argparse.ArgumentParser()
    for name in ('prepared', 'run-root', 'evaluation-dir', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('fresh diagnostic output required')
    plan_path = args.run_root / 'cohort_plan.json'
    summary_path = args.evaluation_dir / 'summary.json'
    plan, summary = json.loads(plan_path.read_text()), json.loads(summary_path.read_text())
    if digest(plan_path) != summary['cohort_plan_sha256']:
        raise ValueError('cohort plan producer changed')
    prepared_audit = json.loads((args.prepared / 'audit.json').read_text())
    if digest(args.prepared / 'audit.json') != plan['prepared_audit_sha256']:
        raise ValueError('prepared audit producer changed')
    for name in ('omni_graph.json', 'labels.jsonl'):
        if digest(args.prepared / name) != prepared_audit['artifacts_sha256'][name]:
            raise ValueError('diagnostic graph/label producer changed')
    labels = {row['question_id']: row for row in (
        json.loads(line) for line in (args.prepared / 'labels.jsonl').read_text().splitlines())}
    graphs = {graph['person_id']: graph for graph in json.loads((args.prepared / 'omni_graph.json').read_text())}
    diagnostics, raw_rows, per_user, projections = [], [], [], []
    for job in plan['jobs']:
        path = Path(job['output'])
        if digest(path) != summary['reader_sha256'][str(path)]:
            raise ValueError('cohort Reader producer changed')
        smoke = json.loads(path.read_text())
        atoms_by_id = {atom['atom_id']: atom for atom in graphs[job['person_id']]['atoms']}
        for row in smoke['rows']:
            diagnostics.append({'person_id': job['person_id'], **diagnose_row(row, labels[row['question_id']], atoms_by_id)})
        raw_rows.extend(smoke['rows'])
        per_user.append({'person_id': job['person_id'], 'embedding_store': smoke['embedding_state_reuse'],
                         'index_construction_seconds': smoke['construction_seconds'],
                         'model_load_seconds': smoke['model_load_seconds']})
        projection_path = args.evaluation_dir / f"user_{job['person_id']}_joint_capacity.json"
        joint = json.loads(projection_path.read_text())
        if joint['smoke_sha256'] != digest(path):
            raise ValueError('joint-capacity Reader producer changed')
        projections.append({'person_id': job['person_id'], 'sha256': digest(projection_path),
                            'known_reservations_bytes': joint['known_reservations_total_bytes'],
                            'placed_weight_bytes': joint['placed_weight_bytes'],
                            'warm_linear_speedup_scenario': joint['decode_linear_projection']['warm_linear_speedup_scenario'],
                            'cold_linear_speedup_scenario': joint['decode_linear_projection']['cold_linear_speedup_scenario']})
    keys = ('source_session_recall_retrieved', 'source_session_recall_candidates', 'source_session_recall_selected',
            'label_image_recall_retrieved', 'label_image_recall_candidates', 'label_image_recall_selected')
    means = {}
    for name in keys:
        values = [row[name] for row in diagnostics if row[name] is not None]
        means[name] = {'defined_questions': len(values), 'mean': sum(values) / len(values) if values else None}
    hits = sum(row['embedding_store']['hits'] for row in per_user)
    misses = sum(row['embedding_store']['misses'] for row in per_user)
    result = {'schema_version': 'omni-cohort-evidence-diagnostics-v1',
              'cohort_summary_sha256': digest(summary_path), 'cohort_plan_sha256': digest(plan_path),
              'rows': diagnostics, 'coverage_means': means,
              'image_gap_counts': dict(Counter(row['image_reference_gap_category'] for row in diagnostics)),
              'empirical_stages_all_requests': {name: empirical_latency([row[name] for row in raw_rows])
                  for name in ('edge_embedding_seconds', 'evidence_planning_seconds', 'prefill_seconds',
                               'decode_seconds', 'query_retrieval_seconds')},
              'peak_cuda_allocated_bytes': max(row['service_peak_cuda_allocated_bytes'] for row in raw_rows),
              'peak_cuda_reserved_bytes': max(row['service_peak_cuda_reserved_bytes'] for row in raw_rows),
              'embedding_cache_hits': hits, 'embedding_cache_misses': misses,
              'embedding_cache_hit_fraction': hits / (hits + misses),
              'per_user': per_user, 'shared_capacity_scenarios': projections,
              'script_sha256': digest(Path(__file__)),
              'boundary': 'evaluation only; provenance-session and image-reference coverage is not critical-span sufficiency or visual grounding; stage gap labels are descriptive, not causal failure attribution; cache hits are not measured net speedup'}
    with args.output.open('x') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps({'coverage_means': means, 'image_gap_counts': result['image_gap_counts'],
                      'embedding_cache_hit_fraction': result['embedding_cache_hit_fraction']}))


if __name__ == '__main__':
    main()
