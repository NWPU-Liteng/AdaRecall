"""Post-serving diagnosis of whole-development joint-index candidate coverage."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from audit_omni_smoke_closure import digest


def coverage(expected_sessions, expected_images, candidate_atoms):
    sessions = {atom['session_id'] for atom in candidate_atoms}
    images = {ref.get('resolved', ref['normalized']) for atom in candidate_atoms
              for ref in atom.get('image_refs', []) if ref.get('status') in {'exact', 'unique_user_basename'}}
    return {'source_session_recall': len(expected_sessions & sessions) / len(expected_sessions)
            if expected_sessions else None,
            'image_recall': len(expected_images & images) / len(expected_images) if expected_images else None,
            'missing_source_sessions': sorted(expected_sessions - sessions),
            'missing_images': sorted(expected_images - images)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--replay', type=Path, required=True)
    parser.add_argument('--dense-replay', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('fresh diagnostic output required')
    audit = json.loads((args.prepared / 'audit.json').read_text())
    for name in ('labels.jsonl', 'manifests.jsonl', 'omni_graph.json'):
        if digest(args.prepared / name) != audit['artifacts_sha256'][name]:
            raise ValueError('evaluation or serving source changed')
    replay = json.loads(args.replay.read_text())
    if not replay.get('complete') or replay.get('evaluation_label_files_read') != []:
        raise ValueError('complete label-isolated replay required')
    if replay['prepared_audit_sha256'] != digest(args.prepared / 'audit.json'):
        raise ValueError('prepared release differs from retrieval replay')
    dense_rows = {}
    if args.dense_replay:
        if digest(args.dense_replay) != replay['reference_replay_sha256']:
            raise ValueError('dense reference producer differs')
        dense = json.loads(args.dense_replay.read_text())
        dense_rows = {row['question_id']: row for user in dense['users'] for row in user['rows']}
    labels = {row['question_id']: row for row in map(json.loads, (args.prepared / 'labels.jsonl').read_text().splitlines())}
    graphs = {graph['person_id']: {atom['atom_id']: atom for atom in graph['atoms']}
              for graph in json.loads((args.prepared / 'omni_graph.json').read_text())}
    manifests = {row['question_id']: row for row in map(json.loads, (args.prepared / 'manifests.jsonl').read_text().splitlines())}
    rows, gaps = [], Counter()
    for row in replay['rows']:
        manifest = manifests[row['question_id']]
        if manifest['split'] != 'development' or manifest['person_id'] != row['person_id']:
            raise ValueError('only own-user development requests may be evaluated')
        label = labels[row['question_id']]
        atoms = graphs[row['person_id']]
        candidate_atoms = set(row['candidate_atom_ids'])
        candidate_sessions = {atoms[atom_id]['session_id'] for atom_id in candidate_atoms}
        candidate_images = {ref.get('resolved', ref['normalized']) for atom_id in candidate_atoms
                            for ref in atoms[atom_id].get('image_refs', [])
                            if ref.get('status') in {'exact', 'unique_user_basename'}}
        gold_sessions = set(label['source_session_ids'])
        gold_images = {ref.get('resolved', ref['normalized']) for ref in label['evidence_image_refs']
                       if ref.get('status') in {'exact', 'unique_user_basename'}}
        session_recall = len(gold_sessions & candidate_sessions) / len(gold_sessions) if gold_sessions else None
        image_recall = len(gold_images & candidate_images) / len(gold_images) if gold_images else None
        if not gold_images:
            gap = 'no_label_image_reference'
        elif not gold_images <= candidate_images:
            gap = 'joint_top16_image_candidate_missing'
        else:
            gap = 'all_label_images_in_joint_candidates'
        gaps[gap] += 1
        rows.append({'question_id': row['question_id'], 'person_id': row['person_id'],
                     'question_type': label.get('question_type'),
                     'candidate_count': len(row['edge_ids']),
                     'gold_source_session_count': len(gold_sessions),
                     'candidate_source_session_recall': session_recall,
                     'gold_image_count': len(gold_images), 'candidate_image_recall': image_recall,
                     'gap_category': gap,
                     'missing_source_sessions': sorted(gold_sessions - candidate_sessions),
                     'missing_images': sorted(gold_images - candidate_images)})
        if dense_rows:
            retrieved = set(dense_rows[row['question_id']]['retrieved_session_ids'])
            dense_atoms = [atom for atom in atoms.values() if atom['session_id'] in retrieved]
            joint_atoms = [atoms[atom_id] for atom_id in candidate_atoms]
            rows[-1]['dense_top4_full_session_pool'] = coverage(gold_sessions, gold_images, dense_atoms)
            rows[-1]['union_pool'] = coverage(gold_sessions, gold_images, dense_atoms + joint_atoms)
    result = {'complete': True, 'requests': len(rows), 'rows': rows,
              'gap_counts': dict(gaps),
              'mean_session_recall': sum(row['candidate_source_session_recall'] for row in rows
                                         if row['candidate_source_session_recall'] is not None) /
                                     sum(row['candidate_source_session_recall'] is not None for row in rows),
              'image_defined_requests': sum(row['gold_image_count'] > 0 for row in rows),
              'mean_image_recall_defined': sum(row['candidate_image_recall'] for row in rows
                                               if row['candidate_image_recall'] is not None) /
                                           sum(row['candidate_image_recall'] is not None for row in rows),
              'category_counts': dict(Counter(row['question_type'] for row in rows)),
              'inputs_sha256': {str(path): digest(path) for path in
                                (args.prepared / 'audit.json', args.prepared / 'labels.jsonl',
                                 args.prepared / 'manifests.jsonl', args.prepared / 'omni_graph.json', args.replay)},
              'evaluation_label_files_read': ['labels.jsonl'],
              'boundary': 'post-serving candidate coverage only; source/session membership is not factual sufficiency and no labels entered retrieval'}
    if dense_rows:
        result['pool_boundary'] = 'dense full-session pool versus top16 image-bearing exchanges have unequal candidate bytes; this is omission localization, not a fair-budget ranking comparison'
        result['pool_summaries'] = {}
        for scope in ('dense_top4_full_session_pool', 'union_pool'):
            result['pool_summaries'][scope] = {
                'mean_source_session_recall_defined': sum(row[scope]['source_session_recall'] for row in rows
                    if row[scope]['source_session_recall'] is not None) /
                    sum(row[scope]['source_session_recall'] is not None for row in rows),
                'mean_image_recall_defined': sum(row[scope]['image_recall'] for row in rows
                    if row[scope]['image_recall'] is not None) /
                    sum(row[scope]['image_recall'] is not None for row in rows)}
        result['inputs_sha256'][str(args.dense_replay)] = digest(args.dense_replay)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({key: result[key] for key in ('requests', 'gap_counts', 'mean_session_recall',
          'mean_image_recall_defined')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
