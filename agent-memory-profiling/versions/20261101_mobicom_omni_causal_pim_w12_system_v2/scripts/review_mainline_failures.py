"""Post-run diagnosis of saved development answers; never used by serving."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from diagnose_omni_cohort_evidence import diagnose_row, image_members


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepared = args.root / 'artifacts/omni_paper_scope_v3_20261021/prepared'
    run_roots = {
        'v1': args.root / 'artifacts/evidence_compiler_mainline_dev3x64_v1_20261024',
        'v2': args.root / 'artifacts/evidence_compiler_mainline_dev3x64_v2_20261025',
    }
    cohort = read(run_roots['v2'] / 'cohort.json')['question_ids_by_user']
    question_ids = {identity for identities in cohort.values() for identity in identities}
    labels = {}
    for line in (prepared / 'labels.jsonl').read_text().splitlines():
        label = json.loads(line)
        if label['question_id'] in question_ids:
            labels[label['question_id']] = label
    graphs = {graph['person_id']: graph for graph in read(prepared / 'omni_graph.json')
              if graph['person_id'] in cohort}
    result = {'scope': 'saved 192 development requests only; no new inference',
              'boundary': 'observational attribution; image labels are not exhaustive necessity labels; no causal claim',
              'runs': {}, 'sources_sha256': {}, 'paired_v1_v2': {}, 'image_parent_structure': {}}
    saved = {}
    for version, root in run_roots.items():
        judge = read(root / 'judge.json')
        judged = {(Path(row['run']).name, row['question_id']): row for row in judge['rows']}
        for policy in ('reference', 'candidate'):
            name = version + '_' + policy
            rows = []
            for user, identities in cohort.items():
                path = root / f'{policy}_user_{user}.json'
                run = read(path)
                assert run['complete'] and [row['question_id'] for row in run['rows']] == identities
                result['sources_sha256'][str(path.relative_to(args.root))] = digest(path)
                atoms = {atom['atom_id']: atom for atom in graphs[user]['atoms']}
                for row in run['rows']:
                    judgment = judged[(path.name, row['question_id'])]
                    assert judgment['prediction_sha256'] == row['answer_sha256']
                    plan = row['hypergraph_evidence_plan']
                    row = {**row, **diagnose_row(row, labels[row['question_id']], atoms),
                           'correct': judgment['parse_valid'] and judgment['judge_label'] == 'CORRECT',
                           'invalid': not judgment['parse_valid'],
                           'has_image_label': bool(labels[row['question_id']]['evidence_image_refs']),
                           'question_type': labels[row['question_id']]['question_type']}
                    if version == 'v2' and policy == 'candidate':
                        index_path = args.root / f'artifacts/evidence_compiler_index_dev3_20261024_v4/index_user_{user}'
                        if user not in saved:
                            metadata = read(index_path / 'index.json')
                            payload = np.load(index_path / 'evidence_view_vectors.npy', allow_pickle=False)
                            saved[user] = ({view['view_id']: payload[position] for position, view
                                            in enumerate(metadata['evidence_views'])}, metadata)
                        vectors_by_view, _ = saved[user]
                        winners = plan['winning_evidence_views']
                        vectors = np.stack([vectors_by_view[item['view_id']] for item in winners])
                        selected_edges = {edge['edge_id'] for edge in plan['selected_hyperedges']}
                        selected = [position for position, item in enumerate(winners)
                                    if item['edge_id'] in selected_edges]
                        affinity = np.clip(vectors @ vectors.T, 0., 1.)
                        selected_coverage = affinity[:, selected].max(axis=1) if selected else np.zeros(len(winners))
                        weights = np.maximum([item['score'] for item in winners], 0.) ** 2
                        row['fraction_of_all_candidate_surrogate'] = float(weights @ selected_coverage / weights.sum())
                        row['image_winning_views'] = sum(item['view_type'] == 'image' for item in winners)
                        row['selected_exchange_count'] = len(selected)
                    rows.append(row)
            saved[name] = {row['question_id']: row for row in rows}
            groups = {}
            for group in ('all', 'with_image_label', 'without_image_label'):
                subset = [row for row in rows if group == 'all' or
                          row['has_image_label'] == (group == 'with_image_label')]
                groups[group] = {
                    'requests': len(subset), 'correct': sum(row['correct'] for row in subset),
                    'invalid': sum(row['invalid'] for row in subset),
                    'image_gap_counts': dict(Counter(row['image_reference_gap_category'] for row in subset)),
                }
            services = sum(row['service_seconds'] for row in rows)
            result['runs'][name] = {
                'groups': groups,
                'per_category': {category: {'requests': sum(row['question_type'] == category for row in rows),
                    'correct': sum(row['correct'] for row in rows if row['question_type'] == category)}
                    for category in sorted({row['question_type'] for row in rows})},
                'service_seconds': services,
                'decode_fraction': sum(row['decode_seconds'] for row in rows) / services,
                'prefill_fraction': sum(row['prefill_seconds'] for row in rows) / services,
                'planner_fraction': sum(row['evidence_planning_seconds'] for row in rows) / services,
                'output_tokens': sum(len(row['answer_token_ids']) for row in rows),
                'mean_decode_seconds_per_output_token': sum(row['decode_seconds'] for row in rows) /
                    sum(len(row['answer_token_ids']) for row in rows),
                'requests_truncated': sum(row['stop_reason'] != 'eos' for row in rows),
                'peak_cuda_allocated_bytes': max(row['service_peak_cuda_allocated_bytes'] for row in rows),
            }
            if name == 'v2_candidate':
                result['runs'][name]['surrogate'] = {
                    'mean_fraction_of_all_candidate_surrogate': float(np.mean([
                        row['fraction_of_all_candidate_surrogate'] for row in rows])),
                    'mean_selected_exchange_count': float(np.mean([row['selected_exchange_count'] for row in rows])),
                    'image_winning_views': sum(row['image_winning_views'] for row in rows),
                    'total_winning_views': sum(len(row['hypergraph_evidence_plan']['winning_evidence_views']) for row in rows),
                }
    for group in ('all', 'with_image_label', 'without_image_label'):
        pairs = [(saved['v1_candidate'][identity], saved['v2_candidate'][identity]) for identity in sorted(question_ids)
                 if group == 'all' or bool(labels[identity]['evidence_image_refs']) == (group == 'with_image_label')]
        losses = [after['question_id'] for before, after in pairs if before['correct'] and not after['correct']]
        wins = [after['question_id'] for before, after in pairs if not before['correct'] and after['correct']]
        result['paired_v1_v2'][group] = {'requests': len(pairs), 'losses': losses, 'wins': wins}
    for user in cohort:
        atoms = {atom['atom_id']: atom for atom in graphs[user]['atoms']}
        _, metadata = saved[user]
        expected = {ref.get('resolved', ref['normalized']) for identity in cohort[user]
                    for ref in labels[identity]['evidence_image_refs']}
        parents = {image: [] for image in expected}
        for edge in metadata['edges']:
            images = image_members([atoms[identity] for identity in edge['atom_ids']])
            for image in images & expected:
                parents[image].append({'edge_id': edge['edge_id'], 'image_count': len(images),
                                      'atom_count': len(edge['atom_ids'])})
        result['image_parent_structure'][user] = {
            'unique_label_images': len(expected),
            'images_absent_from_all_index_edges': [image for image, owners in parents.items() if not owners],
            'images_with_only_over_4_image_parents': [image for image, owners in parents.items()
                if owners and all(owner['image_count'] > 4 for owner in owners)],
            'parents': parents,
        }
    result['sources_sha256'][str(Path(__file__).resolve().relative_to(args.root))] = digest(Path(__file__))
    with args.output.open('x') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps({key: value for key, value in result.items()
                      if key not in {'sources_sha256', 'image_parent_structure'}}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
