"""Post-serving paired diagnosis for dense-session recall width."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from audit_omni_smoke_closure import digest, verify_source_snapshot


def paired_quality(rows, *, bootstrap_samples=20000, seed=20261023):
    if not rows:
        raise ValueError('nonempty paired outcomes required')
    counts = Counter()
    for row in rows:
        before, after = row['reference_correct'], row['expanded_correct']
        counts['both_correct' if before and after else 'reference_only' if before else
               'expanded_only' if after else 'both_wrong'] += 1
    discordant = counts['reference_only'] + counts['expanded_only']
    smaller = min(counts['reference_only'], counts['expanded_only'])
    probability = (min(1.0, 2 * sum(math.comb(discordant, index) for index in range(smaller + 1))
                       / 2 ** discordant) if discordant else 1.0)
    generator = np.random.default_rng(seed)
    samples = np.zeros(bootstrap_samples)
    for person_id in sorted({row['person_id'] for row in rows}):
        differences = np.asarray([int(row['expanded_correct']) - int(row['reference_correct'])
                                  for row in rows if row['person_id'] == person_id])
        samples += generator.choice(differences, size=(bootstrap_samples, len(differences))).sum(axis=1)
    samples /= len(rows)
    return {'requests': len(rows), 'partition': dict(counts),
            'reference_correct': counts['both_correct'] + counts['reference_only'],
            'expanded_correct': counts['both_correct'] + counts['expanded_only'],
            'paired_accuracy_delta': (counts['expanded_only'] - counts['reference_only']) / len(rows),
            'paired_delta_bootstrap_ci95': np.quantile(samples, [0.025, 0.975]).tolist(),
            'exact_mcnemar_two_sided_p': probability,
            'bootstrap': {'samples': bootstrap_samples, 'seed': seed,
                          'scope': 'paired requests resampled within each observed user; conditional on these users, not a cross-user population confidence interval'}}


def image_ids(atoms, atom_ids):
    return {ref.get('resolved', ref['normalized']) for atom_id in atom_ids
            for ref in atoms.get(atom_id, {}).get('image_refs', [])
            if ref.get('status') in {'exact', 'unique_user_basename'}}


def selected_atom_ids(row):
    plan = row.get('hypergraph_evidence_plan', {})
    ids = plan.get('accepted_atom_ids')
    if ids is None:
        raise ValueError(f'missing selected atom ids for {row["question_id"]}')
    return list(ids)


def image_omission_stage(expected_images, retrieved_images, bounded_images, reader_images, correct):
    if not expected_images:
        return 'no_resolved_label_image'
    if not expected_images <= retrieved_images:
        return 'source_session_pool_image_missing'
    if not expected_images <= bounded_images:
        return 'bounded_candidate_or_embedding_budget_image_missing'
    if not expected_images <= reader_images:
        return 'selection_budget_image_missing'
    return 'label_images_retained_answer_correct' if correct else 'label_images_retained_answer_wrong'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--judge', type=Path, required=True)
    parser.add_argument('--candidate-root', type=Path)
    parser.add_argument('--candidate-prefix', default='expanded')
    parser.add_argument('--candidate-judge', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('fresh diagnostic output required')
    prepared = args.prepared
    audit_path = prepared / 'audit.json'
    audit = json.loads(audit_path.read_text())
    source_paths = [prepared / name for name in ('labels.jsonl', 'manifests.jsonl', 'omni_graph.json')]
    for path in source_paths:
        if digest(path) != audit['artifacts_sha256'][path.name]:
            raise ValueError('prepared artifact changed')
    cohort_path = args.run_root / 'cohort.json'
    cohort = json.loads(cohort_path.read_text())
    expected = cohort['question_ids_by_user']
    manifests = {row['question_id']: row for row in
                 map(json.loads, (prepared / 'manifests.jsonl').read_text().splitlines())}
    labels = {row['question_id']: row for row in
              map(json.loads, (prepared / 'labels.jsonl').read_text().splitlines())}
    graphs = {graph['person_id']: {atom['atom_id']: atom for atom in graph['atoms']}
              for graph in json.loads((prepared / 'omni_graph.json').read_text())}
    judge = json.loads(args.judge.read_text())
    if not judge.get('complete'):
        raise ValueError('complete evaluator required')
    judge_rows = {(Path(row['run']).resolve(), row['question_id']): row for row in judge['rows']}
    if len(judge_rows) != len(judge['rows']):
        raise ValueError('duplicate judge record')
    candidate_root = args.candidate_root or args.run_root
    candidate_judge_path = args.candidate_judge or args.judge
    candidate_judge = json.loads(candidate_judge_path.read_text())
    if not candidate_judge.get('complete'):
        raise ValueError('complete candidate evaluator required')
    candidate_judge_rows = {(Path(row['run']).resolve(), row['question_id']): row
                            for row in candidate_judge['rows']}
    if len(candidate_judge_rows) != len(candidate_judge['rows']):
        raise ValueError('duplicate candidate judge record')
    if (judge['provenance']['model_files_sha256'] != candidate_judge['provenance']['model_files_sha256']
            or any(judge['provenance'].get(name) != candidate_judge['provenance'].get(name)
                   for name in ('protocol', 'temperature', 'max_new_tokens', 'omni_prompt_source_sha256'))):
        raise ValueError('evaluator protocol differs')
    runs = {}
    inputs = [audit_path, cohort_path, args.judge, candidate_judge_path, *source_paths]
    for policy in ('reference', 'expanded'):
        runs[policy] = {}
        for person_id, identities in expected.items():
            root = args.run_root if policy == 'reference' else candidate_root
            prefix = 'reference' if policy == 'reference' else args.candidate_prefix
            path = root / f'{prefix}_user_{person_id}.json'
            run = json.loads(path.read_text())
            registration_path = path.with_suffix('.registration.json')
            registration = json.loads(registration_path.read_text())
            if (not run.get('complete') or run.get('evaluation_label_files_read') != []
                    or [row['question_id'] for row in run['rows']] != identities
                    or not verify_source_snapshot(run, registration)):
                raise ValueError(f'incomplete or label-contaminated run: {path}')
            evaluator = judge if policy == 'reference' else candidate_judge
            if evaluator['provenance']['run_sha256'].get(str(path)) != digest(path):
                raise ValueError('judge and serving run identities differ')
            inputs.extend((path, registration_path))
            runs[policy].update({row['question_id']: row for row in run['rows']})
    if set(runs) != {'reference', 'expanded'}:
        raise ValueError('reference and expanded runs required')
    question_ids = [identity for identities in expected.values() for identity in identities]
    if (len(question_ids) != len(set(question_ids)) or set(runs['reference']) != set(question_ids)
            or set(runs['expanded']) != set(question_ids)):
        raise ValueError('whole unique frozen cohort required')

    rows = []
    coverage = Counter()
    omission_stages = {policy: Counter() for policy in ('reference', 'expanded')}
    for question_id in question_ids:
        reference, expanded = runs['reference'][question_id], runs['expanded'][question_id]
        if reference['person_id'] != expanded['person_id']:
            raise ValueError(f'user mismatch for {question_id}')
        atoms = graphs[reference['person_id']]
        if manifests[question_id]['split'] != 'development':
            raise ValueError('only development questions may be diagnosed')
        label = labels[question_id]
        gold_sessions = set(label.get('source_session_ids', []))
        gold_images = {ref.get('resolved', ref['normalized']) for ref in label.get('evidence_image_refs', [])
                       if ref.get('status') in {'exact', 'unique_user_basename'}}
        ref_ids = selected_atom_ids(reference)
        exp_ids = selected_atom_ids(expanded)
        ref_retrieved = set(reference['retrieved_session_ids'])
        exp_retrieved = set(expanded['retrieved_session_ids'])
        ref_selected_sessions = {atoms[atom_id]['session_id'] for atom_id in ref_ids}
        exp_selected_sessions = {atoms[atom_id]['session_id'] for atom_id in exp_ids}
        ref_judge = judge_rows[((args.run_root / f'reference_user_{reference["person_id"]}.json').resolve(), question_id)]
        exp_judge = candidate_judge_rows[((candidate_root / f'{args.candidate_prefix}_user_{expanded["person_id"]}.json').resolve(), question_id)]
        for serving, evaluated in ((reference, ref_judge), (expanded, exp_judge)):
            if evaluated['prediction_sha256'] != serving['answer_sha256']:
                raise ValueError('evaluated answer identity differs')
        ref_correct = ref_judge['parse_valid'] and ref_judge['judge_label'] == 'CORRECT'
        exp_correct = exp_judge['parse_valid'] and exp_judge['judge_label'] == 'CORRECT'
        stages = {}
        for name, retrieved, selected in (
                ('reference', ref_retrieved, ref_selected_sessions),
                ('expanded', exp_retrieved, exp_selected_sessions)):
            coverage[f'{name}_retrieved_session_full'] += bool(gold_sessions <= retrieved)
            coverage[f'{name}_selected_session_full'] += bool(gold_sessions <= selected)
            if gold_images <= image_ids(atoms, selected_atom_ids(reference if name == 'reference' else expanded)):
                coverage[f'{name}_selected_image_full'] += bool(gold_images)
            coverage[f'{name}_gold_image_free'] += not gold_images
            served = reference if name == 'reference' else expanded
            retrieved_ids = {identity for identity, atom in atoms.items() if atom['session_id'] in retrieved}
            bounded_ids = {identity for edge in served['hypergraph_evidence_plan']['candidate_replay']
                           for identity in edge['atom_ids']}
            passed_images = {image['archive_member'] for image in served['images']}
            if passed_images != image_ids(atoms, selected_atom_ids(served)):
                raise ValueError('selected atoms and actual Reader images differ')
            stages[name] = image_omission_stage(gold_images, image_ids(atoms, retrieved_ids),
                image_ids(atoms, bounded_ids), passed_images, ref_correct if name == 'reference' else exp_correct)
            omission_stages[name][stages[name]] += 1
        rows.append({
            'question_id': question_id,
            'person_id': reference['person_id'],
            'reference_correct': ref_correct,
            'expanded_correct': exp_correct,
            'reference_judge_valid': ref_judge['parse_valid'],
            'expanded_judge_valid': exp_judge['parse_valid'],
            'image_omission_stage': stages,
            'reference_stop_reason': reference['stop_reason'],
            'expanded_stop_reason': expanded['stop_reason'],
            'reference_retrieved_sessions': sorted(ref_retrieved),
            'expanded_retrieved_sessions': sorted(exp_retrieved),
            'expanded_added_retrieved_sessions': sorted(exp_retrieved - ref_retrieved),
            'reference_selected_sessions': sorted(ref_selected_sessions),
            'expanded_selected_sessions': sorted(exp_selected_sessions),
            'reference_selected_atom_count': len(ref_ids),
            'expanded_selected_atom_count': len(exp_ids),
            'reference_selected_image_count': len(image_ids(atoms, ref_ids)),
            'expanded_selected_image_count': len(image_ids(atoms, exp_ids)),
            'gold_source_sessions': sorted(gold_sessions),
            'gold_images': sorted(gold_images),
            'reference_retrieved_session_recall': (len(gold_sessions & ref_retrieved) / len(gold_sessions)
                                                   if gold_sessions else None),
            'expanded_retrieved_session_recall': (len(gold_sessions & exp_retrieved) / len(gold_sessions)
                                                  if gold_sessions else None),
            'reference_selected_session_recall': (len(gold_sessions & ref_selected_sessions) / len(gold_sessions)
                                                  if gold_sessions else None),
            'expanded_selected_session_recall': (len(gold_sessions & exp_selected_sessions) / len(gold_sessions)
                                                 if gold_sessions else None),
            'reference_selected_image_recall': (len(gold_images & image_ids(atoms, ref_ids)) / len(gold_images)
                                                if gold_images else None),
            'expanded_selected_image_recall': (len(gold_images & image_ids(atoms, exp_ids)) / len(gold_images)
                                               if gold_images else None),
            'reference_service_seconds': reference['service_seconds'],
            'expanded_service_seconds': expanded['service_seconds'],
            'reference_input_tokens': reference['input_tokens'],
            'expanded_input_tokens': expanded['input_tokens'],
        })
    def mean(values):
        values = [value for value in values if value is not None]
        return sum(values) / len(values) if values else None
    result = {
        'schema_version': 'omni-recall-width-diagnostic-v1',
        'requests': len(rows),
        'candidate_role': {'expanded_keys_refer_to': args.candidate_prefix,
                           'root': str(candidate_root)},
        'quality': paired_quality(rows),
        'per_user_quality': {person_id: paired_quality([row for row in rows if row['person_id'] == person_id])
                             for person_id in expected},
        'coverage_counts': dict(coverage),
        'image_omission_stages': {policy: dict(counts) for policy, counts in omission_stages.items()},
        'mean_retrieved_session_recall': {
            policy: mean([row[f'{policy}_retrieved_session_recall'] for row in rows])
            for policy in ('reference', 'expanded')},
        'mean_selected_session_recall': {
            policy: mean([row[f'{policy}_selected_session_recall'] for row in rows])
            for policy in ('reference', 'expanded')},
        'mean_selected_image_recall_on_image_questions': {
            policy: mean([row[f'{policy}_selected_image_recall'] for row in rows])
            for policy in ('reference', 'expanded')},
        'evaluation_label_files_read_after_serving': [str(prepared / 'labels.jsonl')],
        'inputs_sha256': {str(path): digest(path) for path in inputs},
        'script_sha256': digest(Path(__file__)),
        'boundary': 'post-serving diagnosis; labels do not affect candidate retrieval, planning, or generation; source-session/image coverage is not factual sufficiency; retained label images with wrong answers do not alone prove Reader failure',
        'rows': rows,
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({key: result[key] for key in ('requests', 'quality',
                                                    'mean_retrieved_session_recall',
                                                    'mean_selected_image_recall_on_image_questions')},
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
