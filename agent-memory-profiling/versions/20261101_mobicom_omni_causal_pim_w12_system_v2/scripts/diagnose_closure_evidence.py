"""Post-serving category/source attribution; never imported by the planner."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def diagnose(prepared, run_paths, judge_path):
    audit = json.loads((prepared / 'audit.json').read_text())
    if digest(prepared / 'labels.jsonl') != audit['artifacts_sha256']['labels.jsonl']:
        raise ValueError('evaluation labels changed')
    labels = {row['question_id']: row for row in
              map(json.loads, (prepared / 'labels.jsonl').read_text().splitlines())}
    judge = json.loads(judge_path.read_text())
    if not judge.get('complete'):
        raise ValueError('complete evaluation required')
    verdicts = {(row['run'], row['question_id']): row for row in judge['rows']}
    outputs = []
    for path in run_paths:
        run = json.loads(path.read_text())
        if not run.get('complete') or digest(path) != judge['provenance']['run_sha256'][str(path)]:
            raise ValueError('serving run changed or incomplete')
        categories = {}
        rows = []
        for row in run['rows']:
            label = labels[row['question_id']]
            reference_sessions = set(label['source_session_ids'])
            candidate_sessions = set(row['retrieved_session_ids'])
            selected_sessions = {selection['session_id'] for selection in row['evidence_selection']
                                 if selection['accepted_atom_ids']}
            reference_images = {image.get('resolved', image['normalized'])
                                for image in label['evidence_image_refs']
                                if image['status'] in ('exact', 'unique_user_basename')}
            selected_images = {image['archive_member'] for image in row['images']}
            verdict = verdicts[(str(path), row['question_id'])]
            category = label['question_type']
            counts = categories.setdefault(category, Counter())
            counts['requests'] += 1
            counts['correct'] += verdict['judge_label'] == 'CORRECT'
            counts['judge_invalid'] += verdict['judge_label'] is None
            counts['eos'] += row['stop_reason'] == 'eos'
            counts['requests_with_images'] += bool(row['images'])
            counts['resolved_source_sessions'] += len(reference_sessions)
            counts['candidate_source_sessions'] += len(reference_sessions & candidate_sessions)
            counts['selected_source_sessions'] += len(reference_sessions & selected_sessions)
            counts['resolved_source_images'] += len(reference_images)
            counts['selected_source_images'] += len(reference_images & selected_images)
            rows.append({'question_id': row['question_id'], 'category': category,
                         'judge_label': verdict['judge_label'],
                         'resolved_source_sessions': len(reference_sessions),
                         'candidate_source_sessions': len(reference_sessions & candidate_sessions),
                         'selected_source_sessions': len(reference_sessions & selected_sessions),
                         'resolved_source_images': len(reference_images),
                         'selected_source_images': len(reference_images & selected_images),
                         'stop_reason': row['stop_reason']})
        outputs.append({'run': str(path), 'policy': run['evidence_policy_name'],
                        'categories': {name: dict(value) for name, value in categories.items()}, 'rows': rows})
    return {'complete': True, 'runs': outputs,
            'evaluation_label_files_read': ['labels.jsonl'],
            'serving_algorithm_modified': False, 'judge_sha256': digest(judge_path),
            'boundary': 'post-serving attribution only; source/session membership is an imperfect recall proxy, not proof of answer sufficiency or grounding; unresolved references stay out of resolved-recall denominator and in QA denominator'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    parser.add_argument('--judge', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('fresh diagnostic output required')
    result = diagnose(args.prepared, args.runs, args.judge)
    with args.output.open('x') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
