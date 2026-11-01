"""Separate evaluator drift from workload and serving-protocol differences."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def compare_judgments(old_rows, new_rows):
    old = {row['question_id']: row for row in old_rows}
    new = {row['question_id']: row for row in new_rows}
    if len(old) != len(old_rows) or len(new) != len(new_rows) or set(old) != set(new):
        raise ValueError('identical unique complete question sets required')
    rows = []
    for question_id, current in new.items():
        previous = old[question_id]
        if previous['prediction_sha256'] != current['prediction_sha256']:
            raise ValueError('saved prediction changed; not evaluator-only alignment')
        if not previous['parse_valid'] or not current['parse_valid']:
            raise ValueError('invalid evaluator outputs require separate treatment')
        rows.append({'question_id': question_id, 'prediction_sha256': current['prediction_sha256'],
                     'previous_correct': previous['value'] == 1.0,
                     'current_correct': current['judge_label'] == 'CORRECT'})
    return {
        'requests': len(rows),
        'previous_correct': sum(row['previous_correct'] for row in rows),
        'current_correct': sum(row['current_correct'] for row in rows),
        'previous_only_correct': sum(row['previous_correct'] and not row['current_correct'] for row in rows),
        'current_only_correct': sum(row['current_correct'] and not row['previous_correct'] for row in rows),
        'rows': rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--previous-judge', type=Path, required=True)
    parser.add_argument('--aligned-judge', type=Path, required=True)
    parser.add_argument('--previous-prepared', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--historical-runs', type=Path, nargs='+', required=True)
    parser.add_argument('--current-runs', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('fresh protocol-alignment summary required')
    old = json.loads(args.previous_judge.read_text())
    aligned = json.loads(args.aligned_judge.read_text())
    if not old['complete'] or not aligned['complete']:
        raise ValueError('complete judgments required')
    comparison = compare_judgments(old['rows'], aligned['rows'])
    identities = {row['question_id'] for row in comparison['rows']}
    labels = {row['question_id']: row for row in map(json.loads, (args.prepared / 'labels.jsonl').read_text().splitlines())
              if row['question_id'] in identities}
    previous_labels = {row['question_id']: row for row in map(json.loads,
                       (args.previous_prepared / 'labels.jsonl').read_text().splitlines())
                       if row['question_id'] in identities}
    if set(labels) != identities or set(previous_labels) != identities:
        raise ValueError('missing historical labels')
    old_models = old['provenance']['model_hashes']
    current_models = aligned['provenance']['model_files_sha256']
    shared_names = sorted(set(old_models) & set(current_models))
    model_differences = [name for name in shared_names if old_models[name] != current_models[name]]
    runs = {}
    for group, paths in [('historical', args.historical_runs), ('current', args.current_runs)]:
        data = [json.loads(path.read_text()) for path in paths]
        if any(not run['complete'] for run in data):
            raise ValueError('complete serving results required')
        question_ids = [row['question_id'] for run in data for row in run['rows']]
        if len(question_ids) != len(set(question_ids)):
            raise ValueError('one policy per group required')
        runs[group] = {'question_ids': question_ids,
            'requests': len(question_ids),
            'reader_identities': sorted({(run.get('reader_identity') or run.get('reader_config_sha256')
                                         or run.get('reader')) for run in data}),
            'run_sha256': {str(path): digest(path) for path in paths},
            'protocols': [{key: run.get(key) for key in (
                'max_images', 'max_new_tokens', 'total_text_tokens', 'image_max_pixels',
                'image_novelty_weight', 'embedding_state_reuse_enabled', 'prepared_audit_sha256',
                'evidence_policy_name', 'runtime_environment', 'system_prompt_sha256')}
                for run in data]}
    overlap = sorted(set(runs['historical']['question_ids']) & set(runs['current']['question_ids']))
    result = {
        'complete': True, 'evaluator_only_comparison': comparison,
        'historical_categories': dict(Counter(row['question_type'] for row in labels.values())),
        'labels_changed': sorted(identity for identity in identities if labels[identity] != previous_labels[identity]),
        'judge_shared_model_files': shared_names, 'judge_model_file_differences': model_differences,
        'cohort_overlap': overlap, 'cohort_overlap_count': len(overlap), 'runs': runs,
        'inputs_sha256': {str(path): digest(path) for path in (args.previous_judge, args.aligned_judge)},
        'boundary': 'same saved answers isolate evaluator-protocol drift; unlike cohorts and image budgets do not identify algorithm regression; proxy judge only',
        'main_experiment_closed': False,
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key not in ('runs', 'evaluator_only_comparison')}, ensure_ascii=False))
    print(json.dumps({key: value for key, value in comparison.items() if key != 'rows'}))


if __name__ == '__main__':
    main()
