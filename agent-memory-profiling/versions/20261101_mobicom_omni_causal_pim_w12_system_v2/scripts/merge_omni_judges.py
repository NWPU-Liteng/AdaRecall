from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            result.update(block)
    return result.hexdigest()


def merge(run_paths, judge_paths, output):
    if len(run_paths) != len(judge_paths) or not run_paths or output.exists():
        raise ValueError('paired nonempty run/judge lists and a fresh output are required')
    judges = [json.loads(path.read_text()) for path in judge_paths]
    if any(not judge.get('complete') for judge in judges):
        raise ValueError('every per-run judge must be complete')
    provenance = judges[0]['provenance']
    protocol_fields = ('model_files_sha256', 'protocol', 'temperature', 'max_new_tokens',
                       'omni_prompt_source_sha256', 'prepared_audit_sha256')
    if any(provenance.get(field) is None for field in protocol_fields):
        raise ValueError('explicit evaluator model and protocol identity required')
    for judge in judges[1:]:
        other = judge['provenance']
        for field in protocol_fields:
            if provenance.get(field) != other.get(field):
                raise ValueError('per-user judge protocols or models differ')
    merged_rows, summaries, hashes = [], {}, {}
    expected_ids = set()
    for run_path, judge_path, judge in zip(run_paths, judge_paths, judges):
        run = json.loads(run_path.read_text())
        if (not run.get('complete') or run.get('evaluation_label_files_read') != []
                or not run.get('rows')):
            raise ValueError('complete label-isolated serving runs required')
        run_hash = digest(run_path)
        if judge['provenance']['run_sha256'] != {str(run_path): run_hash}:
            raise ValueError('judge must attest exactly this unchanged serving run')
        judge_rows = judge['rows']
        run_ids = [row['question_id'] for row in run['rows']]
        if len(run_ids) != len(set(run_ids)):
            raise ValueError('duplicate request within serving run')
        if [row['question_id'] for row in judge_rows] != run_ids:
            raise ValueError('all ordered requests including invalid judgments must be preserved')
        if any(Path(row['run']) != run_path for row in judge_rows):
            raise ValueError('judge rows reference another serving run')
        if expected_ids.intersection(row['question_id'] for row in judge_rows):
            raise ValueError('duplicate request across users')
        for served, evaluated in zip(run['rows'], judge_rows):
            prediction_hash = hashlib.sha256(served['answer'].encode()).hexdigest()
            if prediction_hash != served['answer_sha256'] or prediction_hash != evaluated['prediction_sha256']:
                raise ValueError('evaluated answer differs from serving run')
            if (evaluated['parse_valid'] is not (evaluated['judge_label'] in ('CORRECT', 'WRONG'))):
                raise ValueError('judge label and parse validity disagree')
        expected_ids.update(row['question_id'] for row in judge_rows)
        merged_rows.extend(judge_rows)
        correct = sum(row['parse_valid'] and row['judge_label'] == 'CORRECT' for row in judge_rows)
        valid = sum(row['parse_valid'] for row in judge_rows)
        summary = {'questions': len(judge_rows), 'correct': correct,
                   'invalid': len(judge_rows) - valid,
                   'full_denominator_accuracy': correct / len(judge_rows),
                   'valid_judge_accuracy': correct / valid if valid else None,
                   'official_judge_accuracy': None}
        if judge['summaries'][str(run_path)] != summary:
            raise ValueError('judge summary must retain every request in the denominator')
        summaries[str(run_path)] = summary
        hashes[str(run_path)] = run_hash
        hashes[str(judge_path)] = digest(judge_path)
    merged_provenance = dict(provenance, run_sha256=hashes_run_paths(hashes, run_paths),
                             component_judge_sha256={str(path): digest(path) for path in judge_paths},
                             merge_script_sha256=digest(Path(__file__)))
    merged = {'complete': True, 'rows': merged_rows, 'summaries': summaries,
              'provenance': merged_provenance,
              'boundary': 'strict per-user judge merge; all requests retained in source order; invalid judgments remain in the denominator'}
    with output.open('x') as handle:
        json.dump(merged, handle, ensure_ascii=False, indent=2)
    return merged


def hashes_run_paths(hashes, run_paths):
    return {str(path): hashes[str(path)] for path in run_paths}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    parser.add_argument('--judges', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    merged = merge(args.runs, args.judges, args.output)
    print(json.dumps({'complete': merged['complete'], 'requests': len(merged['rows']),
                      'summaries': merged['summaries']}))


if __name__ == '__main__':
    main()
