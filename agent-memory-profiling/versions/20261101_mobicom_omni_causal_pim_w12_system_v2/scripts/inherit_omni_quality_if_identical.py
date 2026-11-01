"""Transfer an evaluation result only after complete serving outputs match exactly."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from judge_omni_closure import parse_label


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inherit_quality(reference_root, candidate_root, cohort, judge):
    if not judge.get('complete') or not isinstance(judge.get('summaries'), dict):
        raise ValueError('complete reference judge required')
    expected_users = cohort['question_ids_by_user']
    if not expected_users or not set(expected_users) <= {'0', '1', '2'}:
        raise ValueError('authorized development users required')
    provenance = judge['provenance']
    if any(not provenance.get(field) for field in (
            'model_files_sha256', 'omni_prompt_source_sha256', 'prepared_audit_sha256', 'protocol')):
        raise ValueError('pinned reference evaluator identity required')
    per_user, rows = {}, []
    inputs = {}
    for person_id, question_ids in expected_users.items():
        if not question_ids or len(question_ids) != len(set(question_ids)):
            raise ValueError('nonempty unique registered requests required')
        reference_path = reference_root / f'reference_user_{person_id}.json'
        candidate_path = candidate_root / f'reference_user_{person_id}.json'
        reference = json.loads(reference_path.read_text())
        candidate = json.loads(candidate_path.read_text())
        inputs.update({str(path): digest(path) for path in (reference_path, candidate_path)})
        if provenance['run_sha256'].get(str(reference_path)) != inputs[str(reference_path)]:
            raise ValueError('judged reference serving artifact changed')
        if any(run.get('prepared_audit_sha256') != provenance['prepared_audit_sha256']
               or run.get('evaluation_label_files_read') != [] for run in (reference, candidate)):
            raise ValueError('same prepared data and label-isolated serving required')
        if not reference.get('complete') or not candidate.get('complete'):
            raise ValueError('complete reference and candidate streams required')
        if [row['question_id'] for row in reference['rows']] != question_ids:
            raise ValueError('reference stream differs from registered cohort')
        if [row['question_id'] for row in candidate['rows']] != question_ids:
            raise ValueError('candidate stream differs from registered cohort')
        judgments = [row for row in judge['rows'] if row['run'] == str(reference_path)]
        if [row['question_id'] for row in judgments] != question_ids:
            raise ValueError('complete ordered original judgments required')
        mismatches = []
        for before, after, judgment in zip(reference['rows'], candidate['rows'], judgments):
            fields = ('question_id', 'question_sha256', 'images', 'answer_token_ids',
                      'stop_reason', 'input_tokens', 'visual_patches', 'answer_sha256',
                      'processor_inputs_sha256')
            changed = [field for field in fields if before[field] != after[field]]
            if changed:
                mismatches.append({'question_id': after['question_id'], 'fields': changed})
            for prediction in (before, after):
                if (str(prediction['person_id']) != person_id or not prediction['answer_token_ids']
                        or not prediction['processor_inputs_sha256'].get('input_ids')
                        or digest_answer(prediction['answer']) != prediction['answer_sha256']):
                    raise ValueError('serving answer payload or user identity changed')
            label = parse_label(judgment['response'])
            if (judgment['prediction_sha256'] != before['answer_sha256']
                    or judgment['stop_reason'] != before['stop_reason']
                    or judgment['judge_label'] != label or judgment['parse_valid'] != (label is not None)):
                raise ValueError('reference judgment payload changed')
            rows.append({'question_id': after['question_id'], 'person_id': person_id,
                         'answer_sha256': after['answer_sha256'],
                         'stop_reason': after['stop_reason'], 'judge_label': label,
                         'strict_eos_correct': label == 'CORRECT' and after['stop_reason'] == 'eos'})
        if mismatches:
            raise ValueError(json.dumps({'person_id': person_id, 'mismatches': mismatches}))
        judge_key = str(reference_path)
        if judge_key not in judge['summaries']:
            raise ValueError('reference judge lacks complete per-user summary')
        summary = judge['summaries'][judge_key]
        correct = sum(row['judge_label'] == 'CORRECT' for row in judgments)
        invalid = sum(not row['parse_valid'] for row in judgments)
        if (summary['questions'] != len(question_ids) or summary['correct'] != correct
                or summary['invalid'] != invalid
                or summary['full_denominator_accuracy'] != correct / len(question_ids)):
            raise ValueError('reference judge summary differs from complete raw judgments')
        per_user[person_id] = summary
    if len(judge['rows']) != len(rows):
        raise ValueError('exact registered judge denominator required')
    return {
        'complete': True,
        'quality_inherited': True,
        'reason': 'all registered serving inputs, processor digests, answer token sequences and stop states match exactly',
        'requests': len(rows),
        'rows': rows,
        'per_user': per_user,
        'summaries': judge['summaries'],
        'correct': sum(row['judge_label'] == 'CORRECT' for row in rows),
        'strict_eos_correct': sum(row['strict_eos_correct'] for row in rows),
        'truncated_question_ids': [row['question_id'] for row in rows if row['stop_reason'] != 'eos'],
        'evaluator_provenance': provenance,
        'inputs_sha256': inputs,
        'candidate_root': str(candidate_root),
        'labels_used_by_serving': False,
        'boundary': 'evaluation inheritance only; no new judge inference and no claim beyond the pinned reference evaluator',
    }


def digest_answer(answer):
    return hashlib.sha256(answer.encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference-root', type=Path, required=True)
    parser.add_argument('--candidate-root', type=Path, required=True)
    parser.add_argument('--cohort', type=Path, required=True)
    parser.add_argument('--reference-judge', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('fresh quality inheritance artifact required')
    result = inherit_quality(args.reference_root, args.candidate_root,
        json.loads(args.cohort.read_text()), json.loads(args.reference_judge.read_text()))
    result['inputs_sha256'].update({str(path): digest(path) for path in (
        args.cohort, args.reference_judge, Path(__file__))})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ('complete', 'quality_inherited', 'requests',
                                                 'correct', 'strict_eos_correct')}))


if __name__ == '__main__':
    main()
