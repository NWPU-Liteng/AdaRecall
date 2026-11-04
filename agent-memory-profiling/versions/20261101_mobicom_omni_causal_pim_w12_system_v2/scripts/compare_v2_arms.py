#!/usr/bin/env python3
"""Compare two completed validation arms using their Judge ledgers."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


VALIDATION_USERS = ('4', '5', '6', '7')
EXPECTED_QUESTIONS = 56


def load_rows(run_root: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for user in VALIDATION_USERS:
        for path in sorted((run_root / 'judge_rows' / user).glob('*.json')):
            row = json.loads(path.read_text())
            identity = row.get('question_id')
            if not isinstance(identity, str) or identity in rows:
                raise ValueError(f'invalid or duplicate Judge identity: {path}')
            rows[identity] = row
    if len(rows) != EXPECTED_QUESTIONS:
        raise ValueError(
            f'incomplete Judge ledger: {len(rows)} != {EXPECTED_QUESTIONS}')
    return rows


def is_correct(row: dict) -> bool:
    return row.get('parse_valid') is True and row.get('judge_label') == 'CORRECT'


def exact_mcnemar_p(baseline_only: int, candidate_only: int) -> float:
    discordant = baseline_only + candidate_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, index)
               for index in range(min(baseline_only, candidate_only) + 1))
    return min(1.0, 2 * tail / (2 ** discordant))


def compare(baseline: dict[str, dict], candidate: dict[str, dict]) -> dict:
    if set(baseline) != set(candidate):
        raise ValueError('arms do not contain the same validation questions')
    by_type: dict[str, dict] = {}
    both = baseline_only = candidate_only = neither = 0
    for identity in sorted(baseline):
        left, right = baseline[identity], candidate[identity]
        question_type = left.get('question_type')
        if question_type != right.get('question_type'):
            raise ValueError(f'question type differs for {identity}')
        bucket = by_type.setdefault(str(question_type), {
            'n': 0, 'baseline_correct': 0, 'candidate_correct': 0,
            'baseline_only': 0, 'candidate_only': 0,
        })
        bucket['n'] += 1
        left_correct, right_correct = is_correct(left), is_correct(right)
        bucket['baseline_correct'] += int(left_correct)
        bucket['candidate_correct'] += int(right_correct)
        if left_correct and right_correct:
            both += 1
        elif left_correct:
            baseline_only += 1
            bucket['baseline_only'] += 1
        elif right_correct:
            candidate_only += 1
            bucket['candidate_only'] += 1
        else:
            neither += 1
    for bucket in by_type.values():
        bucket['delta_correct'] = (bucket['candidate_correct']
                                   - bucket['baseline_correct'])
    baseline_correct = both + baseline_only
    candidate_correct = both + candidate_only
    return {
        'n': len(baseline),
        'baseline_correct': baseline_correct,
        'candidate_correct': candidate_correct,
        'delta_correct': candidate_correct - baseline_correct,
        'baseline_accuracy': baseline_correct / len(baseline),
        'candidate_accuracy': candidate_correct / len(candidate),
        'paired': {
            'both_correct': both,
            'baseline_only_correct': baseline_only,
            'candidate_only_correct': candidate_only,
            'both_wrong': neither,
            'mcnemar_exact_p': exact_mcnemar_p(baseline_only, candidate_only),
        },
        'by_type': dict(sorted(by_type.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = {
        'schema': 'mobilemem-v2-paired-judge-comparison-v1',
        'baseline': str(args.baseline),
        'candidate': str(args.candidate),
        **compare(load_rows(args.baseline), load_rows(args.candidate)),
    }
    text = json.dumps(report, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end='')


if __name__ == '__main__':
    main()
