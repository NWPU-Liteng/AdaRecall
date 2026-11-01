#!/usr/bin/env python3
"""Audit how a diagnostic precision arm moves frozen development flips.

The r2 decomposition defines the original transition.  A later arm is not
allowed to claim "lower hidden drift" as semantic progress: this script joins
the strict Judge+EOS outcome of that arm back to the original transition and
reports which r2 successes survived and which r2 failures were recovered.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            result.update(block)
    return result.hexdigest()


def classify(transition: str, r4_correct: bool) -> str:
    if transition == 'candidate_only':
        return 'r2_success_preserved' if r4_correct else 'r2_success_lost'
    if transition == 'reference_only':
        return 'r2_failure_recovered' if r4_correct else 'r2_failure_not_recovered'
    raise ValueError('only the 13 r2 flip transitions are admissible')


def build_audit(r2: dict, r4: dict, paired: dict) -> dict:
    if r2.get('complete') is not True or len(r2.get('flips', [])) != 13:
        raise ValueError('complete 13-flip r2 decomposition required')
    if r4.get('complete') is not True or r4.get('requests') != 13:
        raise ValueError('complete 13-request r4 evaluation required')
    if paired.get('complete') is not True or paired.get('requests') != 13:
        raise ValueError('complete 13-request r4 paired audit required')
    if paired.get('arms', {}).get('candidate') != 'w8a16_r4':
        raise ValueError('paired audit must describe w8a16_r4')
    if paired.get('candidate_residual_bits') != 4:
        raise ValueError('paired audit must bind residual bits to four')
    if paired.get('candidate_fallback_scope') != 'none':
        raise ValueError('paired audit must bind fallback scope to none')

    r2_rows = {row['question_id']: row for row in r2['flips']}
    r4_rows = {row['question_id']: row for row in r4['rows']}
    paired_rows = {row['question_id']: row for row in paired['rows']}
    if not set(r2_rows) == set(r4_rows) == set(paired_rows):
        raise ValueError('r2, r4 and paired-audit identities differ')
    if len(r2_rows) != 13 or len(r4_rows) != 13 or len(paired_rows) != 13:
        raise ValueError('duplicate flip identity')
    users = sorted({str(row['person_id']) for row in r2_rows.values()}, key=int)
    if users != ['0', '1', '2']:
        raise ValueError('development-only users 0,1,2 required')

    rows = []
    for question_id in sorted(r2_rows):
        original = r2_rows[question_id]
        candidate = r4_rows[question_id]
        pairing = paired_rows[question_id]
        transition = original.get('transition')
        if transition not in ('candidate_only', 'reference_only'):
            raise ValueError('r2 flip must be candidate-only or reference-only')
        if bool(original.get('candidate_correct')) != (transition == 'candidate_only'):
            raise ValueError('r2 transition and candidate correctness disagree')
        if bool(original.get('reference_correct')) != (transition == 'reference_only'):
            raise ValueError('r2 transition and reference correctness disagree')
        if pairing.get('upstream_identical') is not True:
            raise ValueError('r4 paired audit does not prove identical pre-Reader execution')
        r4_correct = bool(candidate.get('candidate_correct'))
        if r4_correct and not (candidate.get('candidate_parse_valid')
                               and candidate.get('candidate_eos')):
            raise ValueError('r4 strict success must include Judge-valid output and EOS')
        rows.append({
            'question_id': question_id,
            'person_id': str(original['person_id']),
            'r2_transition': transition,
            'r2_bf16_correct': bool(original['reference_correct']),
            'r2_candidate_correct': bool(original['candidate_correct']),
            'r4_bf16_correct': bool(candidate['reference_correct']),
            'r4_candidate_correct': r4_correct,
            'r4_stop_reason_matches_bf16': bool(pairing['stop_reason_equal']),
            'outcome': classify(transition, r4_correct),
        })

    transitions = Counter(row['r2_transition'] for row in rows)
    outcomes = Counter(row['outcome'] for row in rows)
    candidate_only = [row for row in rows if row['r2_transition'] == 'candidate_only']
    reference_only = [row for row in rows if row['r2_transition'] == 'reference_only']
    r4_bf16_correct = sum(row['r4_bf16_correct'] for row in rows)
    r4_correct = sum(row['r4_candidate_correct'] for row in rows)
    if r4_bf16_correct != r4['quality']['reference_correct']:
        raise ValueError('joined r4 BF16 count differs from evaluation summary')
    if r4_correct != r4['quality']['candidate_correct']:
        raise ValueError('joined r4 candidate count differs from evaluation summary')

    return {
        'schema_version': 'development-flip-transition-audit-v1',
        'complete': True,
        'scope': 'development-only 13 frozen flip cohort; not a population quality estimate',
        'inputs': {
            'r2_decomposition_sha256': digest(Path(r2['_path'])),
            'r4_evaluation_sha256': digest(Path(r4['_path'])),
            'r4_paired_audit_sha256': digest(Path(paired['_path'])),
        },
        'arms': {
            'r2': 'w8a16_r2',
            'r4': 'w8a16_r4',
            'reference': 'bf16',
        },
        'summary': {
            'requests': len(rows),
            'r2_partition': dict(transitions),
            'r2_candidate_correct': sum(row['r2_candidate_correct'] for row in rows),
            'r2_bf16_correct': sum(row['r2_bf16_correct'] for row in rows),
            'r4_candidate_correct_strict_proxy': r4_correct,
            'r4_bf16_correct_strict_proxy': r4_bf16_correct,
            'r4_strict_proxy_delta_vs_r2_pp': 100 * (r4_correct - (len(candidate_only)))
            / len(rows),
            'candidate_only': {
                'requests': len(candidate_only),
                'preserved': sum(row['r4_candidate_correct'] for row in candidate_only),
                'lost': sum(not row['r4_candidate_correct'] for row in candidate_only),
                'preserved_questions': [row['question_id'] for row in candidate_only
                                        if row['r4_candidate_correct']],
                'lost_questions': [row['question_id'] for row in candidate_only
                                   if not row['r4_candidate_correct']],
            },
            'reference_only': {
                'requests': len(reference_only),
                'recovered': sum(row['r4_candidate_correct'] for row in reference_only),
                'not_recovered': sum(not row['r4_candidate_correct']
                                     for row in reference_only),
                'recovered_questions': [row['question_id'] for row in reference_only
                                        if row['r4_candidate_correct']],
                'not_recovered_questions': [row['question_id'] for row in reference_only
                                            if not row['r4_candidate_correct']],
            },
            'outcomes': dict(outcomes),
            'net_r2_success_recovery': (outcomes['r2_failure_recovered']
                                        - outcomes['r2_success_lost']),
        },
        'rows': rows,
        'mechanism_finding': (
            'The r4 arm lowers teacher-forced hidden drift on every one of the 13 '
            'flips, yet it loses r2 candidate-only successes and recovers only a '
            'subset of r2 reference-only failures.  Hidden drift or bit width is '
            'therefore not a monotone proxy for semantic correctness.'),
        'boundary': (
            'development-only diagnostic; same-family local 8B Judge; strict proxy '
            'requires a valid Judge Yes and EOS; 13 selected flips cannot estimate '
            'population accuracy or establish a deployable precision arm'),
    }


def load(path: Path) -> dict:
    value = json.loads(path.read_text())
    value['_path'] = str(path)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--r2-decomposition', type=Path, required=True)
    parser.add_argument('--r4-evaluation', type=Path, required=True)
    parser.add_argument('--r4-paired-audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = build_audit(load(args.r2_decomposition), load(args.r4_evaluation),
                         load(args.r4_paired_audit))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print(json.dumps(result['summary'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
