#!/usr/bin/env python3
"""Join a later precision arm to the frozen r2 development flips."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from audit_development_flip_transitions import classify, digest


def build_audit(r2: dict, candidate: dict, paired: dict, *,
                candidate_label: str, candidate_residual_bits: int | None,
                candidate_weight_bits: int | None) -> dict:
    if r2.get('complete') is not True or len(r2.get('flips', [])) != 13:
        raise ValueError('complete 13-flip r2 decomposition required')
    if candidate.get('complete') is not True or candidate.get('requests') != 13:
        raise ValueError('complete 13-request candidate evaluation required')
    if paired.get('complete') is not True or paired.get('requests') != 13:
        raise ValueError('complete 13-request candidate paired audit required')
    if paired.get('arms', {}).get('candidate') != candidate_label:
        raise ValueError('paired audit candidate label differs')
    if paired.get('candidate_fallback_scope') not in (None, 'none'):
        raise ValueError('candidate fallback scope must be none')
    if (candidate_residual_bits is not None
            and paired.get('candidate_residual_bits') != candidate_residual_bits):
        raise ValueError('candidate residual bits differ from paired audit')
    if (candidate_weight_bits is not None
            and candidate.get('serving_pair', {}).get('candidate_quantization') != candidate_label):
        raise ValueError('candidate evaluation label differs')

    r2_rows = {row['question_id']: row for row in r2['flips']}
    candidate_rows = {row['question_id']: row for row in candidate['rows']}
    paired_rows = {row['question_id']: row for row in paired['rows']}
    if not set(r2_rows) == set(candidate_rows) == set(paired_rows):
        raise ValueError('r2, candidate and paired-audit identities differ')
    if len(r2_rows) != 13 or len(candidate_rows) != 13 or len(paired_rows) != 13:
        raise ValueError('duplicate flip identity')
    users = sorted({str(row['person_id']) for row in r2_rows.values()}, key=int)
    if users != ['0', '1', '2']:
        raise ValueError('development-only users 0,1,2 required')

    rows = []
    for question_id in sorted(r2_rows):
        original = r2_rows[question_id]
        later = candidate_rows[question_id]
        pairing = paired_rows[question_id]
        transition = original.get('transition')
        if transition not in ('candidate_only', 'reference_only'):
            raise ValueError('r2 flip must be candidate-only or reference-only')
        if pairing.get('upstream_identical') is not True:
            raise ValueError('later arm does not prove identical pre-Reader execution')
        later_correct = bool(later.get('candidate_correct'))
        if later_correct and not (later.get('candidate_parse_valid')
                                  and later.get('candidate_eos')):
            raise ValueError('strict success must include Judge-valid output and EOS')
        rows.append({
            'question_id': question_id,
            'person_id': str(original['person_id']),
            'r2_transition': transition,
            'r2_bf16_correct': bool(original['reference_correct']),
            'r2_candidate_correct': bool(original['candidate_correct']),
            'candidate_bf16_correct': bool(later['reference_correct']),
            'candidate_correct': later_correct,
            'candidate_stop_reason_matches_bf16': bool(pairing['stop_reason_equal']),
            'outcome': classify(transition, later_correct),
        })

    transitions = Counter(row['r2_transition'] for row in rows)
    outcomes = Counter(row['outcome'] for row in rows)
    candidate_only = [row for row in rows if row['r2_transition'] == 'candidate_only']
    reference_only = [row for row in rows if row['r2_transition'] == 'reference_only']
    correct = sum(row['candidate_correct'] for row in rows)
    bf16_correct = sum(row['candidate_bf16_correct'] for row in rows)
    if correct != candidate['quality']['candidate_correct']:
        raise ValueError('joined candidate count differs from evaluation summary')
    if bf16_correct != candidate['quality']['reference_correct']:
        raise ValueError('joined BF16 count differs from evaluation summary')

    return {
        'schema_version': 'precision-transition-audit-v1',
        'complete': True,
        'candidate_arm': candidate_label,
        'candidate_weight_bits': candidate_weight_bits,
        'candidate_residual_bits': candidate_residual_bits,
        'scope': 'development-only 13 frozen flip cohort; not a population quality estimate',
        'inputs': {
            'r2_decomposition_sha256': digest(Path(r2['_path'])),
            'candidate_evaluation_sha256': digest(Path(candidate['_path'])),
            'candidate_paired_audit_sha256': digest(Path(paired['_path'])),
        },
        'summary': {
            'requests': len(rows),
            'r2_partition': dict(transitions),
            'r2_candidate_correct': sum(row['r2_candidate_correct'] for row in rows),
            'r2_bf16_correct': sum(row['r2_bf16_correct'] for row in rows),
            'candidate_correct_strict_proxy': correct,
            'candidate_bf16_correct_strict_proxy': bf16_correct,
            'candidate_strict_proxy_delta_vs_r2_pp': 100 * (correct - len(candidate_only))
            / len(rows),
            'candidate_only': {
                'requests': len(candidate_only),
                'preserved': sum(row['candidate_correct'] for row in candidate_only),
                'lost': sum(not row['candidate_correct'] for row in candidate_only),
                'preserved_questions': [row['question_id'] for row in candidate_only
                                        if row['candidate_correct']],
                'lost_questions': [row['question_id'] for row in candidate_only
                                   if not row['candidate_correct']],
            },
            'reference_only': {
                'requests': len(reference_only),
                'recovered': sum(row['candidate_correct'] for row in reference_only),
                'not_recovered': sum(not row['candidate_correct'] for row in reference_only),
                'recovered_questions': [row['question_id'] for row in reference_only
                                        if row['candidate_correct']],
                'not_recovered_questions': [row['question_id'] for row in reference_only
                                            if not row['candidate_correct']],
            },
            'outcomes': dict(outcomes),
            'net_r2_success_recovery': (outcomes['r2_failure_recovered']
                                        - outcomes['r2_success_lost']),
        },
        'rows': rows,
        'boundary': (
            'development-only diagnostic; same-family local 8B Judge; strict proxy '
            'requires a valid Judge Yes and EOS; 13 selected flips cannot estimate '
            'population accuracy'),
    }


def load(path: Path) -> dict:
    value = json.loads(path.read_text())
    value['_path'] = str(path)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--r2-decomposition', type=Path, required=True)
    parser.add_argument('--candidate-evaluation', type=Path, required=True)
    parser.add_argument('--candidate-paired-audit', type=Path, required=True)
    parser.add_argument('--candidate-label', required=True)
    parser.add_argument('--candidate-residual-bits', type=int, choices=(0, 2, 4), default=0)
    parser.add_argument('--candidate-weight-bits', type=int, choices=(8, 12), default=None)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = build_audit(
        load(args.r2_decomposition), load(args.candidate_evaluation),
        load(args.candidate_paired_audit), candidate_label=args.candidate_label,
        candidate_residual_bits=args.candidate_residual_bits,
        candidate_weight_bits=args.candidate_weight_bits)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print(json.dumps(result['summary'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
