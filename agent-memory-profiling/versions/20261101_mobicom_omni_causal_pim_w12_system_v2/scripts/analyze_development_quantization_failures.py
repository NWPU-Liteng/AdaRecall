#!/usr/bin/env python3
"""Decompose development-only W8A16+r2 flips without touching holdout users."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            result.update(block)
    return result.hexdigest()


def transition(row: dict) -> str:
    if row['reference_correct'] and row['candidate_correct']:
        return 'both_correct'
    if not row['reference_correct'] and not row['candidate_correct']:
        return 'both_wrong'
    return 'candidate_only' if row['candidate_correct'] else 'reference_only'


def median(values):
    return statistics.median(values) if values else None


def quantile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def prefix_bucket(prefix: int) -> str:
    if prefix <= 8:
        return '00-08'
    if prefix <= 16:
        return '09-16'
    if prefix <= 32:
        return '17-32'
    if prefix <= 64:
        return '33-64'
    if prefix <= 128:
        return '65-128'
    return '129+'


def length_bucket(tokens: int) -> str:
    if tokens < 128:
        return '000-127'
    if tokens < 256:
        return '128-255'
    if tokens < 384:
        return '256-383'
    if tokens < 512:
        return '384-511'
    return '512+'


def aggregate(rows: list[dict]) -> dict:
    prefixes = [row['common_prefix_tokens'] for row in rows]
    reference_tokens = [row['reference_tokens'] for row in rows]
    candidate_tokens = [row['candidate_tokens'] for row in rows]
    return {
        'requests': len(rows),
        'common_prefix_tokens': {
            'median': median(prefixes),
            'p25': quantile(prefixes, 0.25),
            'p75': quantile(prefixes, 0.75),
            'le_8': sum(value <= 8 for value in prefixes),
            'le_16': sum(value <= 16 for value in prefixes),
            'le_32': sum(value <= 32 for value in prefixes),
            'le_64': sum(value <= 64 for value in prefixes),
        },
        'reference_tokens': {'median': median(reference_tokens),
                             'min': min(reference_tokens, default=None),
                             'max': max(reference_tokens, default=None)},
        'candidate_tokens': {'median': median(candidate_tokens),
                             'min': min(candidate_tokens, default=None),
                             'max': max(candidate_tokens, default=None)},
        'length_ratio_candidate_over_reference': {
            'median': median([candidate / reference for candidate, reference
                              in zip(candidate_tokens, reference_tokens)]),
            'mean': (sum(candidate / reference for candidate, reference
                         in zip(candidate_tokens, reference_tokens)) / len(rows)
                     if rows else None),
        },
        'parse_valid': {
            'reference': sum(row['reference_parse_valid'] for row in rows),
            'candidate': sum(row['candidate_parse_valid'] for row in rows),
        },
        'eos': {
            'reference': sum(row['reference_eos'] for row in rows),
            'candidate': sum(row['candidate_eos'] for row in rows),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--weight-sensitivity', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    evaluation = json.loads(args.evaluation.read_text())
    audit = json.loads(args.audit.read_text())
    if evaluation.get('complete') is not True or evaluation.get('requests') != 192:
        raise ValueError('complete 192-request development evaluation required')
    if audit.get('complete') is not True or audit.get('requests') != 192:
        raise ValueError('complete 192-request development audit required')
    users = sorted({str(row['person_id']) for row in evaluation['rows']}, key=int)
    if users != ['0', '1', '2']:
        raise ValueError('development-only analysis requires users 0,1,2')
    evaluation_by_id = {row['question_id']: row for row in evaluation['rows']}
    audit_by_id = {row['question_id']: row for row in audit['rows']}
    if set(evaluation_by_id) != set(audit_by_id):
        raise ValueError('evaluation/audit identity mismatch')

    rows = []
    for question_id, quality in evaluation_by_id.items():
        alignment = audit_by_id[question_id]['token_alignment']
        row = {
            'question_id': question_id,
            'person_id': str(quality['person_id']),
            'transition': transition(quality),
            'reference_correct': quality['reference_correct'],
            'candidate_correct': quality['candidate_correct'],
            'common_prefix_tokens': alignment['common_prefix_tokens'],
            'reference_tokens': alignment['reference_tokens'],
            'candidate_tokens': alignment['candidate_tokens'],
            'reference_parse_valid': quality['reference_parse_valid'],
            'candidate_parse_valid': quality['candidate_parse_valid'],
            'reference_eos': quality['reference_eos'],
            'candidate_eos': quality['candidate_eos'],
            'exact_greedy_tokens_equal': quality['exact_greedy_tokens_equal'],
            'upstream_identical': audit_by_id[question_id]['upstream_identical'],
            'prefix_bucket': prefix_bucket(alignment['common_prefix_tokens']),
            'reference_length_bucket': length_bucket(alignment['reference_tokens']),
            'candidate_length_bucket': length_bucket(alignment['candidate_tokens']),
        }
        rows.append(row)
    if not all(row['upstream_identical'] for row in rows):
        raise ValueError('nonidentical upstream execution in development audit')

    by_transition = {name: aggregate([row for row in rows if row['transition'] == name])
                     for name in ('both_correct', 'candidate_only', 'reference_only', 'both_wrong')}
    by_user = {}
    for user in users:
        selected = [row for row in rows if row['person_id'] == user]
        by_user[user] = {
            'transitions': dict(Counter(row['transition'] for row in selected)),
            'aggregate': aggregate(selected),
        }
    by_prefix = {}
    for bucket in ('00-08', '09-16', '17-32', '33-64', '65-128', '129+'):
        selected = [row for row in rows if row['prefix_bucket'] == bucket]
        by_prefix[bucket] = {
            'requests': len(selected),
            'reference_correct': sum(row['reference_correct'] for row in selected),
            'candidate_correct': sum(row['candidate_correct'] for row in selected),
            'delta_correct': (sum(row['candidate_correct'] for row in selected)
                              - sum(row['reference_correct'] for row in selected)),
            'transitions': dict(Counter(row['transition'] for row in selected)),
        }
    flips = [row for row in rows if row['transition'] in ('candidate_only', 'reference_only')]
    flips.sort(key=lambda row: (int(row['person_id']), row['question_id']))

    weight_summary = None
    if args.weight_sensitivity:
        weights = json.loads(args.weight_sensitivity.read_text())
        records = weights.get('records', [])
        if len(records) != 249:
            raise ValueError('expected all 249 text Linear matrices')
        def category(name: str) -> str:
            for part in ('embed_tokens', 'lm_head', 'linear_attn', 'self_attn', 'mlp'):
                if part in name:
                    return part
            return 'other'
        grouped = defaultdict(list)
        for record in records:
            grouped[category(record['name'])].append(record)
        weight_summary = {
            'records': len(records),
            'rel_fro': {
                'mean': sum(record['rel_fro'] for record in records) / len(records),
                'max': max(record['rel_fro'] for record in records),
            },
            'by_category': {
                key: {
                    'matrices': len(group),
                    'rel_fro_mean': sum(record['rel_fro'] for record in group) / len(group),
                    'rel_fro_max': max(record['rel_fro'] for record in group),
                    'top5pct_row_error_share_mean': (
                        sum(record['top5pct_row_error_share'] for record in group)
                        / len(group)),
                    'top5pct_row_error_share_max': max(
                        record['top5pct_row_error_share'] for record in group),
                }
                for key, group in sorted(grouped.items())
            },
            'highest_rel_fro': sorted(
                ({key: record[key] for key in (
                    'name', 'rel_fro', 'top5pct_row_error_share', 'top5pct_extra_bytes')}
                 for record in records),
                key=lambda record: record['rel_fro'], reverse=True)[:12],
        }

    result = {
        'schema': 'hyperpim-sc-development-failure-decomposition-v1',
        'complete': True,
        'as_of': datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds'),
        'scope': 'development users 0,1,2 only; never use for holdout tuning',
        'inputs': {
            str(args.evaluation): digest(args.evaluation),
            str(args.audit): digest(args.audit),
            **({str(args.weight_sensitivity): digest(args.weight_sensitivity)}
               if args.weight_sensitivity else {}),
        },
        'cohort': {
            'users': users,
            'requests': len(rows),
            'transitions': dict(Counter(row['transition'] for row in rows)),
        },
        'by_transition': by_transition,
        'by_user': by_user,
        'by_first_divergence_bucket': by_prefix,
        'flips': flips,
        'weight_sensitivity': weight_summary,
        'observations': {
            'reference_only_prefix_median': by_transition['reference_only']['common_prefix_tokens']['median'],
            'candidate_only_prefix_median': by_transition['candidate_only']['common_prefix_tokens']['median'],
            'reference_only_prefix_le_16': by_transition['reference_only']['common_prefix_tokens']['le_16'],
            'candidate_only_prefix_le_16': by_transition['candidate_only']['common_prefix_tokens']['le_16'],
            'divergent_requests': len(flips),
            'known_boundary': (
                'Output-token decomposition cannot identify which layer or channel caused a flip; '
                'teacher-forced hidden/logit drift is required before allocating residual bits.'),
        },
        'boundary': (
            'development-only output decomposition; no evaluation labels for holdout; no '
            'counterfactual quality claim; no physical PIM execution'),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({
        'cohort': result['cohort'],
        'flip_prefix_medians': {
            'candidate_only': result['observations']['candidate_only_prefix_median'],
            'reference_only': result['observations']['reference_only_prefix_median'],
        },
        'output': str(args.output),
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
