#!/usr/bin/env python3
"""Summarize development-only r0/r2 teacher-forced drift without a quality claim."""
from __future__ import annotations

import argparse
from collections import defaultdict
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


def median(values):
    return statistics.median(values) if values else None


def linear_layer_elements(records: list[dict]) -> dict[int, int]:
    values = defaultdict(int)
    for record in records:
        marker = '.layers.'
        if marker not in record['name']:
            continue
        layer = record['name'].split(marker, 1)[1].split('.', 1)[0]
        if layer.isdigit():
            values[int(layer)] += record['weight_elements']
    return dict(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--drift-r0', type=Path, required=True)
    parser.add_argument('--drift-r2', type=Path, required=True)
    parser.add_argument('--failure-decomposition', type=Path, required=True)
    parser.add_argument('--weight-sensitivity', type=Path, required=True)
    parser.add_argument('--development-evidence', type=Path, required=True)
    parser.add_argument('--capacity-summary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    failure = json.loads(args.failure_decomposition.read_text())
    if failure.get('cohort', {}).get('users') != ['0', '1', '2']:
        raise ValueError('development-only decomposition required')
    flips = {row['question_id']: row for row in failure['flips']}
    if len(flips) != 13:
        raise ValueError('13 development flips required')

    arms = {}
    layer_means = {}
    for name, root in (('r0', args.drift_r0), ('r2', args.drift_r2)):
        manifest = json.loads((root / 'manifest.json').read_text())
        if manifest.get('complete') is not True:
            raise ValueError(f'complete {name} drift manifest required')
        rows = {}
        for path in sorted(root.glob(f'{name}_shard_*.json')):
            document = json.loads(path.read_text())
            if document.get('complete') is not True:
                raise ValueError(f'incomplete drift shard: {path}')
            for row in document['rows']:
                rows[row['question_id']] = row
        if set(rows) != set(flips):
            raise ValueError(f'{name} drift does not cover exactly the 13 flips')
        arms[name] = rows
        layer_means[name] = {
            str(layer): statistics.mean(row['layers'][layer]['relative_l2_mean']
                                        for row in rows.values())
            for layer in range(33)
        }

    rows = []
    for question_id in sorted(flips, key=lambda value: (int(flips[value]['person_id']), value)):
        left, right = arms['r0'][question_id], arms['r2'][question_id]
        max_left = max(left['layers'], key=lambda row: row['relative_l2_mean'])
        max_right = max(right['layers'], key=lambda row: row['relative_l2_mean'])
        rows.append({
            'question_id': question_id,
            'person_id': flips[question_id]['person_id'],
            'transition': flips[question_id]['transition'],
            'output_common_prefix_tokens': flips[question_id]['common_prefix_tokens'],
            'r0': {
                'max_hidden_relative_l2_mean': max_left['relative_l2_mean'],
                'max_hidden_layer': max_left['layer'],
                'first_layer_relative_mean_ge_1pct': next(
                    (row['layer'] for row in left['layers']
                     if row['relative_l2_mean'] >= 0.01), None),
                'top1_agreement': left['logits']['top1_agreement'],
                'positions': left['logits']['positions'],
                'first_top1_flip_position': left['logits']['first_top1_flip_position'],
                'margin_ratio_candidate_over_reference': left['logits'][
                    'margin_ratio_candidate_over_reference'],
                'logit_cosine_mean': left['logits']['logit_cosine_mean'],
            },
            'r2': {
                'max_hidden_relative_l2_mean': max_right['relative_l2_mean'],
                'max_hidden_layer': max_right['layer'],
                'first_layer_relative_mean_ge_1pct': next(
                    (row['layer'] for row in right['layers']
                     if row['relative_l2_mean'] >= 0.01), None),
                'top1_agreement': right['logits']['top1_agreement'],
                'positions': right['logits']['positions'],
                'first_top1_flip_position': right['logits']['first_top1_flip_position'],
                'margin_ratio_candidate_over_reference': right['logits'][
                    'margin_ratio_candidate_over_reference'],
                'logit_cosine_mean': right['logits']['logit_cosine_mean'],
            },
            'r2_minus_r0_max_hidden_relative_l2_mean': (
                max_right['relative_l2_mean'] - max_left['relative_l2_mean']),
        })

    r0_jumps = sorted(((layer_means['r0'][str(layer)] - layer_means['r0'][str(layer - 1)],
                        layer) for layer in range(1, 33)), reverse=True)
    r2_jumps = sorted(((layer_means['r2'][str(layer)] - layer_means['r2'][str(layer - 1)],
                        layer) for layer in range(1, 33)), reverse=True)
    weights = json.loads(args.weight_sensitivity.read_text())
    elements = linear_layer_elements(weights['records'])
    capacity_summary = json.loads(args.capacity_summary.read_text())
    base = next((row for row in capacity_summary.get('variants', [])
                 if row.get('variant') == 'base'), None)
    if base is None:
        raise ValueError('base capacity summary required')
    capacity = base['capacity_bytes']
    r2_pool = base['pool_total_bytes']
    spare = capacity - r2_pool
    upgrade_bytes = {layer: math.ceil(count / 4) for layer, count in elements.items()}
    ranked_layers = [layer for _, layer in r2_jumps if layer <= 31]
    ranked_unique = []
    for layer in ranked_layers:
        if layer not in ranked_unique:
            ranked_unique.append(layer)
    cumulative = 0
    feasible = []
    for layer in ranked_unique:
        if cumulative + upgrade_bytes[layer] > spare:
            break
        feasible.append(layer)
        cumulative += upgrade_bytes[layer]

    transition_summary = {}
    for transition in ('candidate_only', 'reference_only'):
        selected_rows = [row for row in rows if row['transition'] == transition]
        transition_summary[transition] = {
            'requests': len(selected_rows),
            'output_prefix_median': median([
                row['output_common_prefix_tokens'] for row in selected_rows]),
            'r0_max_hidden_relative_l2_mean': median([
                row['r0']['max_hidden_relative_l2_mean'] for row in selected_rows]),
            'r2_max_hidden_relative_l2_mean': median([
                row['r2']['max_hidden_relative_l2_mean'] for row in selected_rows]),
            'r0_top1_agreement_rate': (
                sum(row['r0']['top1_agreement'] for row in selected_rows)
                / sum(row['r0']['positions'] for row in selected_rows)),
            'r2_top1_agreement_rate': (
                sum(row['r2']['top1_agreement'] for row in selected_rows)
                / sum(row['r2']['positions'] for row in selected_rows)),
        }

    result = {
        'schema': 'hyperpim-sc-development-flip-drift-summary-v1',
        'complete': True,
        'as_of': datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds'),
        'scope': 'development users 0,1,2 only; 13 flipped requests; no holdout data',
        'inputs': {
            str(args.drift_r0 / 'manifest.json'): digest(args.drift_r0 / 'manifest.json'),
            str(args.drift_r2 / 'manifest.json'): digest(args.drift_r2 / 'manifest.json'),
            str(args.failure_decomposition): digest(args.failure_decomposition),
            str(args.weight_sensitivity): digest(args.weight_sensitivity),
            str(args.development_evidence): digest(args.development_evidence),
            str(args.capacity_summary): digest(args.capacity_summary),
        },
        'questions': rows,
        'transition_summary': transition_summary,
        'global': {
            'r0_mean_max_hidden_relative_l2_mean': statistics.mean(
                row['r0']['max_hidden_relative_l2_mean'] for row in rows),
            'r2_mean_max_hidden_relative_l2_mean': statistics.mean(
                row['r2']['max_hidden_relative_l2_mean'] for row in rows),
            'relative_reduction_from_r2': 1 - (
                statistics.mean(row['r2']['max_hidden_relative_l2_mean'] for row in rows)
                / statistics.mean(row['r0']['max_hidden_relative_l2_mean'] for row in rows)),
            'r0_top1_agreement_rate': (
                sum(row['r0']['top1_agreement'] for row in rows)
                / sum(row['r0']['positions'] for row in rows)),
            'r2_top1_agreement_rate': (
                sum(row['r2']['top1_agreement'] for row in rows)
                / sum(row['r2']['positions'] for row in rows)),
            'r2_improved_max_hidden_requests': sum(
                row['r2_minus_r0_max_hidden_relative_l2_mean'] < 0 for row in rows),
        },
        'layer_cumulative_hidden_relative_l2_mean': layer_means,
        'largest_layer_increments': {
            'r0': [{'layer': layer, 'increment': value} for value, layer in r0_jumps[:10]],
            'r2': [{'layer': layer, 'increment': value} for value, layer in r2_jumps[:10]],
        },
        'capacity_hypothesis': {
            'r2_pool_total_bytes': r2_pool,
            'capacity_bytes': capacity,
            'spare_bytes': spare,
            'upgrade_2bit_to_4bit_bytes_by_layer': {
                str(layer): value for layer, value in sorted(upgrade_bytes.items())},
            'layer_rank_by_r2_increment': ranked_unique,
            'feasible_upgrade_layers_in_rank_order': feasible,
            'feasible_upgrade_bytes': cumulative,
            'status': 'hypothesis_only_not_an_executed_quality_result',
        },
        'observations': {
            'r2_reduces_hidden_drift_but_not_output_flips': (
                'fixed r2 lowers cumulative hidden error but all 13 development flips remain '
                'in the observed outputs; 8 are candidate-only and 5 are reference-only'),
            'error_is_not_localized_to_one_layer': (
                'the largest cumulative hidden drift is at the final state, while measurable '
                'error starts at layer 1 and receives additional increments at layers 3, 16, '
                '31 and 32'),
            'implication': (
                'a global 2-bit residual is not a sufficient quality mechanism; the next '
                'candidate must allocate residual precision under the fixed 8 GiB capacity and '
                'be validated on development data before any new sealed confirmation'),
            'small_sample_boundary': (
                '13 flips cannot estimate population quality or select a final allocation; the '
                'capacity list is a pre-experiment hypothesis, not a completed method'),
        },
        'boundary': (
            'development-only teacher-forced mechanism diagnosis; no holdout users, no '
            'counterfactual quality gate, no physical PIM execution'),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({
        'global': result['global'],
        'largest_r2_layer_increments': result['largest_layer_increments']['r2'][:5],
        'capacity_hypothesis': result['capacity_hypothesis'],
        'output': str(args.output),
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
