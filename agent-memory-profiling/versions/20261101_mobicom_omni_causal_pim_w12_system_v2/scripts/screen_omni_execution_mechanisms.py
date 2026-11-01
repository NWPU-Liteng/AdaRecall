"""Screen all observed requests before choosing a reuse or PIM mechanism."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from audit_omni_smoke_closure import digest


def summarize(audit, bank_count):
    if not audit.get('complete') or bank_count < 1:
        raise ValueError('complete whole-cohort audit and positive physical bank count required')
    rows = [row for user in audit['users'] for row in user['rows']]
    if not rows or len(rows) != audit['requests']:
        raise ValueError('all registered requests must remain in the denominator')
    matched = sorted(row['prefix_opportunity']['matched_prefix_tokens'] for row in rows)
    inputs = sum(row['input_tokens'] for row in rows)
    if inputs <= 0 or any(value < 0 for value in matched):
        raise ValueError('nonempty inputs and nonnegative prefix opportunity required')
    screenings = []
    for row in rows:
        graph = row['graph_analysis']
        groups = graph['shared_linear_input_groups_with_unchanged_write_epoch']
        duplicate_input_bytes = 0
        conditional_bank_bytes = 0
        for group in groups:
            input_sizes = {linear['input_logical_bytes'] for linear in group}
            if len(input_sizes) != 1:
                raise ValueError('shared exact input must have one logical size')
            size = input_sizes.pop()
            duplicate_input_bytes += (len(group) - 1) * size
            deliveries = [min(bank_count, linear['out_features']) for linear in group]
            conditional_bank_bytes += size * (sum(deliveries) - max(deliveries))
        screenings.append({'question_id': row['question_id'],
            'shared_input_groups': len(groups),
            'redundant_logical_input_payload_bytes': duplicate_input_bytes,
            'conditional_repeated_bank_input_payload_bytes': conditional_bank_bytes,
            'linear_weight_logical_read_bytes': graph['linear_weight_logical_read_bytes_per_step'],
            'bank_input_payload_to_linear_weight_payload_ratio': (
                conditional_bank_bytes / graph['linear_weight_logical_read_bytes_per_step']
                if graph['linear_weight_logical_read_bytes_per_step'] else None),
            'operator_counts': graph['operator_counts'],
            'visible_mutated_cache_storage_bytes': graph['visible_mutated_cache_storage_bytes']})
    return {'requests': len(rows), 'full_greedy_token_matches': audit['complete_greedy_token_matches'],
        'input_prefix_opportunities': {
            'matched_tokens_including_common_system_prompt': sum(matched), 'total_input_tokens': inputs,
            'matched_token_fraction_not_speedup': sum(matched) / inputs,
            'p50_tokens_nearest_rank': matched[math.ceil(len(matched) * .5) - 1],
            'p95_tokens_nearest_rank': matched[math.ceil(len(matched) * .95) - 1],
            'max_tokens': max(matched),
            'boundary': 'unbounded past-only exact multimodal input prefix opportunities; includes generic prompt; not qualified KV state or time saved'},
        'all_request_mechanism_screen': screenings,
        'bank_count_from_unchanged_profile': bank_count,
        'cross_request_reader_state_reuse_executed': False,
        'shared_input_pim_fusion_executed': False,
        'end_to_end_speedup': None,
        'boundary': 'conditional input-payload deduplication only if legal co-resident kernels use the same unchanged activation; does not reduce weight payload; no free buffering, weight placement, command savings, DRAM traffic, throughput or E2E claim; opaque fused kernels need explicit boundaries before complete scheduling'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads(args.audit.read_text())
    profile = json.loads(args.profile.read_text())
    banks = profile['config']['channels'] * profile['config']['banks_per_channel']
    result = {'schema_version': 'omni-whole-cohort-mechanism-screen-v1', **summarize(audit, banks),
              'producer_audit_sha256': digest(args.audit), 'profile_sha256': digest(args.profile),
              'profile_calibration_status': profile['calibration_status'],
              'script_sha256': digest(Path(__file__)), 'evaluation_label_files_read': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps({'requests': result['requests'],
        'input_prefix_opportunities': result['input_prefix_opportunities'],
        'per_request_payload_bytes': sorted({row['conditional_repeated_bank_input_payload_bytes']
            for row in result['all_request_mechanism_screen']})}, indent=2))


if __name__ == '__main__':
    main()
