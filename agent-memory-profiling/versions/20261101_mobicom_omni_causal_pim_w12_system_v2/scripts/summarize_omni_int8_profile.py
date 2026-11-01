"""Summarize non-overlapping CUDA activity attribution from frozen diagnostics."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from omni_decode_lifecycle import normalized_service_sensitivity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--profiles', type=Path, nargs='+', required=True)
    parser.add_argument('--boundary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--projection', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or len(args.profiles) < 2:
        raise ValueError('fresh output and at least two profiles required')
    boundary = json.loads(args.boundary.read_text())
    projection = json.loads(args.projection.read_text())
    if (not boundary.get('complete') or not projection.get('complete')
            or boundary['projection_sha256'] != hashlib.sha256(args.projection.read_bytes()).hexdigest()):
        raise ValueError('same frozen architecture projection required')
    selected_host = sum(item['host_per_call_ns'] * item['calls'] for user in projection['per_user']
                        for item in user['matrices'] if item['model_route'] == 'pim')
    selected_pim = sum(item['pim_per_call_ns'] * item['calls'] for user in projection['per_user']
                       for item in user['matrices'] if item['model_route'] == 'pim')
    selected_speedup = selected_host / selected_pim
    aggregate = {}
    rows = []
    for path in args.profiles:
        profile = json.loads(path.read_text())
        if (not profile.get('complete') or not profile.get('prefix_tokens_equal_frozen') == 133
                or len(profile.get('windows', [])) != 2):
            raise ValueError('complete two-window frozen profile required')
        for window in profile['windows']:
            if not window.get('kernel_activity_partition_valid'):
                raise ValueError('complete non-overlapping kernel attribution required')
            activity = window['kernel_activity_partition_microseconds']
            attributed = sum(activity[name] for name in activity if name != 'unattributed')
            for name, value in activity.items():
                aggregate[name] = aggregate.get(name, 0.) + value
            rows.append({'profile': str(path), 'first_decode_step': window['first_decode_step'],
                         'attributed_activity_microseconds': attributed,
                         'fractions': window['kernel_activity_fractions_of_attributed_decode']})
    attributed = sum(value for name, value in aggregate.items() if name != 'unattributed')
    fractions = {name: value / attributed for name, value in aggregate.items()
                 if name != 'unattributed'}
    decode_fraction = boundary['aggregate_decode_fraction_of_service']
    selected_fraction = fractions['selected_integer_product']
    activity_conditioned = normalized_service_sensitivity(
        decode_fraction, selected_speedup, selected_fraction)
    result = {'complete': True, 'profiles': rows,
              'aggregate_kernel_activity_microseconds': aggregate,
              'aggregate_kernel_activity_fractions': fractions,
              'profiled_decode_fraction_of_aligned_host_service': decode_fraction,
              'selected_integer_speedup_condition': selected_speedup,
              'whole_linear_model_ratio_not_selected_only': boundary['linear_speedup_condition'],
              'activity_conditioned_selected_only_sensitivity': activity_conditioned,
              'activity_conditioned_free_selected_removal_upper_bound': 1 / (1 - decode_fraction * selected_fraction),
              'interpretation': {
                  'selected_integer_product_fraction': 'non-overlapping CUDA kernel/memcpy activity, not critical-path wall time',
                  'activity_conditioned_sensitivity': 'counterfactual only: treats the activity fraction as if it were a critical-path decode fraction and accelerates only selected integer product',
                  'not_claimed': ['phone latency', 'physical PIM speedup', 'full decode critical-path fraction', 'quality improvement']},
              'physical_pim_execution': False, 'full_system_speedup': None,
              'inputs_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in (*args.profiles, args.boundary, args.projection, Path(__file__))}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'complete': True, 'selected_activity_fraction': selected_fraction,
                      'activity_conditioned_speedup': activity_conditioned['conditional_service_speedup']}))


if __name__ == '__main__':
    main()
