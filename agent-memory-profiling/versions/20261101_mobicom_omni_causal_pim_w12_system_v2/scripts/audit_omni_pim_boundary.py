"""Build an explicit boundary report for the conditional PIM model.

Architecture-model nanoseconds and measured server seconds are intentionally
kept in separate ledgers. The server table is a dimensionless sensitivity only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from omni_decode_lifecycle import normalized_service_sensitivity


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def empirical(values):
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError('positive finite timing values required')
    ordered = sorted(values)
    return {'count': len(ordered), 'sum_seconds': sum(ordered),
            'mean_seconds': sum(ordered) / len(ordered),
            'p50_nearest_rank_seconds': ordered[(len(ordered) + 1) // 2 - 1],
            'p95_nearest_rank_seconds': ordered[(95 * len(ordered) + 99) // 100 - 1],
            'max_seconds': ordered[-1]}


def required_fraction(decode_fraction: float, linear_speedup: float, target: float):
    if (not all(math.isfinite(value) for value in (decode_fraction, linear_speedup, target))
            or not 0 < decode_fraction <= 1 or target <= 1 or linear_speedup <= 1):
        raise ValueError('positive decode fraction and finite speedups above one required')
    denominator = decode_fraction * (1 - 1 / linear_speedup)
    return (1 - 1 / target) / denominator


def compare_decode_trace(reference, current):
    if (not reference.get('complete') or not current.get('complete')
            or reference.get('evaluation_label_files_read') != []
            or current.get('evaluation_label_files_read') != []):
        raise ValueError('complete label-isolated runs required')
    for field in ('reader_identity', 'reader_quantization', 'reader_patch_projection',
                  'prepared_audit_sha256', 'archive_sha256'):
        if reference[field] != current[field]:
            raise ValueError('immutable execution input changed: ' + field)
    if [row['question_id'] for row in reference['rows']] != [row['question_id'] for row in current['rows']]:
        raise ValueError('request order changed')
    if any(any(before[field] != after[field] for field in (
            'question_sha256', 'answer_token_ids', 'stop_reason', 'processor_inputs_sha256'))
           for before, after in zip(reference['rows'], current['rows'])):
        raise ValueError('answer token or stop state changed')
    before = {(item['phase'], item['name'], item['rows']): item
              for item in reference['operator_trace']['operators']}
    after = {(item['phase'], item['name'], item['rows']): item
             for item in current['operator_trace']['operators']}
    if (len(before) != len(reference['operator_trace']['operators'])
            or len(after) != len(current['operator_trace']['operators'])):
        raise ValueError('operator trace identities must be unique')
    if before.keys() != after.keys():
        raise ValueError('operator trace identity changed')
    for key in before:
        for field in ('in_features', 'out_features', 'calls', 'element_bytes', 'weight_bytes'):
            if before[key][field] != after[key][field]:
                raise ValueError(f'operator trace field changed: {key} {field}')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--aligned-host-root', type=Path, required=True)
    parser.add_argument('--frozen-projection', type=Path, required=True)
    parser.add_argument('--frozen-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--linear-speedup', type=float, default=1.7660982211008143)
    args = parser.parse_args()
    if args.output.exists() or not math.isfinite(args.linear_speedup) or args.linear_speedup <= 1:
        raise ValueError('fresh output and speedup above one required')
    projection = json.loads(args.frozen_projection.read_text())
    if not projection.get('complete') or projection.get('full_system_speedup') is not None:
        raise ValueError('conditional frozen projection required')
    users = []
    all_rows = []
    for user in ('0', '1', '2'):
        aligned_path = args.aligned_host_root / f'reference_user_{user}.json'
        frozen_path = args.frozen_root / f'reference_user_{user}.json'
        aligned = json.loads(aligned_path.read_text())
        frozen = json.loads(frozen_path.read_text())
        compare_decode_trace(frozen, aligned)
        profile = next(item for item in projection['per_user']
                       if Path(item['run']).name == frozen_path.name)
        if profile['run_sha256'] != digest(frozen_path):
            raise ValueError('projected frozen serving run changed')
        service = empirical([row['service_seconds'] for row in aligned['rows']])
        decode = sum(row['decode_seconds'] for row in aligned['rows'])
        total = sum(row['service_seconds'] for row in aligned['rows'])
        fraction = decode / total
        sweep = [normalized_service_sensitivity(fraction, args.linear_speedup, share)
                 for share in (0., .25, .5, .75, 1.)]
        users.append({'person_id': user, 'aligned_host_run': str(aligned_path),
                      'aligned_host_run_sha256': digest(aligned_path),
                      'frozen_run_sha256': digest(frozen_path),
                      'operator_trace_equivalence': 'pass',
                      'service': service, 'decode_seconds_sum': decode,
                      'decode_fraction_of_service': fraction,
                      'known_linear_host_ns': profile['known_linear_host_ns'],
                      'known_linear_hybrid_ns': profile['known_linear_hybrid_ns'],
                      'known_linear_warm_ratio': profile['known_linear_warm_ratio'],
                      'maximum_unmapped_residual_ns_for_1_3x_warm': profile['maximum_unmapped_residual_ns_for_1_3x_warm'],
                      'maximum_unmapped_residual_ns_for_1_3x_cold': profile['maximum_unmapped_residual_ns_for_1_3x_cold'],
                      'server_dimensionless_sensitivity': sweep,
                      'required_linear_decode_fraction_for_targets': {
                          str(target): required_fraction(fraction, args.linear_speedup, target)
                          for target in (1.3, 1.5)}})
        all_rows.extend(aligned['rows'])
    total_service = sum(row['service_seconds'] for row in all_rows)
    total_decode = sum(row['decode_seconds'] for row in all_rows)
    aggregate_fraction = total_decode / total_service
    result = {'complete': True, 'projection_sha256': digest(args.frozen_projection),
              'frozen_root': str(args.frozen_root), 'aligned_host_root': str(args.aligned_host_root),
              'linear_speedup_condition': args.linear_speedup,
              'users': users,
              'aggregate_aligned_host_service': empirical([row['service_seconds'] for row in all_rows]),
              'aggregate_decode_fraction_of_service': aggregate_fraction,
              'aggregate_server_dimensionless_sensitivity': [
                  normalized_service_sensitivity(aggregate_fraction, args.linear_speedup, share)
                  for share in (0., .25, .5, .75, 1.)],
              'aggregate_required_linear_decode_fraction_for_targets': {
                  str(target): required_fraction(aggregate_fraction, args.linear_speedup, target)
                  for target in (1.3, 1.5)},
              'architecture_time_units_not_server_seconds': True,
              'physical_pim_execution': False,
              'full_system_speedup': None,
              'boundary': 'operator-trace-aligned conditional model; architecture nanoseconds are not added to measured server seconds; server values are dimensionless Amdahl sensitivity, not calibrated LPDDR latency',
              'inputs_sha256': {str(path): digest(path) for path in (
                  args.frozen_projection, Path(__file__), Path(__file__).with_name('omni_decode_lifecycle.py'))}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'complete': True, 'aggregate_decode_fraction': aggregate_fraction,
                      'target_1_3_required_linear_fraction': result['aggregate_required_linear_decode_fraction_for_targets']['1.3'],
                      'target_1_5_required_linear_fraction': result['aggregate_required_linear_decode_fraction_for_targets']['1.5']}))


if __name__ == '__main__':
    main()
