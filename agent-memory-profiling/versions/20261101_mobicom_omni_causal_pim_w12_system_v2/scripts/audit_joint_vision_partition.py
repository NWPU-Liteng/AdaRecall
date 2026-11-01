"""Audit a frozen reranker visual-partition diagnostic without rewriting it."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(result_path: Path, registration_path: Path, source_dir: Path):
    result = json.loads(result_path.read_text())
    registration = json.loads(registration_path.read_text())
    if not result.get('complete') or result.get('evaluation_label_files_read') != []:
        raise ValueError('complete label-free result required')
    if result.get('question_id') != registration.get('question_id'):
        raise ValueError('result and execution registration question differ')
    if result.get('source_run_sha256') != registration.get('source_run_sha256'):
        raise ValueError('result and source request provenance differ')
    if (registration.get('mainline_promotion_authorized') is True
            or result.get('mainline_promotion_authorized') is not False
            or result.get('main_experiment_closed') is not False
            or result.get('actual_pim_end_to_end_speedup') is not None):
        raise ValueError('diagnostic registration cannot authorize promotion')
    for name, expected in registration['sources_sha256'].items():
        snapshot = source_dir / name
        if not snapshot.is_file() or sha256(snapshot) != expected:
            raise ValueError(f'frozen source snapshot mismatch: {name}')
    if len(result['image_sha256']) != 2 or len(result['image_grid_thw']) != 2:
        raise ValueError('expected exactly two pinned images')
    comparisons = result['pooled_feature_comparisons'] + result['deepstack_comparisons']
    if not comparisons or any(not item.get('finite') or not item.get('same_shape_dtype')
                              for item in comparisons):
        raise ValueError('feature outputs are malformed or nonfinite')
    if result['all_features_bitwise_equal'] != all(item.get('bitwise_equal') for item in comparisons):
        raise ValueError('feature equality summary is inconsistent')
    return {
        'complete': True,
        'result_sha256': sha256(result_path),
        'registration_sha256': sha256(registration_path),
        'question_id': result['question_id'],
        'source_snapshots_verified': len(registration['sources_sha256']),
        'image_count': len(result['image_sha256']),
        'pooled_feature_count': len(result['pooled_feature_comparisons']),
        'deepstack_stage_count': len(result['deepstack_comparisons']),
        'all_features_bitwise_equal': result['all_features_bitwise_equal'],
        'first_batch_feature_gate_passed': result['all_features_bitwise_equal'],
        'exact_feature_reuse_qualified': False,
        'reference_synchronized_seconds': result['reference_profiled_seconds'],
        'reference_timing_field_note': 'legacy JSON key; this run used synchronized module hooks, not torch.profiler',
        'largest_pooled_max_abs_difference': max(
            item['max_absolute_difference'] for item in result['pooled_feature_comparisons']),
        'largest_deepstack_max_abs_difference': max(
            item['max_absolute_difference'] for item in result['deepstack_comparisons']),
        'mainline_promotion_authorized': False,
        'main_experiment_closed': False,
        'actual_pim_end_to_end_speedup': None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.result, args.registration, args.sources)
    with args.output.open('x') as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
