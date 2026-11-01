"""Compute-state opportunity bounds on immutable serving operands, without labels."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def audit_run(run):
    if not run.get('complete') or not run.get('rows') or run.get('evaluation_label_files_read'):
        raise ValueError('complete label-free serving run required')
    batches = set()
    raw_images = set()
    records = []
    total = saved = vision = 0.0
    for row in run['rows']:
        total += row['service_seconds']
        nested = row['service_stage_ledger']['nested_seconds']['vision_in_prefill_seconds']
        vision += nested
        tensors = row['processor_inputs_sha256']
        if row['images']:
            if 'pixel_values' not in tensors or 'image_grid_thw' not in tensors:
                raise ValueError('complete image producer operands required')
            key = digest({'reader_identity': run['reader_identity'],
                          'pixels': tensors['pixel_values'], 'grid': tensors['image_grid_thw'],
                          'images': [image['sha256'] for image in row['images']]})
            hit = key in batches
            saved += nested if hit else 0
            batches.add(key)
        else:
            if nested != 0 or 'pixel_values' in tensors or row['visual_patches'] != 0:
                raise ValueError('text-only request carries inconsistent visual work')
            key, hit = None, False
        raw_repeat = sum(image['sha256'] in raw_images for image in row['images'])
        raw_images.update(image['sha256'] for image in row['images'])
        records.append({'question_id': row['question_id'], 'whole_batch_producer': key,
                        'unbounded_exact_batch_hit': hit, 'raw_image_repeats': raw_repeat,
                        'zero_restore_cost_saving_upper_seconds': nested if hit else 0})
    return {'person_id': run['rows'][0]['person_id'], 'requests': len(records),
        'total_service_seconds': total, 'nested_vision_seconds': vision,
        'whole_batch_hits': sum(row['unbounded_exact_batch_hit'] for row in records),
        'raw_repeated_occurrences': sum(row['raw_image_repeats'] for row in records),
        'zero_restore_cost_saving_upper_seconds': saved,
        'zero_restore_cost_e2e_upper_ratio': total / (total - saved) if total > saved else None,
        'records': records}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('fresh opportunity audit required')
    results = [audit_run(json.loads(path.read_text())) for path in args.runs]
    if len({row['person_id'] for row in results}) != len(results):
        raise ValueError('one independent user stream per audit required')
    total = sum(row['total_service_seconds'] for row in results)
    saved = sum(row['zero_restore_cost_saving_upper_seconds'] for row in results)
    result = {'complete': True, 'per_user': results,
        'requests': sum(row['requests'] for row in results),
        'whole_batch_hits': sum(row['whole_batch_hits'] for row in results),
        'zero_restore_cost_saving_upper_seconds': saved,
        'zero_restore_cost_e2e_upper_ratio': total / (total - saved) if total > saved else None,
        'inputs_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in args.runs},
        'reader_state_reuse_qualified': False, 'actual_compute_state_reuse_executed': False,
        'boundary': 'unbounded, zero-management-cost visual-producer upper bound only; raw-image repeat is not a legal per-image feature hit; no Reader KV/recurrent reuse assumed; no future-hit information enters serving'}
    with args.output.open('x') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps({key: value for key, value in result.items() if key not in ('per_user', 'inputs_sha256')}, indent=2))


if __name__ == '__main__':
    main()
