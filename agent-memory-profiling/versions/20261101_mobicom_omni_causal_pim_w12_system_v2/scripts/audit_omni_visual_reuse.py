"""Audit natural visual-state reuse separately from semantic and PIM claims."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from audit_omni_smoke_closure import digest, empirical_latency, verify_source_snapshot


def audit_visual_records(smoke, registration, joint):
    rows = smoke.get('rows', [])
    store = smoke.get('visual_state_reuse')
    if (not smoke.get('complete') or not smoke.get('verify_generation') or not rows or not store
            or smoke.get('evaluation_label_files_read') != []):
        raise ValueError('complete independently verified label-free visual execution required')
    identities = [row['question_id'] for row in rows]
    if len(set(identities)) != len(identities) or identities != registration['question_ids']:
        raise ValueError('visual request identities differ from registration')
    if registration['source_files_sha256'] != smoke['source_files_sha256']:
        raise ValueError('visual source identities differ')
    if (store['policy'] != 'ordered-whole-batch-exact-v1' or store['capacity_bytes'] <= 0
            or registration['arguments']['visual_cache_bytes'] != store['capacity_bytes']
            or not smoke.get('visual_state_identity_files_sha256')):
        raise ValueError('missing pinned identity or declared bounded visual store')
    expected_model_identity = hashlib.sha256(json.dumps(
        smoke['visual_state_identity_files_sha256'], sort_keys=True).encode()).hexdigest()
    if store['model_identity'] != expected_model_identity:
        raise ValueError('visual model identity differs from checkpoint content hashes')
    if (joint['known_reservations']['declared_visual_state_budget_bytes'] != store['capacity_bytes']
            or not joint['known_joint_pool_fits'] or joint['end_to_end_speedup'] is not None):
        raise ValueError('visual reservation or PIM claim boundary failed')
    monotonic = ('hits', 'misses', 'evictions', 'rejected', 'bypasses', 'integrity_failures',
                 'image_hits', 'hit_transfer_seconds', 'miss_compute_seconds', 'resolve_seconds')
    previous = None
    prior_batches = set()
    observations = []
    for row in rows:
        before, after = row['visual_cache_before'], row['visual_cache_after']
        if (not row.get('images') or row['generation_equivalent'] is not True
                or row['visual_features_bitwise_equivalent'] is not True):
            raise ValueError('full token or bitwise feature equivalence failed')
        if previous is not None and previous != before:
            raise ValueError('visual cache changed between registered service requests')
        if previous is None and any(before[name] != 0 for name in monotonic):
            raise ValueError('visual cohort did not start from a cold store')
        for snapshot in (before, after):
            if (snapshot['capacity_bytes'] != store['capacity_bytes']
                    or snapshot['model_identity'] != store['model_identity']
                    or snapshot['processor_identity'] != store['processor_identity']
                    or not 0 <= snapshot['resident_bytes'] <= snapshot['peak_resident_bytes'] <= store['capacity_bytes']
                    or snapshot['bypasses'] or snapshot['integrity_failures']):
                raise ValueError('visual store identity/capacity/integrity gate failed')
        delta = {name: after[name] - before[name] for name in monotonic}
        if any(value < 0 for value in delta.values()) or delta['hits'] + delta['misses'] != 1:
            raise ValueError('exactly one service lookup and no reference cache lookups required')
        batch = tuple(image['sha256'] for image in row['images'])
        if delta['hits'] and batch not in prior_batches:
            raise ValueError('visual hit without a preceding natural ordered image batch')
        if delta['image_hits'] != len(batch) * delta['hits']:
            raise ValueError('batch and image hit counters differ')
        reference_times = row['reference_vision_seconds']
        if len(reference_times) != 1 or reference_times[0] <= 0:
            raise ValueError('one independent original-image encoder execution required')
        observations.append({'question_id': row['question_id'], 'batch_hit': bool(delta['hits']),
            'natural_image_batch_previously_seen': batch in prior_batches,
            'visual_path_seconds_including_lookup_integrity_and_cpu_store': delta['resolve_seconds'],
            'independent_reference_vision_seconds': reference_times[0],
            'diagnostic_reference_minus_visual_path_seconds': reference_times[0] - delta['resolve_seconds'],
            'prefill_seconds': row['prefill_seconds'], 'service_seconds': row['service_seconds']})
        prior_batches.add(batch)
        previous = after
    if previous != store or store['hits'] + store['misses'] != len(rows):
        raise ValueError('final visual store summary differs from execution ledger')
    if joint['observed_visual_state_peak_bytes'] != store['peak_resident_bytes']:
        raise ValueError('joint pool visual occupancy differs')
    hit_rows = [row for row in observations if row['batch_hit']]
    return {'schema_version': 'omni-visual-exact-reuse-audit-v1', 'execution_integrity': 'pass',
            'requests': len(rows), 'bitwise_equivalent_features': len(rows),
            'identical_complete_generation_token_sequences': len(rows),
            'natural_batch_hits': store['hits'], 'natural_batch_misses': store['misses'],
            'natural_batch_hit_rate': store['hits'] / len(rows),
            'natural_hit_question_ids': [row['question_id'] for row in hit_rows],
            'cache': store, 'rows': observations,
            'service_latency_diagnostic': empirical_latency([row['service_seconds'] for row in rows]),
            'prefill_latency_diagnostic': empirical_latency([row['prefill_seconds'] for row in rows]),
            'diagnostic_total_reference_minus_visual_path_seconds': sum(
                row['diagnostic_reference_minus_visual_path_seconds'] for row in observations),
            'end_to_end_speedup': None, 'measured_pim_speedup': None,
            'reader_kv_or_recurrent_state_reused': False,
            'main_experiment_closed': False, 'publication_ready': False,
            'boundary': 'natural development stream only; ordered whole-batch exact visual features; independent reference bypasses store; diagnostic encoder timings are order/cache/thermal confounded, not controlled E2E speedup or PIM measurements'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', type=Path, required=True)
    parser.add_argument('--joint-capacity', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('fresh visual reuse audit required')
    smoke = json.loads(args.smoke.read_text())
    registration_path = args.smoke.with_suffix('.registration.json')
    registration = json.loads(registration_path.read_text())
    joint = json.loads(args.joint_capacity.read_text())
    if joint['smoke_sha256'] != digest(args.smoke) or not verify_source_snapshot(smoke, registration):
        raise ValueError('visual producer or frozen source mismatch')
    result = audit_visual_records(smoke, registration, joint)
    result['inputs_sha256'] = {str(path): digest(path) for path in (
        args.smoke, registration_path, args.joint_capacity)}
    result['script_sha256'] = digest(Path(__file__))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps({name: result[name] for name in (
        'execution_integrity', 'requests', 'natural_batch_hits', 'natural_batch_hit_rate',
        'end_to_end_speedup')}))


if __name__ == '__main__':
    main()
