"""Disjoint serving-stage accounting from a frozen visual-reuse execution."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from audit_omni_smoke_closure import digest, verify_source_snapshot


def account_stages(smoke):
    if (not smoke.get('complete') or not smoke.get('rows')
            or smoke.get('evaluation_label_files_read') != []
            or smoke.get('numerical_probe_overhead_in_service') is not False):
        raise ValueError('complete nonempty label-free serving without in-service probes required')
    rows = []
    seen = set()
    for request in smoke['rows']:
        question_id = request['question_id']
        if question_id in seen:
            raise ValueError('duplicate request identity')
        seen.add(question_id)
        names = ('service_seconds', 'query_retrieval_seconds', 'edge_embedding_seconds',
                 'evidence_planning_seconds', 'prefill_seconds', 'decode_seconds')
        if any(not isinstance(request.get(name), (int, float))
               or not math.isfinite(request[name]) or request[name] < 0 for name in names):
            raise ValueError('finite nonnegative stage timings required')
        before, after = request.get('visual_cache_before'), request.get('visual_cache_after')
        if before is None or after is None:
            raise ValueError('actual serving visual-cache counters required; reference timers are not serving timers')
        deltas = {}
        for name in ('resolve_seconds', 'miss_compute_seconds', 'hits', 'misses'):
            if any(not isinstance(value.get(name), (int, float))
                   or not math.isfinite(value[name]) or value[name] < 0 for value in (before, after)):
                raise ValueError('finite nonnegative cache counters required')
            delta = after[name] - before[name]
            if delta < 0:
                raise ValueError('cache counters must be monotone')
            deltas[name] = delta
        if deltas['hits'] + deltas['misses'] != 1:
            raise ValueError('one actual visual-batch lookup per request required')
        if (deltas['resolve_seconds'] > request['prefill_seconds']
                or deltas['miss_compute_seconds'] > deltas['resolve_seconds']):
            raise ValueError('visual path is nested within prefill, not an additional disjoint stage')
        stages = {
            'query_retrieval': request['query_retrieval_seconds'],
            'edge_embedding_and_candidate_preparation': request['edge_embedding_seconds'],
            'evidence_selection': request['evidence_planning_seconds'],
            'vision_cache_resolution_including_encoder_on_miss': deltas['resolve_seconds'],
            'remaining_reader_prefill': request['prefill_seconds'] - deltas['resolve_seconds'],
            'reader_decode': request['decode_seconds'],
        }
        remaining = request['service_seconds'] - sum(stages.values())
        if remaining < -1e-6:
            raise ValueError('disjoint serving timers exceed total service')
        stages['other_serving_work'] = max(0.0, remaining)
        rows.append({'question_id': question_id, 'service_seconds': request['service_seconds'],
                     'disjoint_stage_seconds': stages,
                     'visual_miss_encoder_seconds_nested_in_visual_resolution': deltas['miss_compute_seconds'],
                     'batch_hits': deltas['hits'], 'batch_misses': deltas['misses']})
    total = sum(row['service_seconds'] for row in rows)
    if total <= 0:
        raise ValueError('positive service duration required')
    stages = {name: sum(row['disjoint_stage_seconds'][name] for row in rows)
              for name in rows[0]['disjoint_stage_seconds']}
    return {
        'requests': len(rows), 'service_seconds': total,
        'disjoint_stage_seconds': stages,
        'disjoint_stage_service_fractions': {name: value / total for name, value in stages.items()},
        'visual_miss_encoder_seconds_nested_in_visual_resolution': sum(
            row['visual_miss_encoder_seconds_nested_in_visual_resolution'] for row in rows),
        'natural_batch_hits': sum(row['batch_hits'] for row in rows),
        'natural_batch_misses': sum(row['batch_misses'] for row in rows),
        'reader_cross_request_kv_reuse_executed': False,
        'rows': rows,
        'end_to_end_pim_speedup': None,
        'boundary': 'instrumented server diagnostic, not causal gain or a device profile; visual time is subtracted from prefill exactly once; validation/reference time is not serving time; remaining prefill is a residual, not an isolated language-kernel benchmark',
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    smoke = json.loads(args.smoke.read_text())
    registration_path = args.smoke.with_suffix('.registration.json')
    registration = json.loads(registration_path.read_text())
    if (registration['question_ids'] != [row['question_id'] for row in smoke['rows']]
            or registration['source_files_sha256'] != smoke['source_files_sha256']
            or not verify_source_snapshot(smoke, registration)):
        raise ValueError('frozen serving identity or source snapshot differs')
    result = {'schema_version': 'omni-mainline-bottleneck-audit-v1', **account_stages(smoke),
              'producer_smoke_sha256': digest(args.smoke),
              'registration_sha256': digest(registration_path),
              'script_sha256': digest(Path(__file__)),
              'evaluation_label_files_read': [],
              'deployment_capacity_is_research_objective': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps({name: result[name] for name in ('requests', 'disjoint_stage_service_fractions',
                                                    'natural_batch_hits', 'natural_batch_misses')}, indent=2))


if __name__ == '__main__':
    main()
