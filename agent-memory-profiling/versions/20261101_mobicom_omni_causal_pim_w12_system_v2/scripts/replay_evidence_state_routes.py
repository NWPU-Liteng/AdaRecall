"""Replay evidence-state routing over completed Reader traces without inference."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from pim_backend.evidence_state import compile_evidence_state, plan_evidence_route


EVENT_PHASES = (
    'query_parse_and_retrieval',
    'source_grounding',
    'proof_materialize',
)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def nearest_rank(values, quantile):
    if not values or not 0. <= quantile <= 1.:
        raise ValueError('nonempty values and quantile in [0, 1] required')
    ordered = sorted(float(value) for value in values)
    index = max(0, min(len(ordered) - 1, int((len(ordered) - 1) * quantile + .999999)))
    return ordered[index]


def load_trace_rows(run_root):
    run_root = Path(run_root)
    paths = sorted(path for path in run_root.glob('user_*_shard_*.json')
                   if not path.name.endswith('.partial.json'))
    if not paths:
        raise ValueError('completed user trace files required')
    rows = []
    file_digests = {}
    for path in paths:
        payload = json.loads(path.read_text())
        if payload.get('complete') is not True or not isinstance(payload.get('rows'), list):
            raise ValueError(f'complete trace required: {path}')
        file_digests[str(path)] = digest(path)
        for row in payload['rows']:
            if not row.get('question_id') or 'lifecycle_plan' not in row:
                raise ValueError(f'lifecycle-bound row required: {path}')
            rows.append(row)
    return rows, file_digests


def replay_rows(rows):
    replayed = []
    for row in rows:
        lifecycle = row['lifecycle_plan']
        state = compile_evidence_state(
            lifecycle, question_id=row['question_id'],
            reader_input_tokens=int(row['input_tokens']))
        event_seconds = {event['phase']: float(event.get('seconds', 0.))
                         for event in row.get('execution_events', [])}
        if any(phase not in event_seconds for phase in EVENT_PHASES):
            raise ValueError(f'incomplete evidence event chain: {row["question_id"]}')
        service_seconds = float(row['service_seconds'])
        materialize_seconds = event_seconds['proof_materialize']
        grounding_seconds = event_seconds['source_grounding']
        grounding_candidate_count = row.get('proof_plan', {}).get('grounding', {}).get('candidate_count')
        grounding_frontier = row.get('grounding_frontier')
        frontier_levels = row.get('grounding_frontier_levels', [])
        frontier_metrics = None
        if isinstance(grounding_frontier, dict):
            full_count = int(grounding_frontier.get('state_count', 0))
            full_bytes = int(grounding_frontier.get('packed_bytes', 0))
            first_level = frontier_levels[0] if frontier_levels else None
            selected_level = row.get('proof_plan', {}).get('grounding', {}).get('frontier_selected_level')
            selected = next((level for level in frontier_levels
                             if level.get('level') == selected_level), None)
            first_count = (first_level.get('candidate_count')
                           if first_level else full_count)
            linear_fraction = (first_count / full_count
                               if full_count > 0 else None)
            proxy_grounding = (grounding_seconds * linear_fraction
                               if linear_fraction is not None else None)
            proxy_service = (service_seconds - grounding_seconds + proxy_grounding
                             if proxy_grounding is not None else None)
            ttft_seconds = row.get('ttft_seconds')
            proxy_ttft = (float(ttft_seconds) - grounding_seconds + proxy_grounding
                          if proxy_grounding is not None and ttft_seconds is not None else None)
            frontier_metrics = {
                'input_candidate_count': int(grounding_frontier.get(
                    'input_candidate_count', full_count)),
                'frontier_candidate_count': full_count,
                'frontier_packed_bytes': full_bytes,
                'first_level_candidate_count': first_count,
                'first_level_packed_bytes': (first_level.get('packed_bytes')
                                             if first_level else full_bytes),
                'selected_level': selected_level,
                'selected_candidate_count': (selected.get('candidate_count')
                                             if selected else None),
                'selected_packed_bytes': (selected.get('packed_bytes')
                                          if selected else None),
                'linear_grounding_proxy_fraction': linear_fraction,
                'linear_service_proxy_seconds': proxy_service,
                'linear_ttft_proxy_seconds': proxy_ttft,
                'first_level_lost_source_provenance_count': (first_level.get(
                    'lost_source_provenance_count', 0) if first_level else 0),
                'first_level_lost_source_relation_count': (first_level.get(
                    'lost_source_relation_count', 0) if first_level else 0),
                'first_level_lost_binding_key_count': (first_level.get(
                    'lost_binding_key_count', 0) if first_level else 0),
                'level_count': len(frontier_levels),
                'semantic_pruning_applied': bool(
                    grounding_frontier.get('semantic_pruning_applied', False)),
            }
        route = plan_evidence_route(
            state, host_stage_seconds=materialize_seconds,
            total_service_seconds=service_seconds)
        zero_materialize_speedup = (service_seconds / (service_seconds - materialize_seconds)
                                    if service_seconds > materialize_seconds else None)
        replayed.append({
            'person_id': row.get('person_id'),
            'question_id': row['question_id'],
            'service_seconds': service_seconds,
            'event_seconds': event_seconds,
            'evidence_pipeline_seconds': sum(event_seconds[phase] for phase in EVENT_PHASES),
            'grounding_seconds': grounding_seconds,
            'grounding_candidate_count': grounding_candidate_count,
            'grounding_frontier': frontier_metrics,
            'state_materialize_seconds': materialize_seconds,
            'zero_materialize_speedup_upper_bound': zero_materialize_speedup,
            'state_count': state['state_count'],
            'unique_source_count': state['unique_source_count'],
            'payload_bytes': state['payload_bytes'],
            'metadata_bytes': state['metadata_bytes'],
            'packed_bytes': state['packed_bytes'],
            'reader_input_tokens': state['reader_input_tokens'],
            'route': route,
        })
    return replayed


def summarize(rows):
    if not rows:
        raise ValueError('nonempty replay required')
    service_total = sum(row['service_seconds'] for row in rows)
    materialize_total = sum(row['state_materialize_seconds'] for row in rows)
    evidence_total = sum(row['evidence_pipeline_seconds'] for row in rows)
    grounding_rows = [row['grounding_seconds'] for row in rows]
    candidate_counts = [row['grounding_candidate_count'] for row in rows
                        if row['grounding_candidate_count'] is not None]
    zero_materialize_bound = (service_total / (service_total - materialize_total)
                              if service_total > materialize_total else None)
    routes = {}
    for row in rows:
        chosen = row['route']['chosen_backend']
        routes[chosen] = routes.get(chosen, 0) + 1
    frontier_rows = [row['grounding_frontier'] for row in rows
                     if row.get('grounding_frontier') is not None]
    frontier_summary = None
    if frontier_rows:
        first_counts = [row['first_level_candidate_count'] for row in frontier_rows]
        selected_counts = [row['selected_candidate_count'] for row in frontier_rows
                           if row['selected_candidate_count'] is not None]
        full_counts = [row['frontier_candidate_count'] for row in frontier_rows]
        first_bytes = [row['first_level_packed_bytes'] for row in frontier_rows]
        full_bytes = [row['frontier_packed_bytes'] for row in frontier_rows]
        linear_service = [row['linear_service_proxy_seconds'] for row in frontier_rows
                          if row['linear_service_proxy_seconds'] is not None]
        linear_ttft = [row['linear_ttft_proxy_seconds'] for row in frontier_rows
                       if row['linear_ttft_proxy_seconds'] is not None]
        lost_sources = [row['first_level_lost_source_provenance_count'] for row in frontier_rows]
        lost_relations = [row['first_level_lost_source_relation_count'] for row in frontier_rows]
        lost_bindings = [row['first_level_lost_binding_key_count'] for row in frontier_rows]
        frontier_summary = {
            'requests': len(frontier_rows),
            'first_level_candidate_count': {
                'p50': nearest_rank(first_counts, .5),
                'p95': nearest_rank(first_counts, .95),
            },
            'selected_candidate_count': ({
                'p50': nearest_rank(selected_counts, .5),
                'p95': nearest_rank(selected_counts, .95),
            } if selected_counts else None),
            'frontier_candidate_count': {
                'p50': nearest_rank(full_counts, .5),
                'p95': nearest_rank(full_counts, .95),
            },
            'first_level_candidate_reduction_fraction': sum(
                1. - first / full for first, full in zip(first_counts, full_counts)
                if full > 0) / len(frontier_rows),
            'selected_candidate_reduction_fraction': ({
                'mean': sum(1. - row['selected_candidate_count'] / row['frontier_candidate_count']
                            for row in frontier_rows
                            if row['selected_candidate_count'] is not None
                            and row['frontier_candidate_count'] > 0) / len(selected_counts),
            } if selected_counts else None),
            'linear_cost_proxy': {
                'service_seconds_p50': (nearest_rank(linear_service, .5)
                                        if linear_service else None),
                'ttft_seconds_p50': (nearest_rank(linear_ttft, .5)
                                     if linear_ttft else None),
                'assumption': 'grounding time scales linearly with candidate count; not a measured speedup',
            },
            'first_level_packed_bytes': {
                'p50': nearest_rank(first_bytes, .5),
                'p95': nearest_rank(first_bytes, .95),
            },
            'frontier_packed_bytes': {
                'p50': nearest_rank(full_bytes, .5),
                'p95': nearest_rank(full_bytes, .95),
            },
            'first_level_quality_risk_indicators': {
                'requests_with_lost_source_provenance': sum(value > 0 for value in lost_sources),
                'requests_with_lost_source_relations': sum(value > 0 for value in lost_relations),
                'requests_with_lost_binding_keys': sum(value > 0 for value in lost_bindings),
                'max_lost_source_provenance_count': max(lost_sources),
                'max_lost_source_relation_count': max(lost_relations),
                'max_lost_binding_key_count': max(lost_bindings),
            },
            'boundary': 'frontier descriptor accounting only; no quality or E2E speedup claim',
        }
    return {
        'requests': len(rows),
        'service_seconds_total': service_total,
        'evidence_pipeline_seconds_total': evidence_total,
        'state_materialize_seconds_total': materialize_total,
        'evidence_pipeline_fraction': evidence_total / service_total,
        'grounding_seconds_total': sum(grounding_rows),
        'grounding_fraction': sum(grounding_rows) / service_total,
        'state_materialize_fraction': materialize_total / service_total,
        'zero_materialize_speedup_upper_bound': zero_materialize_bound,
        'state_count': {
            'p50': nearest_rank([row['state_count'] for row in rows], .5),
            'p95': nearest_rank([row['state_count'] for row in rows], .95),
        },
        'packed_bytes': {
            'p50': nearest_rank([row['packed_bytes'] for row in rows], .5),
            'p95': nearest_rank([row['packed_bytes'] for row in rows], .95),
        },
        'grounding_candidate_count': ({
            'p50': nearest_rank(candidate_counts, .5),
            'p95': nearest_rank(candidate_counts, .95),
        } if candidate_counts else None),
        'route_counts': routes,
        'candidate_frontier': frontier_summary,
        'boundary': 'trace replay only; no model inference, no labels, no PIM execution or speedup claim',
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('fresh output required')
    rows, file_digests = load_trace_rows(args.run_root)
    replayed = replay_rows(rows)
    payload = {
        'schema': 'evidence-state-route-replay-v1',
        'run_root': str(args.run_root),
        'input_file_sha256': file_digests,
        'summary': summarize(replayed),
        'rows': replayed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(payload['summary'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
