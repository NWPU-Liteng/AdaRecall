"""Keep full-service measurements, diagnostic replay and hardware models separate."""
from __future__ import annotations

import math

from audit_omni_smoke_closure import empirical_latency


def _semantic_evidence_plan(plan):
    if plan is None:
        return None
    if not isinstance(plan, dict):
        raise ValueError('evidence plan must be an object')
    keep = ('text', 'images', 'accepted_atom_ids',
            'text_tokens', 'text_budget', 'max_images')
    result = {key: plan[key] for key in keep if key in plan}
    if 'selected_hyperedges' in plan:
        result['selected_hyperedges'] = [
            {key: edge[key] for key in ('edge_id', 'atom_ids')} if isinstance(edge, dict) else edge
            for edge in plan['selected_hyperedges']]
    return result


def full_service_summary(serving_runs):
    rows = []
    for run in serving_runs:
        if (not run.get('complete') or not run.get('performance_comparable')
                or run.get('verify_generation') or run.get('numerical_probe_overhead_in_service')
                or run.get('evaluation_label_files_read') != []):
            raise ValueError('complete performance-only, label-isolated serving data required')
        rows.extend(run['rows'])
    if not rows:
        raise ValueError('whole nonempty serving cohort required')
    for row in rows:
        values = [row[field] for field in ('retrieval_seconds', 'prefill_seconds', 'decode_seconds', 'service_seconds')]
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError('finite nonnegative wall-clock measurements required')
        if not math.isclose(sum(values[:3]), values[3], abs_tol=1e-6):
            raise ValueError('service stages must partition, not double count visual prefill')
        if 'service_stage_ledger' in row:
            ledger = row['service_stage_ledger']
            parts = ledger['disjoint_seconds']
            nested = ledger['nested_seconds']
            if (any(not math.isfinite(value) or value < 0 for value in (*parts.values(), *nested.values()))
                    or not math.isclose(sum(parts.values()), row['service_seconds'], abs_tol=1e-6)
                    or not math.isclose(ledger['total_service_seconds'], row['service_seconds'], abs_tol=1e-6)
                    or not math.isclose(parts['prefill_seconds'], row['prefill_seconds'], abs_tol=1e-6)
                    or not math.isclose(parts['decode_seconds'], row['decode_seconds'], abs_tol=1e-6)
                    or not math.isclose(sum(nested.values()), row['prefill_seconds'], abs_tol=1e-6)):
                raise ValueError('fine-grained service ledger must partition and retain nested vision')
    result = {'requests': len(rows), 'service_seconds': empirical_latency([row['service_seconds'] for row in rows]),
        'stages_seconds': {field: empirical_latency([row[field] for row in rows]) for field in (
            'retrieval_seconds', 'prefill_seconds', 'decode_seconds')},
        'decode_fraction_of_total_service': sum(row['decode_seconds'] for row in rows) /
                                           sum(row['service_seconds'] for row in rows),
        'timing_origin': 'measured_server_wall_clock_seconds',
        'scope': 'online serving: query retrieval/evidence, raw-image preparation, vision within prefill, decode and answer conversion; excludes model/index startup, network and evaluation',
        'retrieval_remainder_boundary': 'historical retrieval_seconds is service minus prefill minus decode, including image/input processing and answer conversion; not pure retrieval kernel time'}
    if all('service_stage_ledger' in row for row in rows):
        result['fine_grained_service_stages'] = {
            group: {field: empirical_latency([row['service_stage_ledger'][group][field] for row in rows])
                    for field in rows[0]['service_stage_ledger'][group]}
            for group in ('disjoint_seconds', 'nested_seconds')}
    return result


def measured_e2e_pair(reference, candidate):
    required_contract = ('scope', 'hardware_identity', 'concurrency', 'startup_policy',
                         'cohort_registration_sha256', 'input_archive_sha256', 'reader_identity')
    contract = reference.get('measurement_contract', {})
    other = candidate.get('measurement_contract', {})
    if any(not contract.get(field) or contract[field] != other.get(field) for field in required_contract):
        raise ValueError('paired full-service scope, hardware, cohort and startup contract required')
    for run in (reference, candidate):
        if (run.get('timing_origin') != 'measured_server_wall_clock_seconds'
                or run.get('diagnostic_graph_trace_in_service') is not False):
            raise ValueError('model cycles or diagnostic graph replay cannot qualify measured E2E')
        full_service_summary([run])
    expected, actual = reference['rows'], candidate['rows']
    identities = [row['question_id'] for row in expected]
    if len(expected) != len(actual) or len(set(identities)) != len(identities):
        raise ValueError('paired full-cohort denominator mismatch')
    for before, after in zip(expected, actual):
        for field in ('question_id', 'question_sha256', 'images',
                      'answer_token_ids', 'stop_reason'):
            if before[field] != after[field]:
                raise ValueError('paired full-service input/order/complete-output mismatch')
        if _semantic_evidence_plan(before.get('hypergraph_evidence_plan')) != _semantic_evidence_plan(
                after.get('hypergraph_evidence_plan')):
            raise ValueError('paired full-service semantic evidence plan mismatch')
        if before['service_seconds'] <= 0 or after['service_seconds'] <= 0:
            raise ValueError('positive measured service duration required')
    baseline = empirical_latency([row['service_seconds'] for row in expected])
    optimized = empirical_latency([row['service_seconds'] for row in actual])
    return {'requests': len(expected), 'measurement_contract': contract,
        'measured_server_end_to_end_speedup': baseline['sum_seconds'] / optimized['sum_seconds'],
        'p50_latency_ratio': baseline['p50_nearest_rank_seconds'] / optimized['p50_nearest_rank_seconds'],
        'p95_latency_ratio': baseline['p95_nearest_rank_seconds'] / optimized['p95_nearest_rank_seconds'],
        'reference_service_seconds': baseline, 'candidate_service_seconds': optimized,
        'boundary': 'whole-cohort matched complete-serving wall-clock ratios on declared server; not LPDDR-PIM chip speedup or official task success'}


def diagnostic_replay_summary(results):
    rows = [row for result in results for row in result['rows']]
    if not rows:
        raise ValueError('nonempty diagnostic cohort required')
    bracket_residuals = []
    for row in rows:
        timing = row['diagnostic_latency']
        if (timing['evidence_retrieval_included'] is not False
                or timing['graph_observation_included'] is not True
                or timing['performance_comparable'] is not False):
            raise ValueError('diagnostic scope must not be relabeled full serving')
        parts = [timing[field] for field in ('input_preparation_seconds', 'prefill_seconds',
                                            'decode_with_graph_probe_seconds')]
        total = timing['frozen_evidence_request_seconds']
        residual = total - sum(parts)
        if (any(not math.isfinite(value) or value < 0 for value in parts)
                or not math.isfinite(total) or residual < -1e-9):
            raise ValueError('disjoint diagnostic timing stages required')
        bracket_residuals.append(max(0., residual))
    return {'requests': len(rows),
        'frozen_evidence_request_seconds': empirical_latency([
            row['diagnostic_latency']['frozen_evidence_request_seconds'] for row in rows]),
        'stages_seconds': {field: empirical_latency([row['diagnostic_latency'][field] for row in rows])
                          for field in ('input_preparation_seconds', 'prefill_seconds', 'decode_with_graph_probe_seconds')},
        'timer_bracket_unassigned_seconds': empirical_latency(bracket_residuals),
        'performance_comparable': False,
        'scope': 'frozen evidence/raw images through greedy token completion, excluding evidence retrieval and answer text conversion; includes graph observer and diagnostic input artifact write',
        'end_to_end_speedup': None}
