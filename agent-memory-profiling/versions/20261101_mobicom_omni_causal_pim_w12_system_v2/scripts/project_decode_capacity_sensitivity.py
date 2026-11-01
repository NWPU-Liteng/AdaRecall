"""Project complete observed decode Linear placement across PIM capacities."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pim_backend.decode_gemv import project_trace
from pim_backend.lpddr_bank_pim import BankPimConfig


def _merge_operators(run):
    merged = {}
    traces = []
    for row in run.get('rows', []):
        retrieval = row.get('proof_plan', {}).get('retrieval', {})
        trace = retrieval.get('linear_operator_trace')
        if trace is not None:
            traces.append(trace)
        for attempt in row.get('proof_plan', {}).get('grounding', {}).get('frontier_attempts', []):
            trace = attempt.get('linear_operator_trace')
            if trace is not None:
                traces.append(trace)
    for trace in traces:
        for operator in trace.get('operators', []):
            identity = tuple(operator.get(field) for field in (
                'name', 'phase', 'rows', 'in_features', 'out_features',
                'element_bytes', 'weight_bytes'))
            if any(value is None for value in identity):
                raise ValueError('operator trace is missing a placement identity field')
            if identity not in merged:
                merged[identity] = dict(operator)
            else:
                merged[identity]['calls'] += operator.get('calls', 0)
    if not merged:
        raise ValueError('no operator traces found')
    return list(merged.values()), len(traces)


def _host_rows(host_run):
    if host_run is None:
        return {}
    return {row['question_id']: row for row in host_run.get('rows', [])}


def _required_service_fraction(linear_speedup, target_speedup=1.05):
    if linear_speedup <= 1.0:
        return None
    denominator = 1.0 - 1.0 / linear_speedup
    required = (1.0 - 1.0 / target_speedup) / denominator
    return required if required <= 1.0 else None


def project_capacity_sensitivity(run, profile, capacities_gib, host_run=None):
    if not run.get('complete'):
        raise ValueError('complete profiled run required')
    if run.get('evaluation_label_files_read') != []:
        raise ValueError('profiled run must be label isolated')
    if not run.get('pre_reader_operator_profile', {}).get('enabled'):
        raise ValueError('pre-reader operator profile is required for this projection')
    if not capacities_gib or any(value <= 0 for value in capacities_gib):
        raise ValueError('positive capacity scenarios required')
    operators, trace_count = _merge_operators(run)
    profile_config = dict(profile['config'])
    profile_capacity = int(profile_config['pim_capacity_bytes'])
    physical_capacity = int(profile.get('physical_capacity_bytes', profile_capacity))
    if profile_capacity > physical_capacity:
        raise ValueError('declared PIM pool cannot exceed modeled physical capacity')
    host_by_question = _host_rows(host_run)
    scenarios = []
    for capacity_gib in capacities_gib:
        capacity_bytes = int(capacity_gib * 1024**3)
        config = BankPimConfig(**{**profile_config, 'pim_capacity_bytes': capacity_bytes})
        projection = project_trace(operators, config)
        linear_speedup = projection['warm_linear_speedup_scenario']
        required_fraction = _required_service_fraction(linear_speedup)
        within_current_pool = capacity_bytes <= profile_capacity
        within_physical_capacity = capacity_bytes <= physical_capacity
        if within_current_pool:
            physical_status = 'current_declared_pim_pool'
        elif not within_physical_capacity:
            physical_status = 'exceeds_profile_physical_capacity'
        elif capacity_bytes == physical_capacity:
            physical_status = 'uses_all_physical_memory_host_reserve_unmodeled'
        else:
            physical_status = 'requires_pool_reallocation_and_host_reserve_validation'
        scenarios.append({
            'capacity_gib': capacity_gib,
            'capacity_bytes': capacity_bytes,
            'capacity_is_profile_default': capacity_bytes == profile_capacity,
            'within_current_declared_pim_pool': within_current_pool,
            'requires_pim_pool_reallocation': not within_current_pool,
            'physical_capacity_bytes_in_profile': physical_capacity,
            'within_profile_physical_capacity': within_physical_capacity,
            'physically_admissible_under_profile': within_current_pool,
            'host_reserve_validated': False,
            'physical_capacity_status': physical_status,
            'requires_new_physical_profile': capacity_bytes > physical_capacity,
            'resident_bytes': projection['resident_bytes'],
            'resident_gib': projection['resident_bytes'] / 1024**3,
            'candidate_matrices': projection['candidate_matrices'],
            'resident_matrices': projection['resident_matrices'],
            'warm_linear_speedup_scenario': linear_speedup,
            'cold_linear_speedup_scenario': projection['cold_linear_speedup_scenario'],
            'host_linear_cycles': projection['host_linear_cycles'],
            'warm_hybrid_linear_cycles': projection['warm_hybrid_linear_cycles'],
            'one_time_placement_cycles': projection['one_time_placement_cycles'],
            'cold_hybrid_linear_cycles': projection['cold_hybrid_linear_cycles'],
            'host_external_bytes': projection['host_external_bytes'],
            'warm_hybrid_external_bytes': projection['warm_hybrid_external_bytes'],
            'required_fraction_of_total_service_for_1_05x': required_fraction,
            'best_case_e2e_speedup_if_all_service_were_this_linear_subgraph': linear_speedup,
            'boundary': (
                'capacity and phase-model projection over observed decode Linear traces; '
                'capacity above the declared PIM pool requires reallocation and host-reserve '
                'validation; scenarios above total physical capacity require a new physical '
                'profile; not measured PIM latency and not an E2E speedup claim'),
        })
        if host_by_question:
            baseline_seconds = [
                float(row['service_seconds']) for row in host_by_question.values()
                if row.get('service_seconds') is not None]
            scenarios[-1]['host_reference_request_count'] = len(baseline_seconds)
            scenarios[-1]['host_reference_total_service_seconds'] = sum(baseline_seconds)
            scenarios[-1]['required_total_savings_seconds_for_1_05x'] = (
                sum(baseline_seconds) * (1.0 - 1.0 / 1.05))
        else:
            scenarios[-1]['host_reference_request_count'] = None
            scenarios[-1]['host_reference_total_service_seconds'] = None
            scenarios[-1]['required_total_savings_seconds_for_1_05x'] = None
    return {
        'schema_version': 'decode-pim-capacity-sensitivity-v1',
        'profile_name': profile['name'],
        'profile_calibration_status': profile['calibration_status'],
        'profile_declared_capacity_bytes': profile_capacity,
        'profile_declared_capacity_gib': profile_capacity / 1024**3,
        'profile_physical_capacity_bytes': physical_capacity,
        'profile_physical_capacity_gib': physical_capacity / 1024**3,
        'host_runtime_reserve_bytes_in_profile': profile.get('host_runtime_reserve_bytes'),
        'observed_trace_count': trace_count,
        'merged_operator_count': len(operators),
        'decode_single_row_candidate_count': sum(
            item['phase'] == 'decode' and item['rows'] == 1 for item in operators),
        'capacity_scenarios': scenarios,
        'interpretation': {
            'capacity_is_simulator_scenario': True,
            'capacity_can_be_overridden': True,
            'larger_capacity_requires_hardware_or_architecture_justification': True,
            'physical_capacity_is_not_pim_pool_capacity': True,
            'host_runtime_reserve_is_unmodeled': True,
            'capacity_above_current_pool_is_not_hardware_admissible_without_reallocation': True,
            'capacity_above_profile_physical_capacity_is_hypothetical': True,
            'e2e_measurement_status': 'not_measured',
            'quality_status': 'not_rechecked',
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--profiled-run', type=Path, required=True)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--host-run', type=Path)
    parser.add_argument('--capacities-gib', type=float, nargs='+', default=[2, 4])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('fresh output required')
    run = json.loads(args.profiled_run.read_text())
    profile = json.loads(args.profile.read_text())
    host_run = json.loads(args.host_run.read_text()) if args.host_run else None
    result = project_capacity_sensitivity(run, profile, args.capacities_gib, host_run)
    result['profiled_run_sha256'] = hashlib.sha256(args.profiled_run.read_bytes()).hexdigest()
    result['profile_sha256'] = hashlib.sha256(args.profile.read_bytes()).hexdigest()
    if args.host_run:
        result['host_run_sha256'] = hashlib.sha256(args.host_run.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result['capacity_scenarios'], ensure_ascii=False))


if __name__ == '__main__':
    main()
