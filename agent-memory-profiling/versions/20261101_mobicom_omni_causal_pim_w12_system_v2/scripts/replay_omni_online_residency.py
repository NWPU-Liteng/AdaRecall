"""Retrospective dispatch replay with horizon-free, per-call weight admission."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from audit_omni_pim_routes import audit_routes
from audit_omni_smoke_closure import digest
from omni_decode_lifecycle import decode_calls, lifecycle_for_user
from pim_backend.online_residency import OnlineResidency


def replay_user(run, schedules, routes):
    lifecycle_for_user(run, schedules, routes['costs'], routes['placement'], routes['placement_cycle_range'])
    reserved = routes['serving_state_reservations']['reserved_bytes']
    gate = OnlineResidency(routes['costs'], routes['placement'],
        capacity_bytes=routes['capacity_bytes'], reserved_bytes=reserved)
    totals = Counter()
    requests, admissions = [], []
    call_index = 0
    for row, schedule in zip(run['rows'], schedules):
        request = Counter()
        events = [event for event in schedule['events'] if event['kind'] == 'modeled_linear']
        steps = decode_calls(row)
        for position in range(steps):
            for event in events:
                decision = gate.dispatch(event['parameter_identity'])
                call_index += 1
                for field in ('host_min_cycles', 'host_max_cycles', 'hybrid_min_cycles',
                              'hybrid_max_cycles', 'placement_min_cycles', 'placement_max_cycles',
                              'host_external_bytes', 'hybrid_external_bytes'):
                    request[field] += decision[field]
                request['pim_calls'] += decision['route'] == 'pim'
                request['placement_operations'] += decision['placement_now']
                if decision['placement_now']:
                    admissions.append({'question_id': row['question_id'],
                        'retrospective_decode_index': position,
                        'repeated_first_step_event_id': event['event_id'],
                        'observed_linear_call_index': call_index, **decision})
        if gate.resident_bytes + reserved > routes['capacity_bytes']:
            raise ValueError('admission and serving state exceed the common pool')
        totals.update(request)
        requests.append({'question_id': row['question_id'], 'decode_invocations': steps,
            'context_identity': schedule['context_identity'], 'totals': dict(request),
            'resident_weight_bytes_after': gate.resident_bytes})
    certificate = gate.certificate()
    hybrid = totals['hybrid_max_cycles'] + totals['placement_max_cycles']
    return {'person_id': run['rows'][0]['person_id'], 'requests': len(requests),
        'per_request': requests, 'admissions': admissions, 'totals': dict(totals),
        'known_linear_online_max_cycles_including_placement': hybrid,
        'known_linear_host_best_over_online_worst_scenario': totals['host_min_cycles'] / hybrid if hybrid else None,
        'resident_weight_bytes': gate.resident_bytes, 'reserved_state_bytes': reserved,
        'fixed_pool_certificate': certificate}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--route-root', type=Path, required=True)
    parser.add_argument('--execution-root', type=Path, required=True)
    parser.add_argument('--execution-audit', type=Path, required=True)
    parser.add_argument('--cohort-plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('fresh output required; historical evidence is immutable')
    audit_routes(args.route_root, args.execution_audit, args.execution_root)
    route_path = args.route_root / 'routes.json'
    routes = json.loads(route_path.read_text())
    plan_path = args.execution_root / 'execution_plan.json'
    plan = json.loads(plan_path.read_text())
    if digest(args.cohort_plan) != plan['cohort_plan_sha256']:
        raise ValueError('canonical serving cohort differs from the execution cohort')
    cohort = json.loads(args.cohort_plan.read_text())
    if [(job['person_id'], job['question_ids'], job['output']) for job in cohort['jobs']] != [
            (job['person_id'], job['question_ids'], job['smoke']) for job in plan['jobs']]:
        raise ValueError('original complete per-user request order required')
    inputs = {str(path): digest(path) for path in (route_path, plan_path, args.cohort_plan,
        args.execution_audit, args.route_root / 'closure_audit.json')}
    users = []
    for job in plan['jobs']:
        source = Path(job['smoke'])
        if digest(source) != job['smoke_sha256']:
            raise ValueError('serving call ledger changed')
        run = json.loads(source.read_text())
        inputs[str(source)] = digest(source)
        schedules = []
        bindings = []
        for question_id in job['question_ids']:
            name = 'user_' + job['person_id'] + '/' + question_id + '.route.json'
            path = args.route_root / name
            if digest(path) != routes['schedule_files_sha256'][name]:
                raise ValueError('frozen request route changed')
            inputs[str(path)] = digest(path)
            schedule = json.loads(path.read_text())
            operand_path = args.execution_root / ('user_' + job['person_id']) / (question_id + '.input.json')
            operands = json.loads(operand_path.read_text())
            if ('frozen_serving_input_contract' not in operands
                    or operands['input_identity'] != schedule['context_identity']):
                raise ValueError('exact current processor/producer contract required')
            inputs[str(operand_path)] = digest(operand_path)
            bindings.append({'question_id': question_id,
                'accepted_atom_ids': operands['accepted_atom_ids'],
                'selected_hyperedges': operands['selected_hyperedges'],
                'frozen_serving_input_contract': operands['frozen_serving_input_contract'],
                'reader_context_identity': operands['input_identity'],
                'input_file_sha256': digest(operand_path), 'route_file_sha256': digest(path)})
            schedules.append(schedule)
        result = replay_user(run, schedules, routes)
        result['cross_layer_request_bindings'] = bindings
        users.append(result)
    total = sum(user['known_linear_online_max_cycles_including_placement'] for user in users)
    host = sum(user['totals']['host_min_cycles'] for user in users)
    result = {'schema_version': 'omni-online-weight-admission-replay-v1', 'complete': True,
        'requests': sum(user['requests'] for user in users), 'per_user': users,
        'known_linear_host_best_over_online_worst_scenario': host / total,
        'fixed_pool_admission_certificate_pass': all(record['twice_oracle_bound_pass']
            for user in users for record in user['fixed_pool_certificate']['weights']),
        'policy': 'fixed capacity-certified candidate pool from first observed decode; admit on the current observed primitive when accumulated max-envelope host-minus-PIM rental reaches placement max cycles; independent cold user streams, no evictions or future horizon in dispatch',
        'novelty_claim': False, 'actual_pim_routing_executed': False,
        'reader_dispatch_integration_executed': False, 'full_decode_dag_traced': False,
        'end_to_end_speedup': None, 'modeled_lpddr_end_to_end_speedup': None,
        'all_host_non_regression_guarantee': False, 'new_quality_evaluated': False,
        'evaluation_label_files_read': [], 'inputs_sha256': inputs,
        'source_files_sha256': {name: digest(ROOT / name) for name in (
            'scripts/replay_omni_online_residency.py', 'pim_backend/online_residency.py',
            'scripts/omni_decode_lifecycle.py', 'tests/test_online_residency.py',
            'tests/test_omni_online_residency_replay.py')},
        'boundary': 'retrospective replay repeats the first-step Linear order for observed decode counts certified only in aggregate; unknown operators, growing attention and later DAG dependencies remain unknown; two-times certificate applies only to incremental fixed-pool max-envelope surrogate, not robust host-best/PIM-worst ratio or full system; established online rent-or-buy tool, not claimed as a novel algorithm'}
    if result['requests'] != routes['requests']:
        raise ValueError('all requests must remain in denominator')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps({key: result[key] for key in ('requests',
        'known_linear_host_best_over_online_worst_scenario', 'fixed_pool_admission_certificate_pass',
        'actual_pim_routing_executed', 'end_to_end_speedup')}, indent=2))


if __name__ == '__main__':
    main()
