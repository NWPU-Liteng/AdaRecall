"""Bind full decode aggregate call ledgers to frozen first-step PIM routes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from omni_decode_lifecycle import lifecycle_for_user, normalized_service_sensitivity
from omni_latency_accounting import full_service_summary
from run_omni_registered_cohort import digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--route-root', type=Path, required=True)
    parser.add_argument('--cohort-plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('fresh lifecycle artifact required')
    route_path = args.route_root / 'routes.json'
    audit_path = args.route_root / 'closure_audit.json'
    routes = json.loads(route_path.read_text())
    audit = json.loads(audit_path.read_text())
    cohort = json.loads(args.cohort_plan.read_text())
    if (audit['route_compilation_integrity'] != 'pass' or not audit['full_export_recompiled_equal']
            or audit['routes_sha256'] != digest(route_path)):
        raise ValueError('independently audited frozen route required')
    provenance = {str(path): digest(path) for path in (route_path, audit_path, args.cohort_plan)}
    users, runs = [], []
    for job in cohort['jobs']:
        source = Path(job['output'])
        run = json.loads(source.read_text())
        schedules = []
        if job['question_ids'] != [row['question_id'] for row in run['rows']]:
            raise ValueError('full frozen per-user cohort required')
        provenance[str(source)] = digest(source)
        for question in job['question_ids']:
            name = 'user_' + job['person_id'] + '/' + question + '.route.json'
            path = args.route_root / name
            if digest(path) != routes['schedule_files_sha256'][name]:
                raise ValueError('route schedule changed')
            provenance[str(path)] = digest(path)
            schedules.append(json.loads(path.read_text()))
        users.append(lifecycle_for_user(run, schedules, routes['costs'], routes['placement'],
                                        routes['placement_cycle_range']))
        runs.append(run)
    if sum(user['requests'] for user in users) != routes['requests']:
        raise ValueError('full route cohort denominator required')
    service = full_service_summary(runs)
    host = sum(user['totals']['host_min_cycles'] for user in users)
    candidate = sum(user['persistent_stream_hybrid_max_cycles'] for user in users)
    ratio = host / candidate
    result = {'schema_version': 'omni-conditional-decode-lifecycle-v1', 'complete': True,
        'requests': sum(user['requests'] for user in users), 'per_user': users,
        'decode_invocations': sum(user['decode_invocations'] for user in users),
        'linear_calls': sum(user['linear_calls'] for user in users),
        'persistent_stream_known_linear_speedup_lower_scenario': ratio,
        'cohort_service_decomposition': service,
        'dimensionless_sensitivity_not_lpddr_e2e_prediction': [normalized_service_sensitivity(
            service['decode_fraction_of_total_service'], ratio, fraction)
            for fraction in (0., .25, .5, .75, 1.)],
        'full_system_unknown_costs_preserved': True,
        'placement_output_length_oracle': False, 'actual_pim_routing_executed': False,
        'full_decode_dag_traced': False, 'end_to_end_speedup': None,
        'modeled_lpddr_end_to_end_speedup': None, 'publication_ready': False,
        'inputs_sha256': provenance, 'evaluation_label_files_read': [],
        'producer_sha256': digest(Path(__file__)),
        'lifecycle_module_sha256': digest(Path(__file__).with_name('omni_decode_lifecycle.py')),
        'boundary': 'frozen placement independent of future output; observed output length used only for retrospective cost accounting; every user is an independent cold stream; no GPU seconds converted to LPDDR cycles'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps({key: result[key] for key in ('requests', 'decode_invocations', 'linear_calls',
        'persistent_stream_known_linear_speedup_lower_scenario', 'end_to_end_speedup')}))


if __name__ == '__main__':
    main()
