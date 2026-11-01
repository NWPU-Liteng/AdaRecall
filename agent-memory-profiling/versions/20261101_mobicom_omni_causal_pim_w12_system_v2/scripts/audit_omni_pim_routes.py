"""Rebuild every exported route, including its unknowns, from frozen producers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from compile_omni_pim_routes import ROOT, SOURCE_FILES, load_requests
from audit_omni_smoke_closure import digest
from pim_backend.execution_route import compile_route
from pim_backend.lpddr_bank_pim import BankPimConfig


def audit_routes(route_root, audit_path, execution_root):
    summary = json.loads((route_root / 'summary.json').read_text())
    routes_path = route_root / 'routes.json'
    if summary.get('complete') is not True or digest(routes_path) != summary['routes_sha256']:
        raise ValueError('complete hash-bound route export required')
    routes = json.loads(routes_path.read_text())
    if set(routes['source_files_sha256']) != set(SOURCE_FILES):
        raise ValueError('complete compiler/test/cost source snapshot required')
    for name, expected in routes['source_files_sha256'].items():
        if digest(route_root / 'sources' / name) != expected or digest(ROOT / name) != expected:
            raise ValueError('source snapshot or rebuilding code changed')
    if digest(route_root / 'profile.json') != routes['profile_sha256']:
        raise ValueError('frozen hardware profile changed')
    audit, requests, manifest = load_requests(audit_path, execution_root)
    if (manifest != routes['producer_files_sha256']
            or digest(audit_path) != routes['producer_audit_sha256']
            or audit['route_reservations'] != routes['serving_state_reservations']):
        raise ValueError('serving producers, cohort or reservations changed')
    profile = json.loads((route_root / 'profile.json').read_text())
    rebuilt = compile_route(requests, BankPimConfig(**profile['config']),
                            reserved_bytes=audit['route_reservations']['reserved_bytes'])
    expected_schedules = rebuilt.pop('schedules')
    for key, value in rebuilt.items():
        if routes.get(key) != value:
            raise ValueError('compiled route/cost/boundary differs: ' + key)
    paths = set()
    for schedule in expected_schedules:
        name = 'user_' + schedule['person_id'] + '/' + schedule['question_id'] + '.route.json'
        path = route_root / name
        if (digest(path) != routes['schedule_files_sha256'][name]
                or json.loads(path.read_text()) != schedule):
            raise ValueError('exported schedule differs from actual ordered execution graph')
        paths.add(name)
    if (set(routes['schedule_files_sha256']) != paths
            or {str(path.relative_to(route_root)) for path in route_root.glob('user_*/*.route.json')} != paths):
        raise ValueError('missing, duplicated or extra request schedule')
    for key, value in summary.items():
        if key not in {'routes_sha256', 'complete'} and routes.get(key) != value:
            raise ValueError('summary differs from full export: ' + key)
    return {'schema_version': 'omni-execution-pim-route-audit-v1', 'complete': True,
        'route_compilation_integrity': 'pass', 'full_export_recompiled_equal': True,
        'requests': rebuilt['requests'], 'observed_events': rebuilt['observed_events'],
        'modeled_linear_calls': rebuilt['modeled_linear_calls'],
        'unknown_duration_events': rebuilt['unknown_duration_events'],
        'routes_sha256': digest(routes_path), 'summary_sha256': digest(route_root / 'summary.json'),
        'audit_script_sha256': digest(Path(__file__)), 'evaluation_label_files_read': [],
        'actual_pim_routing_executed': False, 'new_quality_evaluated': False,
        'end_to_end_speedup': None, 'publication_ready': False,
        'boundary': 'route/cost/provenance integrity only, not numerical PIM execution, task success or full-system scheduling qualification'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--route-root', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--execution-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('fresh independent route-audit output required')
    result = audit_routes(args.route_root, args.audit, args.execution_root)
    with args.output.open('x') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
