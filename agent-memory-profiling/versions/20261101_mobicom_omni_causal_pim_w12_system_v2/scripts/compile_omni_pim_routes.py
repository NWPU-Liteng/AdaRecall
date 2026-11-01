"""Compile, validate and snapshot routes without running or evaluating a Reader."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from audit_omni_execution_cohort import analyze_graph
from audit_omni_smoke_closure import digest
from pim_backend.execution_route import RouteInput, compile_route
from pim_backend.lpddr_bank_pim import BankPimConfig


SOURCE_FILES = ('scripts/compile_omni_pim_routes.py', 'scripts/audit_omni_pim_routes.py', 'scripts/audit_omni_execution_cohort.py',
    'scripts/audit_omni_smoke_closure.py', 'scripts/omni_prefix_opportunities.py',
    'pim_backend/__init__.py', 'pim_backend/execution_route.py', 'pim_backend/decode_gemv.py',
    'pim_backend/lpddr_bank_pim.py', 'pim_backend/lpddr_pim.py', 'pim_backend/agent_memory_pim.py',
    'tests/test_execution_route.py', 'tests/test_omni_pim_route_inputs.py',
    'tests/test_omni_opaque_regions.py', 'tests/test_omni_opaque_schedule.py', 'scripts/omni_execution_graph.py')
SOURCE_FILES = SOURCE_FILES + ('scripts/omni_latency_accounting.py',
    'scripts/summarize_omni_opaque_closure.py', 'tests/test_omni_latency_accounting.py')


def load_requests(audit_path, execution_root):
    execution_root = execution_root.resolve()
    audit = json.loads(audit_path.read_text())
    manifest = {}

    def checked(path, expected):
        actual = digest(path)
        if actual != expected:
            raise ValueError('producer hash mismatch: ' + str(path))
        manifest[str(path)] = actual
        return json.loads(path.read_text())

    if (not audit.get('complete') or not audit.get('users')
            or audit.get('evaluation_label_files_read') != []):
        raise ValueError('complete, label-isolated fixed-cohort audit required')
    plan = checked(execution_root / 'execution_plan.json', audit['execution_plan_sha256'])
    status = checked(execution_root / 'execution_status.json', audit['execution_status_sha256'])
    if (not status['all_workers_completed'] or not status['all_workers_succeeded']
            or status['execution_plan_sha256'] != audit['execution_plan_sha256']
            or plan['evaluation_label_files_read'] != []
            or [str(job['person_id']) for job in plan['jobs']] !=
               [str(user['person_id']) for user in audit['users']]):
        raise ValueError('completed original execution plan required, in registered order')
    for name, expected in plan['source_sha256'].items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('unsafe producer source path')
        snapshot = execution_root / 'sources' / relative
        if digest(snapshot) != expected:
            raise ValueError('original execution source snapshot differs')
        manifest[str(snapshot)] = expected
    requests, reservations = [], []
    result_hashes = {str((ROOT / name).resolve()): expected
                     for name, expected in audit['user_result_sha256'].items()}
    for user, job in zip(audit['users'], plan['jobs']):
        person_id = str(user['person_id'])
        smoke = checked(Path(job['smoke']), job['smoke_sha256'])
        embedder_config = checked(Path(smoke['embedder']) / 'config.json', smoke['embedder_config_sha256'])
        embedding_budget = smoke['embedding_state_reuse']['capacity_bytes']
        visual_budget = (smoke.get('visual_state_reuse') or {}).get('capacity_bytes', 0)
        dense_bytes = smoke['embedding_windows'] * embedder_config['hidden_size'] * 4
        if (min(embedding_budget, visual_budget, dense_bytes) < 0
                or smoke['evaluation_label_files_read'] != []):
            raise ValueError('nonnegative predeclared serving-state reservations required')
        reservations.append({'person_id': person_id, 'declared_embedding_store_bytes': embedding_budget,
            'declared_visual_store_bytes': visual_budget, 'dense_fp32_index_bytes': dense_bytes,
            'total_bytes': embedding_budget + visual_budget + dense_bytes})
        result_path = execution_root / ('user_' + person_id) / 'result.json'
        if str(result_path) not in result_hashes:
            raise ValueError('user result absent from producer hash manifest')
        result = checked(result_path, result_hashes[str(result_path)])
        question_ids = [row['question_id'] for row in user['rows']]
        if (not result['complete'] or question_ids != job['question_ids']
                or question_ids != [row['question_id'] for row in result['rows']]
                or result['evaluation_label_files_read'] != []
                or result['execution_plan_sha256'] != audit['execution_plan_sha256']):
            raise ValueError('all fixed requests must remain in natural order')
        for row, produced in zip(user['rows'], result['rows']):
            if row['token_ids_equal'] is not True or row['stop_reason_equal'] is not True:
                raise ValueError('complete frozen output equality must pass for every request')
            if row['graph_sha256'] != produced['graph_sha256'] or row['input_sha256'] != produced['input_sha256']:
                raise ValueError('audit and producer request hashes differ')
            graph = checked(Path(produced['graph']), row['graph_sha256'])
            inputs = checked(Path(produced['input']), row['input_sha256'])
            if (inputs['question_id'] != row['question_id']
                    or graph['context_identity'] != inputs['input_identity']
                    or analyze_graph(graph) != row['graph_analysis']):
                raise ValueError('graph must bind the audited raw-image request and full analysis')
            requests.append(RouteInput(person_id, row['question_id'], graph))
    if (len(requests) != audit['requests'] or len(requests) != audit['traced_requests']
            or len(requests) != audit['complete_greedy_token_matches']
            or len(requests) != audit['stop_reason_matches']):
        raise ValueError('whole-cohort denominator or equality count differs')
    manifest[str(audit_path.resolve())] = digest(audit_path)
    audit['route_reservations'] = {'per_user_worker': reservations,
        'reserved_bytes': max(item['total_bytes'] for item in reservations),
        'policy': 'same placement fits every separately replayed user worker; full declared stores plus immutable prebuilt index, never future observed occupancy; workers are not concurrent tenants of one PIM pool'}
    return audit, requests, manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--execution-root', type=Path, required=True)
    parser.add_argument('--profile', type=Path, default=ROOT / 'pim_backend/mobile_lpddr5_bank_profile.json')
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise ValueError('fresh route directory required; frozen artifacts are never overwritten')
    audit, requests, manifest = load_requests(args.audit, args.execution_root)
    profile = json.loads(args.profile.read_text())
    compiled = compile_route(requests, BankPimConfig(**profile['config']),
                             reserved_bytes=audit['route_reservations']['reserved_bytes'])
    provenance = {'producer_audit_sha256': digest(args.audit), 'producer_files_sha256': manifest,
        'profile_name': profile['name'], 'profile_sha256': digest(args.profile),
        'profile_calibration_status': profile['calibration_status'],
        'inherited_frozen_replay_token_matches': audit['complete_greedy_token_matches'],
        'inherited_frozen_replay_stop_matches': audit['stop_reason_matches'],
        'evaluation_label_files_read': [],
        'serving_state_reservations': audit['route_reservations'],
        'source_files_sha256': {name: digest(ROOT / name) for name in SOURCE_FILES}}
    args.output_root.mkdir(parents=True, exist_ok=False)
    for name in SOURCE_FILES:
        destination = args.output_root / 'sources' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    shutil.copyfile(args.profile, args.output_root / 'profile.json')
    compiled.update(provenance)
    schedule_hashes = {}
    for schedule in compiled.pop('schedules'):
        name = 'user_' + schedule['person_id'] + '/' + schedule['question_id'] + '.route.json'
        path = args.output_root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(schedule, ensure_ascii=False, indent=2) + '\n')
        schedule_hashes[name] = digest(path)
    compiled['schedule_files_sha256'] = schedule_hashes
    output = args.output_root / 'routes.json'
    output.write_text(json.dumps(compiled, ensure_ascii=False, indent=2) + '\n')
    summary = {key: value for key, value in compiled.items() if key not in {
        'costs', 'placement', 'schedule_files_sha256', 'producer_files_sha256'}}
    summary['routes_sha256'] = digest(output)
    summary['complete'] = True
    (args.output_root / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: summary[key] for key in ('requests', 'observed_events',
        'modeled_linear_calls', 'opaque_events', 'unknown_duration_events', 'resident_matrices',
        'resident_bytes', 'warm_known_linear_speedup_lower_scenario',
        'cold_first_step_known_linear_speedup_lower_scenario', 'hypothetical_break_even_decode_steps',
        'end_to_end_speedup')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
