"""Close a whole-cohort opaque-boundary replay with explicitly scoped timings."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil

from audit_omni_execution_cohort import analyze_graph
from audit_omni_smoke_closure import digest
from omni_latency_accounting import diagnostic_replay_summary, full_service_summary


ROOT = Path(__file__).resolve().parents[1]
SOURCES = ('scripts/summarize_omni_opaque_closure.py', 'scripts/omni_latency_accounting.py',
           'tests/test_omni_latency_accounting.py')


def summarize(run_root, route_root):
    hashes = {}

    def read(path):
        hashes[str(path.resolve())] = digest(path)
        return json.loads(path.read_text())

    plan = read(run_root / 'execution_plan.json')
    cohort = read(run_root / 'cohort_audit.json')
    routes = read(route_root / 'routes.json')
    route_audit = read(route_root / 'closure_audit.json')
    if (not cohort['complete'] or route_audit['route_compilation_integrity'] != 'pass'
            or not route_audit['full_export_recompiled_equal']
            or route_audit['routes_sha256'] != digest(route_root / 'routes.json')
            or routes['producer_audit_sha256'] != digest(run_root / 'cohort_audit.json')
            or cohort['execution_plan_sha256'] != digest(run_root / 'execution_plan.json')):
        raise ValueError('complete hash-bound cohort and route audit required')
    for name, expected in routes['schedule_files_sha256'].items():
        path = route_root / name
        if digest(path) != expected:
            raise ValueError('exported request schedule changed')
        hashes[str(path.resolve())] = expected
    serving_runs, replay_results = [], []
    region_counts, implementations = Counter(), Counter()
    total_events = total_edges = 0
    for job, user in zip(plan['jobs'], cohort['users']):
        if job['person_id'] != user['person_id']:
            raise ValueError('registered user order differs')
        smoke_path = Path(job['smoke'])
        if digest(smoke_path) != job['smoke_sha256']:
            raise ValueError('original full-service producer changed')
        smoke = read(smoke_path)
        result_path = run_root / ('user_' + job['person_id']) / 'result.json'
        result = read(result_path)
        expected_result_hashes = {str((ROOT / name).resolve()): value
                                  for name, value in cohort['user_result_sha256'].items()}
        if (digest(result_path) != expected_result_hashes[str(result_path.resolve())]
                or not result['complete'] or result['evaluation_label_files_read'] != []
                or [row['question_id'] for row in result['rows']] != job['question_ids']
                or [row['question_id'] for row in smoke['rows']] != job['question_ids']):
            raise ValueError('all fixed full-service and diagnostic rows required')
        for row, audited, original in zip(result['rows'], user['rows'], smoke['rows']):
            if (row['question_id'] != audited['question_id']
                    or row['answer_token_ids'] != original['answer_token_ids']
                    or row['stop_reason'] != original['stop_reason']):
                raise ValueError('full-cohort tokens and stopping must remain unchanged')
            path = Path(row['graph'])
            if digest(path) != row['graph_sha256'] or row['graph_sha256'] != audited['graph_sha256']:
                raise ValueError('opaque graph producer changed')
            graph = read(path)
            if analyze_graph(graph) != audited['graph_analysis']:
                raise ValueError('opaque boundary analysis changed')
            regions = graph['opaque_regions']
            if (row['opaque_callable_regions'] != len(regions) or not regions
                    or {region['name'].split('.')[-1] for region in regions} != {
                        'causal_conv1d_update', 'recurrent_gated_delta_rule'}):
                raise ValueError('required causal-convolution and recurrent boundaries missing')
            region_counts.update(region['name'].split('.')[-1] for region in regions)
            implementations.update(region['implementation'] for region in regions)
            total_events += len(graph['events'])
            total_edges += sum(len(event['dependencies']) for event in graph['events'])
        serving_runs.append(smoke)
        replay_results.append(result)
    count = sum(len(result['rows']) for result in replay_results)
    if (count != cohort['requests'] or count != cohort['complete_greedy_token_matches']
            or count != cohort['stop_reason_matches'] or count != routes['requests']
            or count != route_audit['requests'] or total_events != routes['observed_events']):
        raise ValueError('entire fixed denominator and graph event count must match')
    service = full_service_summary(serving_runs)
    replay = diagnostic_replay_summary(replay_results)
    return {'schema_version': 'omni-opaque-execution-closure-v1', 'complete': True,
        'requests': count, 'complete_greedy_token_matches': count, 'stop_reason_matches': count,
        'first_decode_events_including_completion_markers': total_events,
        'dependency_edges': total_edges, 'inclusive_completion_markers': sum(region_counts.values()),
        'regions_by_callable': dict(region_counts), 'implementations': dict(implementations),
        'original_full_service_baseline': service, 'current_diagnostic_replay': replay,
        'measured_server_end_to_end_speedup': None, 'modeled_lpddr_end_to_end_speedup': None,
        'end_to_end_speedup': None,
        'e2e_missing_gates': ['optimized complete-serving execution under matched measurement contract',
            'paired full-cohort complete-output and EOS equality',
            'LPDDR-comparable costs for vision/prefill/retrieval/opaque residuals and shared resources',
            'all decode steps and cold/warm placement lifecycle, not first-step extrapolation'],
        'pim_known_linear_subgraph_only': {key: routes[key] for key in (
            'resident_matrices', 'resident_bytes', 'shared_reserved_bytes',
            'warm_known_linear_speedup_lower_scenario',
            'cold_first_step_known_linear_speedup_lower_scenario')},
        'actual_pim_routing_executed': False, 'cross_request_reader_state_restored': False,
        'evaluation_label_files_read': [], 'new_quality_evaluated': False,
        'main_experiment_closed': False, 'publication_ready': False,
        'producer_sha256': hashes,
        'source_files_sha256': {name: digest(ROOT / name) for name in SOURCES},
        'boundary': 'first-decode callable completion coverage and full frozen output qualification; completion markers are inclusive accounting nodes, not additional device kernels or traffic; original full-service and current diagnostic seconds are different scopes, never a speedup pair; no measured PIM or modeled E2E gain'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', required=True, type=Path)
    parser.add_argument('--route-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    snapshot = args.output.with_suffix('.sources')
    if args.output.exists() or snapshot.exists():
        raise ValueError('fresh closure summary and source snapshot required')
    result = summarize(args.run_root, args.route_root)
    for name in SOURCES:
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    with args.output.open('x') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps({key: result[key] for key in ('requests', 'complete_greedy_token_matches',
        'stop_reason_matches', 'first_decode_events_including_completion_markers',
        'dependency_edges', 'inclusive_completion_markers', 'regions_by_callable',
        'measured_server_end_to_end_speedup', 'modeled_lpddr_end_to_end_speedup')}, indent=2))


if __name__ == '__main__':
    main()
