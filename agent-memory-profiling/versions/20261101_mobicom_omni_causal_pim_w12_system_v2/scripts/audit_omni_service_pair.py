"""Audit complete serving pairs against the original registered development stream."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audit_omni_smoke_closure import empirical_latency, verify_source_snapshot
from omni_latency_accounting import _semantic_evidence_plan, full_service_summary, measured_e2e_pair
from run_omni_registered_cohort import digest


def function_source_identity(run, registration, function_name):
    snapshot = Path(registration['source_snapshot_directory'])
    source_name = 'scripts/run_omni_raw_image_smoke.py'
    if source_name not in run['source_files_sha256']:
        return None
    source = (snapshot / source_name).read_text()
    tree = ast.parse(source)
    matches = [node for node in tree.body
               if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name]
    if len(matches) != 1:
        raise ValueError(f'expected exactly one {function_name} implementation in frozen serving source')
    return ast.dump(matches[0], annotate_fields=True, include_attributes=False)


def embedding_namespace_equivalence(before, after, before_registration, after_registration):
    identities = [run.get('embedding_execution_identity') for run in (before, after)]
    for field in ('embedding_producer_policy', 'embedder_identity_files_sha256', 'runtime_environment'):
        if before.get(field) != after.get(field):
            raise ValueError('paired immutable embedding execution environment changed')
    if identities[0] == identities[1]:
        return {'namespace_equal': True, 'equivalent': True}
    contracts = []
    for run, registration in ((before, before_registration), (after, after_registration)):
        if not verify_source_snapshot(run, registration):
            raise ValueError('embedding namespace diagnosis requires intact frozen source snapshots')
        source_name = 'scripts/run_omni_raw_image_smoke.py'
        source_path = Path(registration['source_snapshot_directory']) / source_name
        source = source_path.read_text()
        tree = ast.parse(source)
        matches = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'token_vector']
        if len(matches) != 1:
            raise ValueError('one frozen token_vector implementation required')
        function_source = ast.get_source_segment(source, matches[0])
        function_digest = hashlib.sha256((function_source + '\n').encode()).hexdigest()
        payload = {'files': run['embedder_identity_files_sha256'], 'dtype': 'torch.bfloat16',
                   'attention': 'sdpa', 'pooling': 'last_nonpad_token_then_fp32_l2_normalize'}
        valid_identities = {}
        for schema, source_digest in (('legacy_whole_runner', run['source_files_sha256'][source_name]),
                                      ('encoder_function', function_digest)):
            value = dict(payload, token_vector_source_sha256=source_digest)
            valid_identities[hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()] = schema
        identity = run.get('embedding_execution_identity')
        if identity not in valid_identities:
            raise ValueError('unrecognized embedding execution namespace; cannot substitute identity')
        contracts.append({'schema': valid_identities[identity],
            'encoder_ast': function_source_identity(run, registration, 'token_vector'),
            'producer_module': run['source_files_sha256']['scripts/omni_embedding_producer.py']})
    if any(contracts[0][field] != contracts[1][field] for field in ('encoder_ast', 'producer_module')):
        raise ValueError('embedding producer implementation differs; namespaces are not equivalent')
    return {'namespace_equal': False, 'equivalent': True,
            'schemas': [contract['schema'] for contract in contracts],
            'boundary': 'only verified whole-runner versus encoder-function namespace changes; model/runtime, encoder AST and producer source identical; no cross-run cache transfer permitted or tensor tolerance granted'}


def row_differences(before, after, *, equivalent_producer_namespace=False):
    fields = ('question_id', 'question_sha256', 'images', 'answer_token_ids', 'stop_reason',
              'input_tokens', 'visual_patches', 'answer_sha256')
    differences = [field for field in fields if before[field] != after[field]]
    for field in ('input_token_ids_sha256', 'processor_inputs_sha256'):
        if before.get(field) != after.get(field):
            differences.append(field)
    producers = []
    producer_identities = []
    for row in (before, after):
        records = row.get('embedding_producers')
        producers.append(None if records is None else [
            (record['edge_ids'], record['output_sha256']) for record in records])
        producer_identities.append(None if records is None else [
            record['producer_sha256'] for record in records])
    if producers[0] != producers[1]:
        differences.append('embedding_producer_outputs')
    if producer_identities[0] != producer_identities[1] and not equivalent_producer_namespace:
        differences.append('embedding_producer_identity')
    if _semantic_evidence_plan(before['hypergraph_evidence_plan']) != _semantic_evidence_plan(
            after['hypergraph_evidence_plan']):
        differences.append('semantic_evidence_plan')
    return differences


def functional_backend_summary(run):
    if run.get('reader_functional_backend', 'host_primary') == 'host_primary':
        return None
    if (run.get('reader_functional_backend') != 'pimnast_bank_layout_reference'
            or run.get('reader_quantization') != 'w8a8'
            or run.get('physical_pim_execution') is not False
            or run.get('performance_comparable') is not False):
        raise ValueError('explicit non-performance INT8 functional backend required')
    from pim_backend.pimnast_functional import frozen_route
    startup = run['pim_functional_startup']
    path = Path(startup['projection_path'])
    if digest(path) != startup['projection_sha256']:
        raise ValueError('functional PIM route changed')
    candidates, capacity = frozen_route(json.loads(path.read_text()), str(run['rows'][0]['person_id']))
    names = {name for name, item in candidates.items() if item['model_route'] == 'pim'}
    resident_bytes = sum(candidates[name]['candidate_resident_bytes'] for name in names)
    summary = run['pim_functional_dispatch']
    if (not names or resident_bytes > capacity or capacity != startup['declared_weight_pool_bytes']
            or resident_bytes != startup['layout_resident_bytes']
            or resident_bytes != summary['resident_layout_bytes']
            or summary['selected_layouts'] != len(names)
            or set(summary['calls_per_module']) != names):
        raise ValueError('complete functional layout residency accounting required')
    total_steps = 0
    requests = []
    for row in run['rows']:
        steps = len(row['answer_token_ids']) - 1
        calls = row.get('pim_functional_calls')
        if steps < 0 or not isinstance(calls, dict) or set(calls) != names:
            raise ValueError('one complete functional call ledger per request required')
        if any(type(count) is not int or count != steps for count in calls.values()):
            raise ValueError('functional replacement calls do not match actual decode tokens')
        total_steps += steps
        requests.append({'question_id': row['question_id'], 'decode_steps': steps,
                         'bank_layout_replacement_calls': sum(calls.values())})
    if (any(count != total_steps for count in summary['calls_per_module'].values())
            or summary['unobserved_selected_modules'] != ([] if total_steps else sorted(names))):
        raise ValueError('functional aggregate and per-request replacement counts differ')
    return {'backend': 'pimnast_bank_layout_reference', 'selected_layouts': len(names),
            'bank_layout_replacement_calls': len(names) * total_steps,
            'resident_layout_bytes': resident_bytes,
            'weight_pool_capacity_bytes': capacity,
            'prefill_or_multirow_host_fallback_calls': summary['prefill_or_multirow_host_fallback_calls'],
            'functional_workspace_bound_bytes': summary['functional_max_workspace_bound_bytes'],
            'functional_extra_cuda_bytes': startup['functional_extra_cuda_bytes'],
            'layout_construction_seconds_server': startup['layout_seconds_server'],
            'requests': requests, 'physical_pim_execution': False,
            'full_device_capacity_qualified': False,
            'boundary': 'functional bank-layout calls only; Host primary duplicate and emulator workspace must not be hidden; no hardware speedup or full PIM-buffer qualification'}


def validate_deferred_finite_validation(run):
    summary = run.get('reader_finite_validation')
    if summary is None:
        return None
    if run.get('reader_quantization') != 'w8a8':
        raise ValueError('deferred finite-check validation requires W8A8')
    required = ('decode_forward_calls', 'finite_predicates_checked', 'host_confirmations',
                'maximum_predicate_payload_bytes')
    if any(type(summary.get(field)) is not int or summary[field] < 0 for field in required):
        raise ValueError('finite-check aggregate counters must be nonnegative integers')
    if summary.get('arithmetic_changed') is not False or summary.get('checks_removed') is not False:
        raise ValueError('deferred finite-check contract changed arithmetic or removed checks')
    requests = []
    for row in run['rows']:
        if not row['answer_token_ids'] or row['stop_reason'] not in {'eos', 'max_new_tokens'}:
            raise ValueError('complete nonempty answer and supported stop reason required')
        requests.append({'question_id': row['question_id'],
                         'decode_steps': len(row['answer_token_ids']) - 1})
    steps = sum(row['decode_steps'] for row in requests)
    modules = run.get('reader_integer_primary', {}).get('modules')
    if type(modules) is not int or modules < 1:
        raise ValueError('integer Reader module count required for finite-check accounting')
    operators = [item for item in run.get('operator_trace', {}).get('operators', [])
                 if item['phase'] == 'decode']
    if (len(operators) != (modules if steps else 0)
            or len({item['name'] for item in operators}) != len(operators)
            or any(item['rows'] != 1 or item['element_bytes'] != 1
                   or type(item['calls']) is not int or item['calls'] != steps for item in operators)):
        raise ValueError('complete single-row INT8 decode operator trace required')
    expected_predicates = steps * modules * 2
    expected_payload = modules * 2 if steps else 0
    if (summary['decode_forward_calls'] != steps
            or summary['finite_predicates_checked'] != expected_predicates
            or summary['host_confirmations'] != steps
            or summary['maximum_predicate_payload_bytes'] != expected_payload):
        raise ValueError('finite-check aggregate does not match complete decode operator ledger')
    return {'decode_forward_calls': steps, 'finite_predicates_checked': expected_predicates,
            'host_confirmations': steps, 'maximum_predicate_payload_bytes': expected_payload,
            'modules': modules, 'requests': requests,
            'per_request_accounting': 'decode steps from actual answer tokens; predicate execution measured only as a run aggregate, checked against complete per-module operator calls',
            'boundary': summary.get('boundary')}


def screening_summary(reference_runs, candidate_runs, expected_by_user, namespace_contracts=None):
    references, candidates, mismatches, users = [], [], [], []
    for person_id, expected in expected_by_user.items():
        before, after = reference_runs[person_id], candidate_runs[person_id]
        if len(expected) != len(set(expected)):
            raise ValueError('unique registered requests required')
        for run in (before, after):
            if (not run.get('complete') or run.get('evaluation_label_files_read') != []
                    or [row['question_id'] for row in run['rows']] != expected):
                raise ValueError('complete registered label-isolated streams required')
            for row in run['rows']:
                total = row['service_seconds']
                if not math.isfinite(total) or total <= 0:
                    raise ValueError('positive finite service time required')
                parts = row['service_stage_ledger']['disjoint_seconds']
                if (any(not math.isfinite(value) or value < 0 for value in parts.values())
                        or not math.isclose(sum(parts.values()), total, abs_tol=1e-6)):
                    raise ValueError('disjoint complete service ledger required')
        for earlier, later in zip(before['rows'], after['rows']):
            fields = row_differences(earlier, later, equivalent_producer_namespace=bool(
                namespace_contracts and namespace_contracts[person_id]['equivalent']))
            if fields:
                mismatches.append({'person_id': person_id, 'question_id': earlier['question_id'],
                                   'fields': fields})
        references.extend(before['rows'])
        candidates.extend(after['rows'])
        users.append({'person_id': person_id,
                      'reference_functional_backend': functional_backend_summary(before),
                      'candidate_functional_backend': functional_backend_summary(after),
                      'reference_finite_validation': validate_deferred_finite_validation(before),
                      'candidate_finite_validation': validate_deferred_finite_validation(after),
                      'embedding_namespace_contract': namespace_contracts.get(person_id) if namespace_contracts else None,
                      'reference_construction_seconds': before['construction_seconds'],
                      'candidate_construction_seconds': after['construction_seconds'],
                      'reference_model_load_seconds': before['model_load_seconds'],
                      'candidate_model_load_seconds': after['model_load_seconds'],
                      'reference_embedding_store': before['embedding_state_reuse'],
                      'candidate_embedding_store': after['embedding_state_reuse']})

    def summarize(rows):
        total = sum(row['service_seconds'] for row in rows)
        edge = sum(row['edge_embedding_seconds'] for row in rows)
        decode = sum(row['decode_seconds'] for row in rows)
        vision = sum(row['service_stage_ledger']['nested_seconds']['vision_in_prefill_seconds'] for row in rows)
        if any(value < 0 or value >= total for value in (edge, decode, vision)):
            raise ValueError('nested stage lies outside complete service')
        timed_rows = [row for row in rows if row.get('ttft_seconds') is not None
                      and row.get('ttlt_seconds') is not None]
        for row in rows:
            values = (row.get('ttft_seconds'), row.get('ttlt_seconds'))
            if (values[0] is None) != (values[1] is None):
                raise ValueError('both first and last token boundaries required together')
            if values[0] is not None and (any(not math.isfinite(value) for value in values)
                    or not 0 <= values[0] <= values[1] <= row['service_seconds']):
                raise ValueError('token boundaries must lie within complete service')
        return {
            'service_seconds': empirical_latency([row['service_seconds'] for row in rows]),
            'token_boundary_requests': len(timed_rows),
            'token_boundary_missing_question_ids': [row['question_id'] for row in rows
                                                    if row.get('ttft_seconds') is None],
            'ttft_seconds': empirical_latency([row['ttft_seconds'] for row in timed_rows])
                            if len(timed_rows) == len(rows) else None,
            'ttlt_seconds': empirical_latency([row['ttlt_seconds'] for row in timed_rows])
                            if len(timed_rows) == len(rows) else None,
            'token_boundary': 'request start to synchronized first/last token decision; excludes network/streaming; absent historical timings are not imputed from prefill completion',
            'ttft_boundary_seconds': empirical_latency([
                sum(row['service_stage_ledger']['disjoint_seconds'][field] for field in
                    ('evidence_seconds', 'input_preparation_seconds', 'prefill_seconds')) for row in rows]),
            'ttft_boundary': 'prefill completion; excludes first argmax and streaming/transport; not measured user-visible TTFT',
            'stages_seconds': {field: sum(row[field] for row in rows) for field in
                              ('query_retrieval_seconds', 'edge_embedding_seconds',
                               'evidence_planning_seconds', 'prefill_seconds', 'decode_seconds')},
            'nested_vision_seconds': vision,
            'embedding_stage_fraction': edge / total,
            'zero_cost_remove_entire_embedding_stage_speedup_upper_bound': total / (total - edge),
            'zero_cost_remove_entire_decode_stage_speedup_upper_bound': total / (total - decode),
            'zero_cost_remove_entire_vision_stage_speedup_upper_bound': total / (total - vision),
            'output_tokens': sum(len(row['answer_token_ids']) for row in rows),
            'truncated_question_ids': [row['question_id'] for row in rows if row['stop_reason'] != 'eos'],
            'rows': [{'question_id': row['question_id'], 'service_seconds': row['service_seconds'],
                      'ttft_seconds': row.get('ttft_seconds'), 'ttlt_seconds': row.get('ttlt_seconds'),
                      'edge_embedding_seconds': row['edge_embedding_seconds'],
                      'decode_seconds': row['decode_seconds'], 'stop_reason': row['stop_reason']} for row in rows],
        }

    reference, candidate = summarize(references), summarize(candidates)
    emulated = any(user['reference_functional_backend'] or user['candidate_functional_backend'] for user in users)
    return {'complete': True, 'requests_per_policy': len(references), 'pair_mismatches': mismatches,
            'execution_integrity': 'pass' if not mismatches else 'failed_pair_equivalence_gate',
            'observed_service_operands_and_answers_equal': all(
                mismatch['fields'] == ['embedding_producer_identity'] for mismatch in mismatches),
            'quality_inherited_from_reference': not mismatches,
            'reference': reference, 'candidate': candidate, 'per_user': users,
            'screening_service_ratio': (reference['service_seconds']['sum_seconds'] /
                                       candidate['service_seconds']['sum_seconds']) if not emulated else None,
            'functional_emulator_server_costs_not_hardware_latency': emulated,
            'measured_causal_end_to_end_speedup': None,
            'actual_pim_end_to_end_speedup': None,
            'boundary': 'shared-host screening only; upper bounds remove complete stages at zero cost, not attainable predictions; quality requires consistent judge'}


def audit_pair(reference_root, candidate_root, frozen_plan_path):
    frozen = json.loads(frozen_plan_path.read_text())
    references, candidates, pairs, mismatches, historical_differences = [], [], [], [], []
    inputs = {str(frozen_plan_path): digest(frozen_plan_path)}
    reference_plan = json.loads((reference_root / 'cohort_plan.json').read_text())
    candidate_plan = json.loads((candidate_root / 'cohort_plan.json').read_text())
    for field in ('prepared_audit_sha256', 'serving_manifest_sha256', 'image_archive_sha256',
                  'max_new_tokens', 'image_edges_per_session', 'image_novelty_weight'):
        frozen_value = frozen.get(field, 0.) if field == 'image_novelty_weight' else frozen[field]
        if reference_plan[field] != candidate_plan[field] or candidate_plan[field] != frozen_value:
            raise ValueError('registered serving inputs or evidence policy changed')
    cohort = [(job['person_id'], job['question_ids']) for job in frozen['jobs']]
    for root, plan in ((reference_root, reference_plan), (candidate_root, candidate_plan)):
        if [(job['person_id'], job['question_ids']) for job in plan['jobs']] != cohort:
            raise ValueError('whole frozen cohort and per-user order required')
        status_path = root / 'worker_status.json'
        status = json.loads(status_path.read_text())
        if not status['sources_unchanged'] or not status['all_workers_exited_zero']:
            raise ValueError('worker completion and source-integrity gates failed')
        for path in (root / 'cohort_plan.json', status_path):
            inputs[str(path)] = digest(path)
    for original, before_job, after_job in zip(frozen['jobs'], reference_plan['jobs'], candidate_plan['jobs']):
        if before_job['gpu'] != after_job['gpu']:
            raise ValueError('each paired user must run on the same physical GPU index')
        files = [Path(job['output']) for job in (original, before_job, after_job)]
        anchor, before, after = [json.loads(path.read_text()) for path in files]
        for path in files:
            inputs[str(path)] = digest(path)
        for path, run, plan in ((files[1], before, reference_plan), (files[2], after, candidate_plan)):
            registration_path = path.with_suffix('.registration.json')
            registration = json.loads(registration_path.read_text())
            inputs[str(registration_path)] = digest(registration_path)
            if (registration['source_files_sha256'] != run['source_files_sha256']
                    or not verify_source_snapshot(run, registration)
                    or any(plan['source_files_sha256'].get(name) != value
                           for name, value in run['source_files_sha256'].items())
                    or registration['question_ids'] != original['question_ids']):
                raise ValueError('frozen serving source snapshot/registration changed')
            if [row['question_id'] for row in run['rows']] != original['question_ids']:
                raise ValueError('incomplete registered user stream')
            for field in ('reader_config_sha256', 'embedder_config_sha256', 'system_prompt_sha256',
                          'candidate_sessions', 'max_images', 'session_text_tokens', 'total_text_tokens',
                          'image_max_pixels', 'max_new_tokens', 'archive_sha256', 'prepared_audit_sha256'):
                if run[field] != anchor[field]:
                    raise ValueError('frozen model, prompt or serving-input contract changed')
        for field in ('embedding_producer_policy', 'embedding_execution_identity',
                      'embedder_identity_files_sha256', 'runtime_environment'):
            if before.get(field) != after.get(field):
                raise ValueError('paired immutable embedding execution environment changed')
        user_mismatch = False
        for expected, earlier, later in zip(anchor['rows'], before['rows'], after['rows']):
            fields = row_differences(earlier, later)
            if fields:
                user_mismatch = True
                mismatches.append({'question_id': earlier['question_id'], 'fields': fields})
            historical_fields = [field for field in row_differences(expected, earlier)
                                 if field not in {'processor_inputs_sha256', 'input_token_ids_sha256',
                                                  'embedding_producer_outputs'}]
            if historical_fields:
                historical_differences.append({'question_id': earlier['question_id'],
                                               'fields': historical_fields})
        try:
            if user_mismatch:
                raise ValueError('whole user stream has mismatched evidence, processor inputs or producer outputs')
            pairs.append({'person_id': original['person_id'], **measured_e2e_pair(before, after)})
        except ValueError as error:
            pairs.append({'person_id': original['person_id'], 'qualification_error': str(error),
                          'measured_server_end_to_end_speedup': None})
        references.append(before)
        candidates.append(after)
    reference_summary = full_service_summary(references)
    candidate_summary = full_service_summary(candidates)
    before_total = reference_summary['service_seconds']['sum_seconds']
    after_total = candidate_summary['service_seconds']['sum_seconds']
    qualified = not mismatches and all('qualification_error' not in pair for pair in pairs)
    return {'schema_version': 'omni-complete-service-pair-audit-v1', 'complete': True,
        'execution_integrity': 'pass' if qualified else 'failed_pair_equivalence_gate',
        'requests': sum(len(run['rows']) for run in candidates),
        'pair_mismatches': mismatches, 'historical_reference_differences': historical_differences,
        'historical_quality_inherited': not historical_differences and qualified,
        'quality_requires_new_evaluation': bool(historical_differences),
        'per_user': pairs, 'reference': reference_summary, 'candidate': candidate_summary,
        'measured_server_end_to_end_speedup': before_total / after_total if qualified else None,
        'latency_reduction_fraction': 1 - after_total / before_total if qualified else None,
        'reference_edge_store': [run['embedding_state_reuse'] for run in references],
        'candidate_edge_store': [run['embedding_state_reuse'] for run in candidates],
        'inputs_sha256': inputs, 'evaluation_label_files_read': [],
        'actual_pim_routing_executed': False, 'modeled_lpddr_end_to_end_speedup': None,
        'new_quality_evaluated': False, 'publication_ready': False,
        'boundary': 'single-pass paired complete server serving; internal edge-vector reuse qualification, not an external baseline/ablation study; no statistical significance, Reader-state reuse or real-PIM acceleration claim'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference-root', type=Path, required=True)
    parser.add_argument('--candidate-root', type=Path, required=True)
    parser.add_argument('--frozen-plan', type=Path)
    parser.add_argument('--screening-cohort', type=Path)
    parser.add_argument('--reference-prefix', default='baseline')
    parser.add_argument('--candidate-prefix', default='reuse')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('fresh pair audit required')
    if bool(args.frozen_plan) == bool(args.screening_cohort):
        raise ValueError('choose frozen paired plan or screening cohort')
    if args.screening_cohort:
        cohort = json.loads(args.screening_cohort.read_text())
        paths, runs, registrations = [], [], []
        for root, prefix in ((args.reference_root, args.reference_prefix),
                             (args.candidate_root, args.candidate_prefix)):
            group, registration_group = {}, {}
            for person_id in cohort['question_ids_by_user']:
                path = root / f'{prefix}_user_{person_id}.json'
                run = json.loads(path.read_text())
                registration = json.loads(path.with_suffix('.registration.json').read_text())
                if not verify_source_snapshot(run, registration):
                    raise ValueError('frozen source identity differs')
                group[person_id] = run
                registration_group[person_id] = registration
                paths.extend((path, path.with_suffix('.registration.json')))
            runs.append(group)
            registrations.append(registration_group)
        contracts = {person_id: embedding_namespace_equivalence(
            runs[0][person_id], runs[1][person_id], registrations[0][person_id], registrations[1][person_id])
            for person_id in cohort['question_ids_by_user']}
        result = screening_summary(*runs, cohort['question_ids_by_user'], namespace_contracts=contracts)
        result['inputs_sha256'] = {str(path): digest(path) for path in [args.screening_cohort, *paths]}
    else:
        result = audit_pair(args.reference_root, args.candidate_root, args.frozen_plan)
    result['producer_sha256'] = digest(Path(__file__))
    result['latency_accounting_sha256'] = digest(Path(__file__).with_name('omni_latency_accounting.py'))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps({key: result[key] for key in ('requests', 'execution_integrity',
        'measured_server_end_to_end_speedup', 'latency_reduction_fraction',
        'requests_per_policy', 'screening_service_ratio') if key in result}))


if __name__ == '__main__':
    main()
