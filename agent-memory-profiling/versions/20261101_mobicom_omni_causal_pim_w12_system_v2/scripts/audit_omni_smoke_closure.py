"""Evaluation-only integrity audit; smoke execution is not mainline qualification."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            result.update(block)
    return result.hexdigest()


def empirical_latency(values):
    if not values or any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError('nonempty finite nonnegative latencies required')
    ordered = sorted(values)
    return {'count': len(values), 'mean_seconds': sum(values) / len(values),
            'p50_nearest_rank_seconds': ordered[math.ceil(0.50 * len(values)) - 1],
            'p95_nearest_rank_seconds': ordered[math.ceil(0.95 * len(values)) - 1],
            'max_seconds': ordered[-1], 'sum_seconds': sum(values),
            'quantile_method': 'nearest-rank ceil(p*N), empirical only'}


def verify_source_snapshot(smoke, registration):
    snapshot_name = registration.get('source_snapshot_directory')
    if snapshot_name is None:
        return False
    snapshot = Path(snapshot_name)
    for name, expected in smoke['source_files_sha256'].items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('unsafe frozen source identity')
        if digest(snapshot / relative) != expected:
            raise ValueError('frozen source snapshot mismatch')
    return True


def audit_performance_pair(verified, performance):
    if (not performance.get('complete') or not performance.get('performance_comparable')
            or performance.get('numerical_probe_overhead_in_service')
            or performance.get('verify_generation')):
        raise ValueError('complete performance-only execution required')
    for name in ('reader_config_sha256', 'embedder_config_sha256', 'prepared_audit_sha256',
                 'archive_sha256', 'system_prompt_sha256', 'evidence_policy',
                 'candidate_sessions', 'max_images', 'max_new_tokens', 'total_text_tokens'):
        if verified[name] != performance[name]:
            raise ValueError('performance and verified execution protocols differ')
    if len(verified['rows']) != len(performance['rows']):
        raise ValueError('performance request count differs')
    rows = []
    for reference, row in zip(verified['rows'], performance['rows']):
        if reference.get('generation_equivalent') is not True:
            raise ValueError('paired reference lacks independent generation validation')
        for name in ('question_id', 'question_sha256', 'answer_sha256', 'answer_token_ids',
                     'input_tokens', 'images', 'retrieved_session_ids', 'evidence_selection',
                     'hypergraph_evidence_plan', 'stop_reason'):
            if reference[name] != row[name]:
                raise ValueError('performance execution differs from verified inputs/output')
        rows.append({name: row[name] for name in ('question_id', 'service_seconds',
            'prefill_seconds', 'decode_seconds', 'retrieval_seconds',
            'service_peak_cuda_allocated_bytes', 'service_peak_cuda_reserved_bytes')})
    return {'input_output_pair_integrity': 'pass', 'rows': rows,
            'boundary': 'two instrumented server requests; CUDA allocator peaks exclude non-PyTorch/CPU memory; no population latency quantile or device/PIM speedup'}


def audit_records(smoke, requests, judge, projection, labels, registration, *, require_verified_execution=True):
    if not smoke.get('complete') or not judge.get('complete') or not smoke.get('rows'):
        raise ValueError('complete nonempty Reader and judge results required')
    if smoke.get('evaluation_label_files_read') != []:
        raise ValueError('serving label isolation not attested')
    if registration.get('source_files_sha256') != smoke.get('source_files_sha256'):
        raise ValueError('registration/runtime source identity differs')
    collections = [smoke['rows'], requests['judge_requests'], judge['rows']]
    maps = []
    for rows in collections:
        mapping = {row['question_id']: row for row in rows}
        if len(mapping) != len(rows):
            raise ValueError('duplicate artifact question identity')
        maps.append(mapping)
    predictions, prompts, judgments = maps
    if set(predictions) != set(prompts) or set(predictions) != set(judgments):
        raise ValueError('artifact question sets differ')
    if registration['question_ids'] != list(predictions):
        raise ValueError('executed questions differ from registration')
    rows = []
    for question_id, row in predictions.items():
        prompt, judgment = prompts[question_id], judgments[question_id]
        if (judgment['parse_valid'] and judgment['value'] not in (0.0, 1.0)) or (
                not judgment['parse_valid'] and judgment['value'] is not None):
            raise ValueError('judge parse/value contract differs')
        answer_hash = hashlib.sha256(row['answer'].encode()).hexdigest()
        if answer_hash != row['answer_sha256'] or any(item['prediction_sha256'] != answer_hash
                                                     for item in (prompt, judgment)):
            raise ValueError('prediction hash chain differs')
        if (hashlib.sha256(prompt['prompt'].encode()).hexdigest() != prompt['prompt_sha256']
                or prompt['prompt_sha256'] != judgment['prompt_sha256']):
            raise ValueError('judge prompt hash chain differs')
        if require_verified_execution and (row.get('generation_equivalent') is not True or row['stop_reason'] != 'eos'):
            raise ValueError('generation equivalence/EOS gate failed')
        if row['stop_reason'] not in {'eos', 'max_new_tokens'}:
            raise ValueError('unknown Reader stop reason')
        if not row.get('images'):
            raise ValueError('text-only request cannot pass raw-image smoke')
        plan = row.get('hypergraph_evidence_plan')
        if plan is not None:
            union = {atom_id for edge in plan['selected_hyperedges'] for atom_id in edge['atom_ids']}
            if union != set(plan['accepted_atom_ids']) or plan['text_tokens'] > plan['text_budget']:
                raise ValueError('hyperedge closure or token budget failed')
            if len(plan['images']) > plan['max_images']:
                raise ValueError('image capacity failed')
            if any(image['source_atom_id'] not in union for image in row['images']):
                raise ValueError('image detached from its selected context')
            planned_members = {ref.get('resolved', ref['normalized']) for ref in plan['images']}
            if planned_members != {ref['archive_member'] for ref in row['images']}:
                raise ValueError('planned and executed image identities differ')
        label = labels[question_id]
        sources = set(label['source_session_ids'])
        expected_images = {ref.get('resolved', ref['normalized']) for ref in label['evidence_image_refs']}
        selected_images = {ref['archive_member'] for ref in row['images']}
        rows.append({'question_id': question_id, 'local_judge_value': judgment['value'],
                     'judge_parse_valid': judgment['parse_valid'],
                     'stop_reason': row['stop_reason'],
                     'generation_equivalent': row.get('generation_equivalent'),
                     'source_session_recall': len(sources & set(row['retrieved_session_ids'])) / len(sources)
                     if sources else None,
                     'label_image_reference_recall': len(expected_images & selected_images) / len(expected_images)
                     if expected_images else None,
                     'service_seconds': row['service_seconds'], 'prefill_seconds': row['prefill_seconds'],
                     'decode_seconds': row['decode_seconds'], 'input_tokens': row['input_tokens']})
    if projection['resident_bytes'] > projection['capacity_bytes'] or projection['end_to_end_speedup'] is not None:
        raise ValueError('projection capacity/claim boundary failed')
    functional = smoke['operator_trace']['functional_samples']
    if smoke.get('numerical_probe_overhead_in_service'):
        if len(functional) != 3 or any(not sample['allclose'] for sample in functional.values()):
            raise ValueError('actual activation functional probes did not pass')
    parsed = all(row['judge_parse_valid'] for row in rows)
    strict_successes = sum(row['judge_parse_valid'] and row['local_judge_value'] == 1.0
                           and row['stop_reason'] == 'eos' for row in rows)
    eos_count = sum(row['stop_reason'] == 'eos' for row in rows)
    return {'schema_version': 'omni-smoke-closure-audit-v1',
            'execution_integrity': 'pass', 'rows': rows,
            'local_semantic_proxy': sum(row['local_judge_value'] for row in rows) / len(rows) if parsed else None,
            'strict_success_rate': strict_successes / len(rows),
            'strict_local_proxy_success_rate': strict_successes / len(rows),
            'strict_local_proxy_successes': strict_successes,
            'attempted_requests': len(registration['question_ids']),
            'eos_requests': eos_count,
            'truncated_question_ids': [row['question_id'] for row in rows if row['stop_reason'] != 'eos'],
            'eos_rate': eos_count / len(rows),
            'empirical_service_latency_all_requests': empirical_latency([row['service_seconds'] for row in rows]),
            'empirical_prefill_latency_all_requests': empirical_latency([row['prefill_seconds'] for row in rows]),
            'empirical_decode_latency_all_requests': empirical_latency([row['decode_seconds'] for row in rows]),
            'invalid_judgments': sum(not row['judge_parse_valid'] for row in rows),
            'performance_comparable': smoke['performance_comparable'],
            'latency_statistics_qualified': False,
            'generation_equivalence_qualified': all(row['generation_equivalent'] is True for row in rows),
            'main_experiment_closed': False, 'publication_ready': False,
            'pim': {key: projection[key] for key in ('resident_bytes', 'capacity_bytes',
                'warm_linear_speedup_scenario', 'cold_linear_speedup_scenario', 'end_to_end_speedup',
                'full_model_residency_validated')},
            'boundary': 'fixed development smoke only; label-side source hits are not visual grounding; no population accuracy, latency quantile, or E2E acceleration claim',
            'remaining_gates': ['larger frozen user-disjoint development protocol',
                                'image-dependent grounding evaluation', 'exact multimodal state reuse',
                                'shared weights/index/state pool allocation',
                                'full model and runtime DRAM capacity',
                                'calibrated hardware assumptions and shared-resource execution schedule']}


def main():
    parser = argparse.ArgumentParser()
    for name in ('prepared', 'smoke', 'judge-requests', 'judge', 'projection', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--performance-smoke', type=Path)
    parser.add_argument('--performance-only', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('fresh audit artifact required')
    inputs = {name: getattr(args, name) for name in ('smoke', 'judge_requests', 'judge', 'projection')}
    records = {name: json.loads(path.read_text()) for name, path in inputs.items()}
    smoke_hash = digest(args.smoke)
    if any(records[name]['smoke_sha256'] != smoke_hash for name in ('judge_requests', 'projection')):
        raise ValueError('Reader producer hash differs')
    if records['judge']['provenance']['scores_sha256'] != digest(args.judge_requests):
        raise ValueError('judge producer hash differs')
    prepared_audit = json.loads((args.prepared / 'audit.json').read_text())
    if records['smoke']['prepared_audit_sha256'] != digest(args.prepared / 'audit.json'):
        raise ValueError('prepared audit identity differs')
    labels_path = args.prepared / 'labels.jsonl'
    if digest(labels_path) != prepared_audit['artifacts_sha256']['labels.jsonl']:
        raise ValueError('label producer hash differs')
    labels = {row['question_id']: row for row in (json.loads(line) for line in labels_path.read_text().splitlines())}
    registration_path = args.smoke.with_suffix('.registration.json')
    registration = json.loads(registration_path.read_text())
    if args.performance_only and (not records['smoke'].get('performance_comparable')
                                 or records['smoke'].get('numerical_probe_overhead_in_service')
                                 or records['smoke'].get('verify_generation')):
        raise ValueError('performance-only mode requires an unprobed, unverified timing execution')
    result = audit_records(records['smoke'], records['judge_requests'], records['judge'],
                           records['projection'], labels, registration,
                           require_verified_execution=not args.performance_only)
    result['performance_only'] = args.performance_only
    result['source_snapshot_verified'] = verify_source_snapshot(records['smoke'], registration)
    result['inputs_sha256'] = {name: digest(path) for name, path in inputs.items()}
    result['registration_sha256'] = digest(registration_path)
    if args.performance_smoke is not None:
        performance = json.loads(args.performance_smoke.read_text())
        result['performance_pair'] = audit_performance_pair(records['smoke'], performance)
        performance_registration = json.loads(args.performance_smoke.with_suffix('.registration.json').read_text())
        if performance_registration['source_files_sha256'] != performance['source_files_sha256']:
            raise ValueError('performance registration/runtime source mismatch')
        if not verify_source_snapshot(performance, performance_registration):
            raise ValueError('paired timing execution lacks a frozen source snapshot')
        result['inputs_sha256']['performance_smoke'] = digest(args.performance_smoke)
        result['performance_pair']['source_snapshot_verified'] = True
    result['script_sha256'] = digest(Path(__file__))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'execution_integrity': result['execution_integrity'],
                      'local_semantic_proxy': result['local_semantic_proxy'],
                      'main_experiment_closed': False}))


if __name__ == '__main__':
    main()
