"""Separate label-side judge, memory ledger, and hash audit for a frozen cohort."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from audit_omni_smoke_closure import audit_records, digest, empirical_latency, verify_source_snapshot


def combine_requests(parts):
    combined, seen = [], set()
    for part in parts:
        for request in part['judge_requests']:
            if request['question_id'] in seen:
                raise ValueError('duplicate cohort judge identity')
            combined.append(request)
            seen.add(request['question_id'])
    if not combined:
        raise ValueError('nonempty cohort judge input required')
    return combined


def verify_candidate_protocol(plan, registration, smoke):
    quota = plan.get('image_edges_per_session', 0)
    novelty_weight = plan.get('image_novelty_weight', 0.0)
    arguments = registration['arguments']
    visual_budget = plan.get('visual_cache_bytes', 0)
    visual_store = smoke.get('visual_state_reuse') or {}
    if (visual_budget < 0 or arguments.get('visual_cache_bytes', 0) != visual_budget
            or visual_store.get('capacity_bytes', 0) != visual_budget
            or (visual_budget and (arguments.get('verify_generation') is not True
                                   or smoke.get('verify_generation') is not True))):
        raise ValueError('registered visual budget or independent validation differs')
    if quota < 0 or arguments.get('image_edges_per_session', 0) != quota:
        raise ValueError('registered image frontier differs from cohort plan')
    if (not math.isfinite(novelty_weight) or novelty_weight < 0
            or arguments.get('image_novelty_weight', 0.0) != novelty_weight):
        raise ValueError('registered image novelty objective differs from cohort plan')
    if smoke['max_new_tokens'] != plan['max_new_tokens'] or arguments['max_new_tokens'] != plan['max_new_tokens']:
        raise ValueError('registered output ceiling differs from executed protocol')
    for row in smoke['rows']:
        evidence = row['hypergraph_evidence_plan']
        if evidence.get('image_candidate_quota_per_session', 0) != quota:
            raise ValueError('executed image frontier differs from cohort plan')
        if evidence.get('image_novelty_weight', 0.0) != novelty_weight:
            raise ValueError('executed image novelty objective differs from cohort plan')
        if (evidence['candidate_hyperedge_limit'] != arguments['max_hyperedges']
                or evidence['candidate_hyperedge_count'] != len(evidence['candidate_replay'])
                or not 0 < evidence['candidate_hyperedge_count'] <= evidence['candidate_hyperedge_limit']):
            raise ValueError('executed candidate frontier exceeds or differs from registration')


def write_json(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def run_script(script, *arguments):
    subprocess.run([sys.executable, str(ROOT / 'scripts' / script), *map(str, arguments)], check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError('fresh evaluation output directory required')
    plan_path, status_path = args.run_root / 'cohort_plan.json', args.run_root / 'worker_status.json'
    plan, status = json.loads(plan_path.read_text()), json.loads(status_path.read_text())
    if not status['all_workers_exited_zero'] or not status['sources_unchanged']:
        raise ValueError('cohort workers did not complete; retain failures and partial outputs, do not qualify cohort')
    if digest(args.prepared / 'audit.json') != plan['prepared_audit_sha256']:
        raise ValueError('cohort prepared audit changed')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    records = []
    visual_audits = []
    visual_enabled = bool(plan.get('visual_cache_bytes', 0))
    for job in plan['jobs']:
        path = Path(job['output'])
        smoke = json.loads(path.read_text())
        registration = json.loads(path.with_suffix('.registration.json').read_text())
        verify_candidate_protocol(plan, registration, smoke)
        if not smoke.get('complete') or [row['question_id'] for row in smoke['rows']] != job['question_ids']:
            raise ValueError('cohort request coverage differs from registration')
        if any(row['person_id'] != job['person_id'] for row in smoke['rows']):
            raise ValueError('cohort user identity differs')
        if any(plan['source_files_sha256'].get(name) != value
               for name, value in smoke['source_files_sha256'].items()):
            raise ValueError('cohort serving source differs from pre-execution plan')
        if (not smoke['performance_comparable'] or smoke['numerical_probe_overhead_in_service']
                or bool(smoke['verify_generation']) != visual_enabled):
            raise ValueError('unprobed execution and registered independent visual-validation mode required')
        if not verify_source_snapshot(smoke, registration):
            raise ValueError('cohort lacks frozen source snapshot')
        prefix = args.output_dir / f"user_{job['person_id']}"
        requests_path = prefix.with_name(prefix.name + '_requests.json')
        projection_path = prefix.with_name(prefix.name + '_pim.json')
        joint_path = prefix.with_name(prefix.name + '_joint_capacity.json')
        run_script('build_omni_smoke_judge_requests.py', '--prepared', args.prepared,
                   '--smoke', path, '--output', requests_path)
        run_script('project_decode_pim.py', '--smoke', path, '--output', projection_path)
        run_script('project_omni_joint_capacity.py', '--smoke', path, '--device-capacity-bytes',
                   4 * 1024**3, '--output', joint_path)
        if visual_enabled:
            visual_path = prefix.with_name(prefix.name + '_visual_reuse_audit.json')
            run_script('audit_omni_visual_reuse.py', '--smoke', path,
                       '--joint-capacity', joint_path, '--output', visual_path)
            visual_audits.append(json.loads(visual_path.read_text()))
        records.append((job, smoke, registration, requests_path, projection_path, joint_path))
    combined_path = args.output_dir / 'judge_requests.json'
    parts = [json.loads(item[3].read_text()) for item in records]
    combined_requests = combine_requests(parts)
    write_json(combined_path, {'judge_requests': combined_requests,
                               'cohort_plan_sha256': digest(plan_path),
                               'part_sha256': {str(item[3]): digest(item[3]) for item in records},
                               'boundary': 'evaluation-only shared judge batch; serving never reads labels'})
    judge_path = args.output_dir / 'judge.json'
    run_script('judge_mobilemem_local.py', '--scores', combined_path, '--output', judge_path)
    judge = json.loads(judge_path.read_text())
    if not judge.get('complete') or judge['provenance']['scores_sha256'] != digest(combined_path):
        raise ValueError('incomplete or mutated cohort judge producer')
    expected_ids = {item['question_id'] for item in combined_requests}
    if len(judge['rows']) != len(expected_ids) or {item['question_id'] for item in judge['rows']} != expected_ids:
        raise ValueError('cohort judge coverage differs')
    labels_path = args.prepared / 'labels.jsonl'
    prepared_audit = json.loads((args.prepared / 'audit.json').read_text())
    if digest(labels_path) != prepared_audit['artifacts_sha256']['labels.jsonl']:
        raise ValueError('evaluation label producer changed')
    labels = {row['question_id']: row for row in (json.loads(line) for line in labels_path.read_text().splitlines())}
    audits, source_hashes, all_rows = [], {}, []
    for job, smoke, registration, request_path, projection_path, joint_path in records:
        requests, projection = json.loads(request_path.read_text()), json.loads(projection_path.read_text())
        if any(item['smoke_sha256'] != digest(Path(job['output'])) for item in (requests, projection)):
            raise ValueError('cohort Reader producer hash differs')
        identities = set(job['question_ids'])
        subset = {**judge, 'rows': [row for row in judge['rows'] if row['question_id'] in identities]}
        audit = audit_records(smoke, requests, subset, projection, labels, registration,
                              require_verified_execution=visual_enabled)
        audit.update(person_id=job['person_id'], source_snapshot_verified=True,
                     provenance={'reader_sha256': digest(Path(job['output'])),
                                 'requests_sha256': digest(request_path), 'pim_sha256': digest(projection_path),
                                 'joint_capacity_sha256': digest(joint_path),
                                 'shared_judge_sha256': digest(judge_path),
                                 'shared_judge_input_sha256': digest(combined_path),
                                 'judge_subset_question_ids': job['question_ids']})
        write_json(args.output_dir / f"user_{job['person_id']}_audit.json", audit)
        audits.append(audit)
        all_rows.extend(audit['rows'])
        source_hashes[str(job['output'])] = digest(Path(job['output']))
    registered_count = sum(len(job['question_ids']) for job in plan['jobs'])
    image_rows = [row for row in all_rows if row['label_image_reference_recall'] is not None]
    summary = {'schema_version': 'omni-registered-cohort-evaluation-v1',
               'registered_requests': registered_count, 'executed_requests': len(all_rows),
               'development_users': [job['person_id'] for job in plan['jobs']],
               'image_edges_per_session': plan.get('image_edges_per_session', 0),
               'image_novelty_weight': plan.get('image_novelty_weight', 0.0),
               'visual_cache_bytes_per_user': plan.get('visual_cache_bytes', 0),
               'visual_independent_validation_qualified': visual_enabled,
               'visual_natural_batch_hits': sum(item['natural_batch_hits'] for item in visual_audits),
               'visual_natural_batch_hit_rate': (sum(item['natural_batch_hits'] for item in visual_audits)
                                                / registered_count if visual_enabled else None),
               'latency_mode': ('diagnostic; independent reference outside service timer; order/cache/thermal confounded'
                                if visual_enabled else 'unprobed serving run'),
               'execution_integrity': 'pass',
               'local_semantic_proxy': judge['summary']['local_semantic_answer_accuracy'],
               'strict_local_proxy_successes': sum(audit['strict_local_proxy_successes'] for audit in audits),
               'strict_local_proxy_success_rate': sum(audit['strict_local_proxy_successes'] for audit in audits) / registered_count,
               'eos_requests': sum(audit['eos_requests'] for audit in audits),
               'truncated_question_ids': [identity for audit in audits for identity in audit['truncated_question_ids']],
               'empirical_service_latency_all_requests': empirical_latency([row['service_seconds'] for row in all_rows]),
               'source_session_recall_mean': sum(row['source_session_recall'] for row in all_rows
                                                 if row['source_session_recall'] is not None) /
                                              sum(row['source_session_recall'] is not None for row in all_rows),
               'image_reference_questions': len(image_rows),
               'label_image_reference_recall_mean': sum(row['label_image_reference_recall'] for row in image_rows) / len(image_rows)
                                                      if image_rows else None,
               'per_user': [{name: audit[name] for name in ('person_id', 'local_semantic_proxy',
                            'strict_local_proxy_success_rate', 'eos_requests', 'truncated_question_ids')}
                            for audit in audits],
               'reader_sha256': source_hashes, 'cohort_plan_sha256': digest(plan_path),
               'worker_status_sha256': digest(status_path), 'judge_sha256': digest(judge_path),
               'script_sha256': digest(Path(__file__)),
               'full_device_residency_qualified': False, 'end_to_end_pim_speedup': None,
               'main_experiment_closed': False, 'publication_ready': False,
               'boundary': 'server development QA only; uniform 512-token protocol differs from earlier runs; empirical quantiles across independent users, no population significance; image-reference hits are not visual grounding; optional exact visual features only, no Reader KV/recurrent reuse or real PIM acceleration'}
    write_json(args.output_dir / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
