"""Independent protocol gate for the fresh linear-reranker development screen."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from audit_joint_closure import audit as audit_costs
from audit_omni_smoke_closure import empirical_latency, verify_source_snapshot
from omni_joint_index import digest


def validate_run(run, registration, protocol, question_ids, user, policy):
    if (not run.get('complete') or not run.get('screening_run')
            or run.get('evaluation_label_files_read') != []
            or run.get('embedding_state_reuse_enabled') is not False
            or run.get('visual_state_reuse') is not None):
        raise ValueError('complete label-isolated no-reuse screening required')
    arguments = registration['arguments']
    expected = {'max_images': protocol['max_images'], 'max_new_tokens': protocol['max_new_tokens'],
        'input_budget': protocol['input_budget'], 'total_text_tokens': protocol['text_budget'],
        'image_max_pixels': protocol['reader_image_max_pixels'], 'user_id': user,
        'evidence_policy': policy, 'disable_edge_reuse': True, 'visual_cache_bytes': 0,
        'profile_reranker': False, 'verify_generation': False, 'validate_bank_gemv': False}
    if policy == 'provenance_milp':
        expected.update(relevance_modality='joint_index', joint_patch_projection='linear',
            relevance_batch_size=protocol['reranker_batch_size'],
            joint_recall_count=protocol['visual_recall_count'],
            joint_text_recall_count=protocol['text_recall_count'],
            joint_index_image_max_pixels=protocol['index_image_max_pixels'])
        if run.get('joint_patch_projection') != 'linear' or run.get('relevance_modality') != 'joint_index':
            raise ValueError('linear reranker numerical contract not executed')
    if any(arguments.get(name) != value for name, value in expected.items()):
        raise ValueError('execution differs from registered budgets or numerical contract')
    if registration['cohort_sha256'] != protocol['cohort_sha256']:
        raise ValueError('registered cohort changed')
    if registration['source_files_sha256'] != run['source_files_sha256']:
        raise ValueError('runtime differs from worker source registration')
    for name, checksum in run['source_files_sha256'].items():
        if protocol['sources_sha256'].get(name) != checksum:
            raise ValueError('runtime differs from global source registration')
    if (registration['question_ids'] != question_ids
            or [row['question_id'] for row in run['rows']] != question_ids):
        raise ValueError('frozen request denominator or order differs')
    for row in run['rows']:
        if row['person_id'] != user or hashlib.sha256(row['answer'].encode()).hexdigest() != row['answer_sha256']:
            raise ValueError('request or answer identity differs')
        ledger = row['service_stage_ledger']['disjoint_seconds']
        if (any(not math.isfinite(value) or value < 0 for value in ledger.values())
                or not math.isclose(sum(ledger.values()), row['service_seconds'], rel_tol=1e-8, abs_tol=1e-7)):
            raise ValueError('service ledger does not reconcile')


def audit(root, judge):
    root = root.resolve()
    protocol = json.loads((root / 'protocol.json').read_text())
    cohort = json.loads((root / 'cohort.json').read_text())
    if (digest(root / 'cohort.json') != protocol['cohort_sha256']
            or set(cohort['question_ids_by_user']) != {'0', '1', '2'}
            or any(len(identities) != 16 for identities in cohort['question_ids_by_user'].values())):
        raise ValueError('original 48-request development cohort required')
    for name, checksum in protocol['sources_sha256'].items():
        if digest(root / 'runtime' / name) != checksum:
            raise ValueError('frozen global runtime changed')
    original = Path(protocol['producer_construction_source_protocol'])
    if digest(original) != protocol['producer_construction_source_protocol_sha256']:
        raise ValueError('producer construction protocol changed')
    latencies = {'local_hyperedge': [], 'provenance_milp': []}
    for user, identities in cohort['question_ids_by_user'].items():
        paired = {}
        for policy in latencies:
            path = root / f'{policy}_user_{user}.json'
            run = json.loads(path.read_text())
            registration = json.loads(path.with_suffix('.registration.json').read_text())
            validate_run(run, registration, protocol, identities, user, policy)
            if not verify_source_snapshot(run, registration):
                raise ValueError('worker source snapshot required')
            paired[policy] = run
            latencies[policy].extend(row['service_seconds'] for row in run['rows'])
            if policy == 'provenance_milp':
                index = protocol['indexes'][user]
                directory = Path(index['path'])
                if Path(registration['arguments']['joint_index']).resolve() != directory.resolve():
                    raise ValueError('unregistered index path')
                files = {str(path.relative_to(directory)): digest(path)
                         for path in directory.rglob('*') if path.is_file()}
                if files != index['files_sha256']:
                    raise ValueError('frozen index changed')
        for name in ('reader_identity', 'reader_config_sha256', 'embedder_config_sha256',
                     'system_prompt_sha256', 'prepared_audit_sha256', 'archive_sha256',
                     'image_max_pixels', 'max_images', 'max_new_tokens', 'total_text_tokens'):
            if paired['local_hyperedge'][name] != paired['provenance_milp'][name]:
                raise ValueError('baseline/candidate shared Reader or input contract differs')
    result = audit_costs(root, judge)
    result.update(protocol_integrity_passed=True, source_snapshots_verified=True,
        service_latency_screening_only={policy: empirical_latency(values) for policy, values in latencies.items()},
        protocol_sha256=digest(root / 'protocol.json'), script_sha256=digest(Path(__file__)),
        isolated_end_to_end_speedup_qualified=False, linear_implementation_repair_is_pim=False,
        boundary='full-budget development numerical requalification; not causal latency, state reuse, PIM gain, task execution success, or publication acceptance')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--judge', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('fresh audit output required')
    result = audit(args.run_root, args.judge)
    with args.output.open('x') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps({'complete': True, 'protocol_integrity_passed': True,
                      'main_experiment_closed': False}))


if __name__ == '__main__':
    main()
