"""Audit the frozen query-only direction diagnostic without opening dataset rows."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path


VERSION = Path(__file__).resolve().parents[1]
ARTIFACTS = VERSION / 'artifacts'
RUN = ARTIFACTS / 'proof_direction_semantic_revalidation_v3_20261025'
PILOT = ARTIFACTS / 'proof_direction_pilot12_v1_20261025'
OPPORTUNITY = ARTIFACTS / 'proof_direction_audit_v1_20261025.json'
ISOLATION = PILOT / 'input_isolation_review.json'
REPORT = VERSION / 'reports/PROOF_DIRECTION_PILOT12_REVIEW_20261025_ZH.md'
OUTPUT = ARTIFACTS / 'proof_direction_completion_audit_v1_20261025.json'


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_json(path: Path) -> dict:
    with path.open(encoding='utf-8') as handle:
        return json.load(handle)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def audit() -> dict:
    protocol_path = RUN / 'protocol.json'
    summary_path = RUN / 'summary.json'
    workers_path = RUN / 'worker_status.json'
    protocol = read_json(protocol_path)
    summary = read_json(summary_path)
    workers = read_json(workers_path)
    pilot_protocol_path = PILOT / 'protocol.json'
    isolation = read_json(ISOLATION)
    opportunity = read_json(OPPORTUNITY)

    cases = protocol['cases']
    case_ids = [case['question_id'] for case in cases]
    require(protocol['registered_before_generation'] is True, 'query cohort was not registered first')
    require(protocol['kind'] == 'query_only_semantic_contract_revalidation', 'unexpected run kind')
    require(len(case_ids) == 12 and len(set(case_ids)) == 12, 'expected twelve unique registered queries')
    require(all(case['split'] == 'development' for case in cases), 'non-development query in registration')
    require(protocol['parent_pilot_protocol_sha256'] == digest(pilot_protocol_path),
            'parent pilot protocol identity mismatch')
    worker_results = workers['workers']
    require(len(worker_results) == len(protocol['users']), 'worker count mismatch')
    require(all(worker['exit_code'] == 0 for worker in worker_results), 'worker failure recorded')
    require({worker['user'] for worker in worker_results} == set(protocol['users']),
            'worker identity mismatch')

    by_user = protocol['question_ids_by_user']
    require(set(by_user) == set(protocol['users']), 'user registration mismatch')
    registered_ids = set(case_ids)
    require({query_id for ids in by_user.values() for query_id in ids} == registered_ids,
            'registered user denominator mismatch')
    require(sum(len(ids) for ids in by_user.values()) == len(case_ids), 'duplicate user assignment')

    snapshot = RUN / 'runtime/scripts'
    for relative_path, expected_hash in protocol['source_files_sha256'].items():
        source_path = snapshot / relative_path
        require(source_path.is_file() and digest(source_path) == expected_hash,
                f'frozen runtime source mismatch: {relative_path}')

    case_by_id = {case['question_id']: case for case in cases}
    observed_ids = []
    output_hashes = {}
    for user, registration in protocol['users'].items():
        output_path = Path(registration['output']).resolve()
        require(output_path.is_relative_to(RUN.resolve()), 'worker output escaped registered run directory')
        run = read_json(output_path)
        expected_ids = registration['question_ids']
        require(run['complete'] is True and run['user'] == user, f'incomplete user output: {user}')
        require(run['question_ids'] == expected_ids, f'user denominator mismatch: {user}')
        rows = run['rows']
        require([row['question_id'] for row in rows] == expected_ids, f'row order mismatch: {user}')
        for row in rows:
            case = case_by_id[row['question_id']]
            require(case['person_id'] == user and row['question'] == case['question'],
                    f'registered query mismatch: {row["question_id"]}')
        observed_ids.extend(row['question_id'] for row in rows)
        output_hashes[user] = digest(output_path)

    require(observed_ids == case_ids, 'generated output denominator/order mismatch')
    require(summary['complete'] is True and summary['requests'] == 12, 'incomplete summary')
    require(summary['user_output_sha256'] == output_hashes, 'summary output hashes mismatch')
    require(summary['parent_pilot_run_modified'] is False, 'parent pilot mutation reported')
    require(summary['answer_labels_read'] is False and summary['history_read'] is False,
            'query-only run reports label/history access')
    require(summary['validation_accessed'] is False and summary['holdout_accessed'] is False,
            'query-only run reports validation/holdout access')
    require(summary['judge_launched'] is False and summary['full_192_run_launched'] is False,
            'query-only run launched a forbidden stage')
    require(summary['baseline_or_ablation_launched'] is False,
            'query-only run launched a baseline or ablation')
    require(summary['method_effectiveness_established'] is False
            and summary['main_experiment_closed'] is False, 'unsupported closure flag')
    require(summary['annotations_registered_before_generation'] is False
            and summary['independent_semantic_gold'] is False,
            'semantic review cannot be represented as independent preregistered gold')

    review_rows = summary['rows']
    require(len(review_rows) == 12, 'semantic review denominator mismatch')
    matches = [row['question_id'] for row in review_rows if row['matches_review_shape']]
    mismatches = [row['question_id'] for row in review_rows if not row['matches_review_shape']]
    require(len(matches) == summary['review_shape_matches'] == 8, 'semantic review count mismatch')
    require(abs(summary['review_shape_fraction'] - 8 / 12) < 1e-12, 'semantic review fraction mismatch')

    require(isolation['frozen_pilot_strict_no_validation_history_read_certified'] is False,
            'historical pilot isolation boundary unexpectedly changed')
    require(isolation['frozen_reader_entry']['all_user_history_read_and_deserialized'] is True,
            'historical reader isolation finding missing')
    require(isolation['frozen_judge_entry']['combined_manifest_and_label_bytes_scanned_or_hashed'] is True,
            'historical judge file-scan finding missing')
    current_review = isolation['current_query_only_semantic_review']
    require(current_review['raw_history_files_opened'] is False
            and current_review['raw_label_files_opened'] is False
            and current_review['validation_or_holdout_records_opened_or_used'] is False,
            'current query-only review isolation finding mismatch')

    require(opportunity['requests'] == 192 and opportunity['correct'] == 61,
            'frozen opportunity audit identity mismatch')
    require(opportunity['accepted_nonempty_witness_domain_size_histogram'] == {'1': 191},
            'frozen witness-domain opportunity count mismatch')
    require(opportunity['state_consumer_count_histogram'] == {'1': 2005},
            'frozen lifecycle-consumer count mismatch')
    require(opportunity['backend_histogram'] == {'host': 192}
            and opportunity['joint_execution_decision_requests'] == 0,
            'frozen backend/execution count mismatch')
    costs = opportunity['cost_seconds']
    online = costs['online_service']
    stage_fractions = {name: costs[name] / online
                       for name in ('source_grounding', 'reader_decode')}

    input_paths = [protocol_path, summary_path, workers_path, pilot_protocol_path,
                   ISOLATION, OPPORTUNITY, REPORT]
    input_hashes = {str(path.relative_to(VERSION)): digest(path) for path in input_paths}
    input_hashes.update({f'{RUN.name}/user_{user}.json': value
                         for user, value in output_hashes.items()})
    input_hashes.update({f'{RUN.name}/runtime/scripts/{path}': expected
                         for path, expected in protocol['source_files_sha256'].items()})

    return {
        'complete': True,
        'audit_kind': 'read_only_development_direction_completion',
        'date': '2026-10-25',
        'registered_development_queries': len(case_ids),
        'post_run_query_shape_review': {
            'matches': len(matches),
            'denominator': len(case_ids),
            'fraction': len(matches) / len(case_ids),
            'mismatched_query_ids': mismatches,
            'interpretation': 'post-run diagnostic against non-independent reviewed shapes; not accuracy or method-effectiveness evidence',
            'annotations_registered_before_generation': False,
            'independent_semantic_gold': False,
        },
        'query_only_protocol': {
            'runtime_sources_verified': len(protocol['source_files_sha256']),
            'registered_query_outputs_verified': len(observed_ids),
            'answer_labels_read': False,
            'history_read': False,
            'validation_accessed': False,
            'holdout_accessed': False,
            'reader_answer_generation': False,
            'judge_baseline_ablation_or_full_192_launched': False,
        },
        'historical_isolation_boundary': {
            'frozen_reader_loaded_combined_history_before_filtering': True,
            'frozen_judge_scanned_or_hashed_combined_manifest_label_bytes': True,
            'validation_or_holdout_records_sent_to_judge': False,
            'frozen_pilot_certified_strict_no_validation_holdout_read': False,
            'current_query_only_review_opened_history_or_labels': False,
            'access_claim_basis': 'frozen protocol, source review and recorded isolation audit; not an OS-level file-access trace',
        },
        'frozen_192_opportunity_diagnostic': {
            'requests': opportunity['requests'],
            'proxy_judge_correct': opportunity['correct'],
            'accepted_nonempty_witness_domains': opportunity['accepted_nonempty_witness_domain_size_histogram'],
            'lifecycle_state_consumer_counts': opportunity['state_consumer_count_histogram'],
            'backend_histogram': opportunity['backend_histogram'],
            'joint_execution_decisions': opportunity['joint_execution_decision_requests'],
            'lifecycle_compiled_after_generation': opportunity['execution_order']['lifecycle_compiled_after_generation'],
            'backend_chosen_after_generation': opportunity['execution_order']['backend_chosen_after_generation'],
            'online_service_seconds': online,
            'source_grounding_fraction': stage_fractions['source_grounding'],
            'reader_decode_fraction': stage_fractions['reader_decode'],
            'decode_zero_time_stage_ratio_upper_bound': opportunity['optimistic_stage_envelopes']['reader_decode']['zero_time_stage_service_ratio_upper_bound'],
        },
        'research_decision': {
            'single_primary_hypothesis': 'source-located, semantically sufficient complementary evidence compilation with unique-object cost and exact Reader budgeting',
            'query_parser': 'unverified interface; stop repeated prompt tuning on this fixed 12-query set',
            'lifecycle': 'correctness/resource accounting only until it changes a pre-Reader execution plan and demonstrates consumer reuse',
            'pim': 'exclude from primary contribution; conditional backend study only after workload and calibrated device justify it',
            'main_experiment_closed': False,
            'method_effectiveness_established': False,
            'full_192_or_comparison_or_ablation_launched': False,
        },
        'input_sha256': input_hashes,
        'audit_script_sha256': digest(Path(__file__).resolve()),
        'boundary': 'No raw histories, answer labels, validation or holdout records are opened by this audit; integrity checks do not independently prove historical process file access.',
    }


def main() -> None:
    require(not OUTPUT.exists(), f'audit output already exists: {OUTPUT}')
    result = audit()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ('input_sha256',)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
