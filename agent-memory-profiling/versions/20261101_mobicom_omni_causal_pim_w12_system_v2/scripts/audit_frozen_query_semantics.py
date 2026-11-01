"""Post-run, query-only structural review of the frozen 12-request pilot."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from omni_proof_query_contract import parse_query_contract
from omni_proof_runtime import file_digest


EXPECTED = {
    '0_q_83': {'mode': 'structured', 'facts': 1, 'operations': [],
               'rationale': 'The conversation is a locator/context for one requested final study-plan outcome, not a comparison operand.'},
    '0_q_95': {'mode': 'structured', 'facts': 1, 'operations': [],
               'rationale': 'A requested purchased-items list is one historical observation.'},
    '0_q_123': {'mode': 'structured', 'facts': 2, 'operations': ['comparison'],
                'rationale': 'The question compares communication states in two explicitly named periods.'},
    '0_q_662': {'mode': 'structured', 'facts': 2, 'operations': ['difference'],
                'rationale': 'Elapsed time needs the two event-time operands and one derived difference.'},
    '1_q_72': {'mode': 'structured', 'facts': 1, 'operations': [],
               'rationale': 'The two prioritized criteria form one requested historical list, not an open advice query.'},
    '1_q_78': {'mode': 'structured', 'facts': 1, 'operations': [],
               'rationale': 'The two protected spending categories form one historical preference observation.'},
    '1_q_270': {'mode': 'structured', 'facts': 1, 'operations': [],
                'rationale': 'The requested usual platform is one fact; the supermarket contrast is contextual.'},
    '1_q_663': {'mode': 'structured', 'facts': 2, 'operations': ['difference'],
                'rationale': 'Elapsed time between two stretching events requires two event-time observations.'},
    '2_q_116': {'mode': 'structured', 'facts': 1, 'operations': [],
                'rationale': 'The initial check-in contents are one requested list.'},
    '2_q_715': {'mode': 'structured', 'facts': 2, 'operations': ['difference'],
                'rationale': 'The question asks for an interval between two events; source sufficiency and answer correctness are separate.'},
    '2_q_747': {'mode': 'structured', 'facts': 4,
                'operations': ['difference', 'difference', 'comparison'],
                'rationale': 'Each event needs its own time and its own dinner reference; a single unscoped dinner fact cannot support both intervals.'},
    '2_q_799': {'mode': 'structured', 'facts': 2, 'operations': ['difference'],
                'rationale': 'Elapsed time from departure to arrival requires both event-time observations.'},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    if output.exists():
        raise ValueError('fresh audit output required')
    protocol = json.loads((root / 'protocol.json').read_text())
    structure = json.loads((root / 'structural_audit.json').read_text())
    summary = json.loads((root / 'summary.json').read_text())
    isolation_path = root / 'input_isolation_review.json'
    if not isolation_path.exists():
        raise ValueError('explicit input-isolation boundary audit required')
    isolation = json.loads(isolation_path.read_text())
    if not protocol.get('registered_before_generation') or len(EXPECTED) != 12:
        raise ValueError('frozen 12-request pilot protocol required')
    registered = {case['question_id']: case['question'] for case in protocol['cases']}
    if set(registered) != set(EXPECTED):
        raise ValueError('question-only denominator differs from registered pilot')
    rows_by_id = {}
    source_hashes = {}
    for job in protocol['jobs']:
        path = Path(job['output'])
        expected_hash = structure['run_sha256'].get(str(path))
        if not expected_hash or file_digest(path) != expected_hash:
            raise ValueError('pilot run changed since its structural audit')
        source_hashes[str(path)] = expected_hash
        run = json.loads(path.read_text())
        if not run.get('complete'):
            raise ValueError('complete frozen run required')
        for row in run['rows']:
            identity = row['question_id']
            if identity in rows_by_id:
                raise ValueError('duplicate pilot question')
            rows_by_id[identity] = row
    if set(rows_by_id) != set(EXPECTED) or summary['requests'] != len(EXPECTED):
        raise ValueError('pilot denominator mismatch')

    audited = []
    for identity, expected in EXPECTED.items():
        row = rows_by_id[identity]
        plan = row['proof_plan']
        contract = plan['query_contract']
        observed_operations = [operation['kind'] for operation in contract['operations']]
        observed = {'mode': contract['answer_mode'], 'facts': len(contract['source_facts']),
                    'operations': observed_operations}
        matches = observed == {key: expected[key] for key in ('mode', 'facts', 'operations')}
        scoped_reference_ok = True
        if identity == '2_q_747':
            roles = [fact['text'] for fact in contract['source_facts']]
            scoped_reference_ok = (len(roles) == 4 and sum('晚饭' in role for role in roles) == 2
                                   and any('外卖' in role for role in roles)
                                   and any('水果酸奶' in role for role in roles))
            matches = matches and scoped_reference_ok
        replay = parse_query_contract(plan['query_parse']['parsed'], registered[identity])
        audited.append({'question_id': identity, 'query': registered[identity],
            'expected_structure': {key: expected[key] for key in ('mode', 'facts', 'operations')},
            'observed_structure': observed, 'current_contract_replay_status': replay['status'],
            'matches_query_only_review': matches, 'scoped_reference_roles_ok': scoped_reference_ok,
            'review_rationale': expected['rationale']})

    result = {'complete': True, 'audit_kind': 'post_run_query_only_semantic_structure_review',
        'registered_requests': len(EXPECTED), 'matched_reviewed_query_shapes': sum(
            row['matches_query_only_review'] for row in audited),
        'reviewed_query_shape_fraction': sum(row['matches_query_only_review'] for row in audited) / len(audited),
        'current_contract_replay_status': dict(Counter(row['current_contract_replay_status'] for row in audited)),
        'answer_labels_read': False, 'evaluation_label_files_read': [], 'gold_answers_read': False,
        'validation_accessed': False, 'holdout_accessed': False,
        'annotations_registered_before_generation': False,
        'frozen_pilot_strict_no_validation_history_read_certified': isolation[
            'frozen_pilot_strict_no_validation_history_read_certified'],
        'evaluation_label_file_opened_during_this_audit': False,
        'shared_all_user_history_file_opened_during_this_audit': False,
        'input_isolation_review_sha256': file_digest(isolation_path),
        'annotation_scope': 'post-run reviewer assessment of query semantics only; no evidence/gold-answer inspection and no inter-rater study',
        'raw_proxy_judge_score_used_for_annotations': False,
        'frozen_pilot_run_sha256': source_hashes,
        'current_parser_sha256': file_digest(Path(__file__).resolve().parent / 'omni_proof_query_contract.py'),
        'current_planner_sha256': file_digest(Path(__file__).resolve().parent / 'omni_proof_planner.py'),
        'prompt_revalidated_by_inference': False,
        'answer_quality_or_method_effectiveness_established': False,
        'main_experiment_closed': False,
        'rows': audited,
        'boundary': 'This is a transparent, post-run semantic-shape review of 12 selected development queries, not a pre-registered benchmark, independent gold annotation, evidence-sufficiency audit or causal gain estimate.'}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps({key: value for key, value in result.items() if key != 'rows'},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
