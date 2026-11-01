"""Descriptive same-run mechanism, quality and full-cost ledger; no serving changes."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics

from omni_proof_runtime import file_digest, write_json


def distribution(values):
    return {'count': len(values), 'total': sum(values), 'mean': statistics.mean(values) if values else None,
            'median': statistics.median(values) if values else None,
            'min': min(values) if values else None, 'max': max(values) if values else None}


def bind_quality(rows, judgments, root):
    by_key = {}
    for judgment in judgments:
        key = (judgment['run'], judgment['question_id'])
        if key in by_key:
            raise ValueError('duplicate fixed-Judge row')
        if bool(judgment['parse_valid']) != (judgment['judge_label'] in {'CORRECT', 'WRONG'}):
            raise ValueError('fixed-Judge validity contract differs')
        by_key[key] = judgment
    bound, seen = [], set()
    for row in rows:
        key = (str(root / f'proof_user_{row["person_id"]}.json'), row['question_id'])
        if key in seen or key not in by_key:
            raise ValueError('serving/Judge denominator or identity differs')
        seen.add(key)
        judgment = by_key[key]
        if judgment['prediction_sha256'] != row['answer_sha256']:
            raise ValueError('fixed Judge prediction binding failed')
        bound.append((row, judgment))
    if seen != set(by_key):
        raise ValueError('Judge denominator differs')
    return bound


def summarize_locator_costs(manifest):
    shards = []
    for shard in manifest['locator_extraction_manifest']['shards']:
        status = shard['status']
        if not status.get('complete'):
            raise ValueError('reused locator extraction incomplete')
        shards.append({'directory': shard['directory'], 'images': status['total'],
            'elapsed_seconds': status['elapsed_seconds'],
            'load_and_identity_seconds': status['load_and_identity_seconds'],
            'batch_seconds': sum(batch['seconds'] for batch in status['batches'])})
    return {'shards': shards, 'images': manifest['locator_extraction_manifest']['images'],
            'shard_elapsed_seconds_sum': sum(shard['elapsed_seconds'] for shard in shards),
            'shard_elapsed_seconds_max': max((shard['elapsed_seconds'] for shard in shards), default=0.),
            'batch_seconds_sum': sum(shard['batch_seconds'] for shard in shards),
            'rerun_for_this_candidate': False,
            'boundary': 'historical extraction retained as an offline dependency; shard service sum and maximum are not current candidate wall time or amortized savings'}


def summarize_budgets(rows, protocol):
    failures = []
    for row in rows:
        render = row['proof_plan']['final_render']
        if (row['input_tokens'] != render['input_tokens']
                or row['input_tokens'] > protocol['input_tokens']
                or render['text_tokens'] > protocol['text_tokens']
                or len(row['images']) > protocol['max_images']):
            failures.append(row['question_id'])
    return {'passed': not failures, 'violating_question_ids': failures,
            'max_final_text_tokens': max((row['proof_plan']['final_render']['text_tokens'] for row in rows), default=0),
            'max_final_input_tokens': max((row['input_tokens'] for row in rows), default=0),
            'max_original_images': max((len(row['images']) for row in rows), default=0)}


def summarize_retained_domains(rows):
    violations, omitted = [], []
    for row in rows:
        plan = row['proof_plan']
        presented = set(plan['grounding']['alias_to_record_id'].values())
        selected = {identity for witness in plan['selected_witnesses'] for identity in witness['retrieval_record_ids']}
        if not selected <= presented:
            violations.append({'question_id': row['question_id'], 'unpresented_records': sorted(selected - presented)})
        omitted.append(len(plan['grounding']['omitted_whole_record_ids_due_to_context']))
    return {'selected_witnesses_with_unpresented_sources': violations,
            'omitted_demand_record_pairs': distribution(omitted),
            'boundary': 'context omissions count demand-record pairs; the same base record can be retained for another demand; nonproof packages are not restricted to the witness prompt domain'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output or root / 'scientific_review.json'
    if output.exists():
        raise ValueError('fresh scientific review output required; retain previous reports')
    cohort = json.loads((root / 'cohort.json').read_text())
    runs, rows, per_user = [], [], {}
    for user, expected in cohort['question_ids_by_user'].items():
        path = root / f'proof_user_{user}.json'
        run = json.loads(path.read_text())
        if not run.get('complete') or run['question_ids'] != expected:
            raise ValueError('complete original-order full denominator required')
        run_rows = run['rows']
        if [row['question_id'] for row in run_rows] != expected:
            raise ValueError('row denominator changed')
        per_user[user] = {'requests': len(run_rows), 'run_sha256': file_digest(path)}
        runs.append(run)
        rows.extend(run_rows)
    judge = json.loads((root / 'judge.json').read_text())
    closure = json.loads((root / 'closure.json').read_text())
    if not judge.get('complete') or not closure.get('complete'):
        raise ValueError('complete fixed Judge and full denominator audit required')
    bound = bind_quality(rows, judge['rows'], root)
    correct = sum(judgment['parse_valid'] and judgment['judge_label'] == 'CORRECT' for _, judgment in bound)
    invalid = sum(not judgment['parse_valid'] for _, judgment in bound)
    if closure['correct'] != correct or closure['invalid_judgments'] != invalid or closure['requests'] != len(rows):
        raise ValueError('closure quality counts differ from exact same-run Judge rows')
    for user, entry in per_user.items():
        selected = [(row, judgment) for row, judgment in bound if row['person_id'] == user]
        entry['correct'] = sum(judgment['parse_valid'] and judgment['judge_label'] == 'CORRECT'
                               for _, judgment in selected)
        entry['invalid_judgments'] = sum(not judgment['parse_valid'] for _, judgment in selected)
    strict = [row for row in rows if row['proof_plan']['selection_mode'] == 'source_located_proof_join']
    relation_queries = [row for row in rows if row['proof_plan']['relations']]
    indexes = [json.loads((root / f'proof_index_user_{user}/index.json').read_text()) for user in per_user]
    locator_manifest = json.loads((root / 'index_manifest.json').read_text())
    locator_costs = summarize_locator_costs(locator_manifest)
    protocol = json.loads((root / 'protocol.json').read_text())
    budgets = summarize_budgets(rows, protocol)
    domains = summarize_retained_domains(rows)
    if not budgets['passed'] or domains['selected_witnesses_with_unpresented_sources']:
        raise ValueError('final budget or retained witness domain audit failed')
    structural = json.loads((root / 'full_structural_audit.json').read_text())
    if not structural.get('structural_checks_passed') or structural['requests'] != len(rows):
        raise ValueError('complete structural audit required')
    for user, entry in per_user.items():
        if structural['run_sha256'][f'proof_user_{user}.json'] != entry['run_sha256']:
            raise ValueError('structural audit does not bind exact serving run')
    result = {'complete': True, 'candidate': protocol['candidate'],
        'requests': len(rows), 'correct': correct, 'invalid_judgments': invalid,
        'full_denominator_accuracy': correct / len(rows), 'per_user': per_user,
        'final_budget_audit': budgets, 'retained_witness_domain_audit': domains,
        'query_parse_status': dict(Counter(row['proof_plan']['query_parse']['status'] for row in rows)),
        'validated_demand_count_histogram': dict(Counter(len(row['proof_plan']['demands']) for row in rows
            if row['proof_plan']['query_parse']['status'] == 'ok')),
        'multi_demand_valid_queries': sum(row['proof_plan']['query_parse']['status'] == 'ok'
            and len(row['proof_plan']['demands']) > 1 for row in rows),
        'relation_queries': len(relation_queries),
        'raw_parser_proposed_relation_queries': sum(bool(row['proof_plan']['query_parse']['parsed'].get('relations'))
            for row in rows if isinstance(row['proof_plan']['query_parse']['parsed'], dict)),
        'source_located_relation_joins': sum(row in strict for row in relation_queries),
        'source_located_joins': len(strict), 'nonproof_packages': len(rows) - len(strict),
        'source_located_multi_demand_joins': sum(len(row['proof_plan']['demands']) > 1 for row in strict),
        'multi_record_join_requests': sum(len({identity for witness in row['proof_plan']['selected_witnesses']
            for identity in witness['retrieval_record_ids']}) > 1 for row in strict),
        'selected_unique_object_count': distribution([len(row['materialized_object_ids']) for row in rows]),
        'input_tokens': distribution([row['input_tokens'] for row in rows]),
        'image_count_histogram': dict(Counter(len(row['images']) for row in rows)),
        'output_tokens': distribution([len(row['answer_token_ids']) for row in rows]),
        'reader_stop_reasons': dict(Counter(row['stop_reason'] for row in rows)),
        'query_output_stop_reasons': dict(Counter(row['proof_plan']['query_parse'].get('stop_reason') for row in rows)),
        'witness_output_stop_reasons': dict(Counter(row['proof_plan']['grounding'].get('stop_reason') for row in rows)),
        'cost_seconds': {'new_index_construction': sum(index['construction_seconds'] for index in indexes),
            'model_and_index_startup_sum': sum(run['model_load_and_index_seconds'] for run in runs),
            'online_service': distribution([row['service_seconds'] for row in rows]),
            'query_parse_and_retrieval': sum(row['proof_plan']['retrieval']['query_vector_seconds'] for row in rows),
            'witness_generation_and_grounding': sum(row['execution_events'][1]['seconds'] for row in rows),
            'planning_and_materialization': sum(row['proof_plan']['planner_seconds'] for row in rows),
            'reader_prefill': sum(row['prefill_seconds'] for row in rows),
            'reader_decode': sum(row['decode_seconds'] for row in rows),
            'fixed_judge': sum(judgment['judge_seconds'] for _, judgment in bound)},
        'reused_visual_locator_sources': locator_costs,
        'full_structural_audit_sha256': file_digest(root / 'full_structural_audit.json'),
        'index_manifest_sha256': file_digest(root / 'index_manifest.json'),
        'protocol_sha256': file_digest(root / 'protocol.json'), 'judge_sha256': file_digest(root / 'judge.json'),
        'summary_script_sha256': file_digest(Path(__file__)),
        'baseline_or_ablation_launched': False, 'independent_validation': False,
        'semantic_entailment_proven': False, 'causal_e2e_speedup': None, 'physical_pim_speedup': None,
        'main_experiment_closed': False,
        'boundary': 'descriptive single development candidate; source-located joins are not semantic proof; no matched baseline or independent confirmation; previous quality scores are historical only'}
    write_json(output, result)
    print(json.dumps({key: result[key] for key in ('candidate', 'requests', 'correct', 'invalid_judgments',
        'multi_demand_valid_queries', 'relation_queries', 'source_located_joins', 'nonproof_packages')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
