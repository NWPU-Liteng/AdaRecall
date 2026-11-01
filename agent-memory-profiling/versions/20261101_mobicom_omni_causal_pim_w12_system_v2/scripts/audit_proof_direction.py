"""Read-only direction audit of a frozen development candidate, not a new trial."""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selected_jsonl(path, identities):
    selected = {}
    pattern = re.compile(r'"question_id"\s*:\s*"([^"\\]+)"')
    with path.open() as handle:
        for line in handle:
            match = pattern.search(line)
            if not match or match.group(1) not in identities:
                continue
            row = json.loads(line)
            if row['question_id'] in selected:
                raise ValueError('duplicate selected question: ' + row['question_id'])
            selected[row['question_id']] = row
    if set(selected) != set(identities):
        raise ValueError('selected development questions missing from ' + str(path))
    return selected


def call_order(path):
    tree = ast.parse(path.read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
    names = {'greedy', 'compile_lifecycle', 'choose_backend'}
    calls = {name: sorted(node.lineno for node in ast.walk(main)
                         if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                         and node.func.id == name) for name in names}
    if any(len(lines) != 1 for lines in calls.values()):
        raise ValueError('execution order audit requires one explicit call per stage')
    return {'call_lines': calls,
            'lifecycle_compiled_after_generation': calls['compile_lifecycle'][0] > calls['greedy'][0],
            'backend_chosen_after_generation': calls['choose_backend'][0] > calls['greedy'][0],
            'boundary': 'statement-order observation in this frozen implementation, not a general data-flow proof'}


def repeat_counts(rows):
    seen_objects, seen_images, seen_inputs = set(), set(), set()
    occurrences = repeats = image_occurrences = image_repeats = input_repeats = 0
    for row in rows:
        objects = set(row['materialized_object_ids'])
        images = {image['image_sha256'] for image in row['images']}
        occurrences += len(objects)
        repeats += len(objects & seen_objects)
        image_occurrences += len(images)
        image_repeats += len(images & seen_images)
        input_hash = row['proof_plan']['final_render']['processor_input_sha256']
        input_repeats += input_hash in seen_inputs
        seen_objects.update(objects)
        seen_images.update(images)
        seen_inputs.add(input_hash)
    return {'requests': len(rows), 'source_object_occurrences': occurrences,
            'unique_source_objects': len(seen_objects), 'repeated_source_object_occurrences': repeats,
            'raw_image_occurrences': image_occurrences, 'unique_raw_images': len(seen_images),
            'repeated_raw_image_occurrences': image_repeats,
            'repeated_final_input_token_hashes': input_repeats,
            'qualified_reader_state_hits': None,
            'boundary': 'unbounded source identity repetition; neither tensor/position/state equivalence nor net compute saving is established'}


def optimistic_stage_bound(total, stage):
    if total <= 0 or not 0 <= stage < total:
        raise ValueError('nonnegative stage strictly below positive total required')
    return {'observed_stage_fraction': stage / total,
            'zero_time_stage_service_ratio_upper_bound': total / (total - stage),
            'boundary': 'arithmetic envelope on these shared-server observations, not a measured speedup or a target-device prediction'}


def audit(root, prepared):
    protocol = json.loads((root / 'protocol.json').read_text())
    scientific = json.loads((root / 'scientific_review.json').read_text())
    cohort = json.loads((root / 'cohort.json').read_text())
    audit_metadata = json.loads((prepared / 'audit.json').read_text())
    split = json.loads((prepared / 'split.json').read_text())
    user_ids = set(cohort['question_ids_by_user'])
    if not user_ids <= set(split['development_people']):
        raise ValueError('only registered development users may be audited')
    if file_hash(prepared / 'audit.json') != protocol['prepared_audit_sha256']:
        raise ValueError('prepared audit identity mismatch')
    if file_hash(root / 'protocol.json') != scientific['protocol_sha256']:
        raise ValueError('protocol identity mismatch')
    if file_hash(root / 'cohort.json') != protocol['cohort_sha256']:
        raise ValueError('cohort identity mismatch')
    for relative, expected in protocol['source_files_sha256'].items():
        if file_hash(root / 'runtime_mainline' / relative) != expected:
            raise ValueError('frozen source changed: ' + relative)
    for filename in ['labels.jsonl', 'manifests.jsonl']:
        if file_hash(prepared / filename) != audit_metadata['artifacts_sha256'][filename]:
            raise ValueError('prepared file identity mismatch: ' + filename)
    identities = [identity for user in sorted(user_ids)
                  for identity in cohort['question_ids_by_user'][user]]
    if len(identities) != len(set(identities)):
        raise ValueError('cohort question identities are not unique')
    manifests = selected_jsonl(prepared / 'manifests.jsonl', identities)
    labels = selected_jsonl(prepared / 'labels.jsonl', identities)
    if any(str(item['person_id']) not in user_ids for item in [*manifests.values(), *labels.values()]):
        raise ValueError('non-development row was selected')
    if file_hash(root / 'judge.json') != scientific['judge_sha256']:
        raise ValueError('Judge identity mismatch')
    judgments = json.loads((root / 'judge.json').read_text())
    if not judgments.get('complete'):
        raise ValueError('complete original Judge required')
    judge_rows = {row['question_id']: row for row in judgments['rows']}
    if len(judge_rows) != len(judgments['rows']) or set(judge_rows) != set(identities):
        raise ValueError('Judge denominator mismatch')
    all_rows, streams, shard_streams = [], {}, {}
    for user in sorted(user_ids):
        filename = f'proof_user_{user}.json'
        run = json.loads((root / filename).read_text())
        if file_hash(root / filename) != scientific['per_user'][user]['run_sha256']:
            raise ValueError('run identity mismatch: ' + filename)
        if not run.get('complete') or run.get('evaluation_label_files_read'):
            raise ValueError('complete label-free original serving run required')
        if [row['question_id'] for row in run['rows']] != cohort['question_ids_by_user'][user]:
            raise ValueError('original user order mismatch')
        streams[user] = repeat_counts(run['rows'])
        all_rows.extend(run['rows'])
        observed = []
        for entry in run['compute_shards']:
            shard_path = root / Path(entry['path']).name
            if file_hash(shard_path) != entry['sha256']:
                raise ValueError('shard identity mismatch')
            shard = json.loads(shard_path.read_text())
            shard_streams[shard_path.name] = repeat_counts(shard['rows'])
            observed.extend(row['question_id'] for row in shard['rows'])
        if Counter(observed) != Counter(cohort['question_ids_by_user'][user]):
            raise ValueError('actual compute shard denominator mismatch')
    quality = defaultdict(lambda: {'requests': 0, 'correct': 0, 'multi_demand': 0, 'with_image': 0})
    domain_sizes, shared_references, consumer_counts, recompute = Counter(), [], Counter(), Counter()
    details, timing, output_tokens = [], Counter(), Counter()
    for row in all_rows:
        identity, plan = row['question_id'], row['proof_plan']
        judgment = judge_rows[identity]
        if hashlib.sha256(row['answer'].encode()).hexdigest() != row['answer_sha256']:
            raise ValueError('answer hash mismatch')
        if judgment['prediction_sha256'] != row['answer_sha256']:
            raise ValueError('Judge/answer binding mismatch')
        if hashlib.sha256(manifests[identity]['question'].encode()).hexdigest() != row['question_sha256']:
            raise ValueError('question identity mismatch')
        group = quality[labels[identity]['question_type']]
        group['requests'] += 1
        group['correct'] += judgment['judge_label'] == 'CORRECT'
        group['multi_demand'] += len(plan['demands']) > 1 and plan['query_parse']['status'] == 'ok'
        group['with_image'] += bool(row['images'])
        domains = Counter(witness['demand_id'] for witness in plan['proof_hypergraph']['witnesses'])
        domain_sizes.update(domains.values())
        selected = plan['selected_witnesses']
        references = sum(len(set(witness['object_ids'])) for witness in selected)
        union = {object_id for witness in selected for object_id in witness['object_ids']}
        shared_references.append(references - len(union))
        for state in row['lifecycle_plan']['states']:
            consumer_counts[len(state['consumers'])] += 1
            recompute[state['recompute']] += 1
        timing['online_service'] += row['service_seconds']
        timing['reader_prefill'] += row['prefill_seconds']
        timing['reader_decode'] += row['decode_seconds']
        for event in row['execution_events']:
            if event['phase'] in ['query_parse_and_retrieval', 'source_grounding', 'proof_materialize']:
                timing[event['phase']] += event['seconds']
        output_tokens['parser'] += plan['query_parse']['output_token_count']
        output_tokens['witness'] += plan['grounding']['output_token_count']
        output_tokens['answer'] += len(row['answer_token_ids'])
        details.append({'question_id': identity, 'question': manifests[identity]['question'],
                        'type_evaluation_only': labels[identity]['question_type'],
                        'demands': plan['demands'], 'relations': plan['relations'],
                        'parse_status': plan['query_parse']['status'],
                        'selection_mode': plan['selection_mode'], 'witness_domain_sizes': dict(domains),
                        'selected_witnesses': len(selected), 'shared_object_reference_occurrences': references - len(union),
                        'source_objects': len(row['materialized_object_ids']), 'images': len(row['images']),
                        'input_tokens': row['input_tokens'], 'final_text_tokens': plan['final_render']['text_tokens'],
                        'chosen_backend': row['backend_plan']['chosen_backend'],
                        'correct_proxy_judge': judgment['judge_label'] == 'CORRECT'})
    total = timing['online_service']
    orders = call_order(root / 'runtime_mainline/scripts/run_proof_mainline.py')
    joint_requests = sum(bool(row['backend_plan']['candidate_route']['enabled'])
                         and not orders['backend_chosen_after_generation']
                         and not orders['lifecycle_compiled_after_generation'] for row in all_rows)
    return {'complete': True, 'audit_kind': 'immutable_development_opportunity_and_direction',
            'candidate': root.name, 'requests': len(all_rows),
            'correct': sum(group['correct'] for group in quality.values()),
            'quality_by_type_evaluation_only': dict(sorted(quality.items())),
            'frozen_source_files_verified': len(protocol['source_files_sha256']),
            'execution_order': orders,
            'accepted_nonempty_witness_domain_size_histogram': dict(sorted(domain_sizes.items())),
            'requests_with_actual_object_sharing_between_selected_witnesses': sum(value > 0 for value in shared_references),
            'shared_object_reference_occurrences': sum(shared_references),
            'state_consumer_count_histogram': dict(consumer_counts), 'recompute_decisions': dict(recompute),
            'backend_histogram': dict(Counter(row['backend_plan']['chosen_backend'] for row in all_rows)),
            'joint_execution_decision_requests': joint_requests,
            'original_user_order_source_repetition': streams,
            'actual_compute_shard_source_repetition': shard_streams,
            'cost_seconds': dict(timing), 'generated_tokens_by_phase': dict(output_tokens),
            'optimistic_stage_envelopes': {key: optimistic_stage_bound(total, timing[key])
                                          for key in ['reader_prefill', 'reader_decode']},
            'query_release_timestamps_present': False,
            'visibility_protocol': audit_metadata['visibility_protocol'],
            'recommendation': {'primary_research_candidate': 'source-constrained evidence compilation; semantic effectiveness still unproven',
                               'lifecycle': 'correctness/accounting support; no independent reuse or joint-scheduling claim',
                               'pim': 'conditional auxiliary feasibility only; calibration/capacity/numerics unknown',
                               'next_trial': 'fact-versus-derived query contract diagnostic before any full regeneration'},
            'rows': details,
            'input_sha256': {str(path): file_hash(path) for path in
                             [root / 'protocol.json', root / 'scientific_review.json', root / 'judge.json',
                              root / 'cohort.json', prepared / 'audit.json', prepared / 'labels.jsonl']},
            'evaluation_label_files_read': [str(prepared / 'labels.jsonl')],
            'label_access_scope': 'only registered development question rows decoded; labels are post-hoc diagnostics, never serving inputs',
            'new_generation_or_judge_launched': False, 'baseline_or_ablation_launched': False,
            'validation_accessed': False, 'holdout_accessed': False,
            'semantic_effectiveness_established': False, 'main_experiment_closed': False,
            'boundary': 'source repetition and post-hoc groups are not cache qualifications, causal quality gains, physical PIM benefits or a natural arrival trace'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('fresh audit output required')
    result = audit(args.root, args.prepared)
    with args.output.open('x') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps({key: value for key, value in result.items() if key not in
                     ['rows', 'actual_compute_shard_source_repetition', 'input_sha256']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
