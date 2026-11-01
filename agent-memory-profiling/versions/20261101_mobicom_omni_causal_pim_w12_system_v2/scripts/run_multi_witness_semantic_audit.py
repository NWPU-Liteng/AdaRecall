"""Verify 4B-proposed source-located witness branches with the 14B audit stack."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time

from omni_multi_witness import branch_source_families, has_independent_branch_pair
from omni_proof_runtime import file_digest, write_json
from omni_witness_verifier import (SCHEMA, audit_verified_frontier,
                                   validate_verifier_result, verifier_payload)
from run_witness_semantic_verifier import (compile_verifier_grammar,
                                           generate_verifier_output, load_manifests,
                                           load_text_model, model_series,
                                           record_session_time, resolve_model_series,
                                           source_payload)


def branch_temporal_context(records, objects):
    starts, ends, intervals = [], [], []
    for record in records:
        start, end = record_session_time(record, objects)
        if start:
            starts.append(start)
        if end:
            ends.append(end)
        if start or end:
            intervals.append(f'{start}..{end}')
    return {
        'session_start': min(starts) if starts else None,
        'session_end': max(ends) if ends else None,
        'session_interval': ' | '.join(intervals),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--manifests', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--devices', default='5,6')
    parser.add_argument('--max-new-tokens', type=int, default=224)
    args = parser.parse_args(argv)

    devices = [int(value) for value in args.devices.split(',') if value.strip()]
    if not devices or len(set(devices)) != len(devices) or any(value < 0 for value in devices):
        raise ValueError('unique nonnegative verifier devices are required')
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(value) for value in devices)

    result_document = json.loads(args.results.read_text())
    index_meta = json.loads((args.index / 'index.json').read_text())
    if index_meta.get('complete') is not True or index_meta.get('schema') != 'proof-index-v2':
        raise ValueError('complete proof-index-v2 is required')
    if result_document.get('index_sha256') != file_digest(args.index / 'index.json'):
        raise ValueError('results and proof index identities differ')
    records_by_id = {record['record_id']: record
                     for record in json.loads((args.index / 'records.json').read_text())}
    objects = json.loads((args.index / 'source.json').read_text())['objects']
    manifests = load_manifests(args.manifests)

    tasks, visual_skipped = [], 0
    branch_times, demand_times = {}, {}
    for row in result_document['rows']:
        question_id = row['question_id']
        if question_id not in manifests:
            raise ValueError(f'question manifest is missing: {question_id}')
        plan = row.get('proof_plan') or {}
        shadow = plan.get('multi_witness_shadow') or {}
        demands = {item['demand_id']: item for item in plan.get('demands', [])}
        for witness in shadow.get('witnesses') or []:
            demand_id = witness.get('demand_id')
            demand = demands.get(demand_id)
            if demand is None:
                raise ValueError(f'witness demand is absent from the proof plan: {demand_id}')
            record_ids = witness.get('retrieval_record_ids') or witness.get('record_ids') or []
            records = [records_by_id.get(record_id) for record_id in record_ids]
            if not record_ids or any(record is None for record in records):
                raise ValueError('witness branch refers to an unknown source record')
            branch = {
                'witness_id': witness.get('witness_id'),
                'demand_id': demand_id,
                'record_ids': list(record_ids),
                'retrieval_record_ids': list(record_ids),
                'verification_state': witness.get('verification_state', 'source_located'),
            }
            context = branch_temporal_context(records, objects)
            branch_times[(question_id, demand_id, branch['witness_id'])] = context
            demand_times.setdefault((question_id, demand_id), []).append(context)
            if any(record.get('kind') != 'source_text' for record in records):
                visual_skipped += 1
                continue
            text_sources = [source_payload(record, objects) for record in records]
            tasks.append({
                'question_id': question_id,
                'demand_id': demand_id,
                'demand': demand,
                'branch': branch,
                'text_sources': text_sources,
            })
    if not tasks:
        raise ValueError('no fully text witness branches are available')

    demand_scope = {}
    for key, contexts in demand_times.items():
        starts = [item['session_start'] for item in contexts if item['session_start']]
        ends = [item['session_end'] for item in contexts if item['session_end']]
        demand_scope[key] = {
            'earliest_session_start': min(starts) if starts else None,
            'latest_session_end': max(ends) if ends else None,
        }
    for task in tasks:
        key = (task['question_id'], task['demand_id'], task['branch']['witness_id'])
        task['temporal_context'] = {
            **branch_times[key],
            **demand_scope[(task['question_id'], task['demand_id'])],
        }

    tokenizer, model, device = load_text_model(args.model, devices)
    compiled_grammar = compile_verifier_grammar(model, tokenizer)
    verifier_series = resolve_model_series(args.model, model)
    reader_series = model_series(result_document.get('reader_model', ''))
    metadata = {
        'model': str(args.model),
        'model_series': verifier_series,
        'reader_model': result_document.get('reader_model'),
        'reader_series': reader_series,
        'cross_series': verifier_series != reader_series,
        'device': device,
        'physical_devices': devices,
        'visible_device_map': {str(index): value for index, value in enumerate(devices)},
        'tensor_dtype': 'bfloat16',
        'generation_backend': 'transformers.generate + xgrammar JSON grammar',
        'input_contract': 'one demand plus one multi-record source-located witness branch',
        'citation_contract': 'exact substring in one cited branch source',
        'independence_boundary': ('different model series than the Reader; same vendor, '
                                  'not a formal independent-evaluator certificate'),
    }

    assessments, started = [], time.perf_counter()
    for offset, task in enumerate(tasks, 1):
        branch = task['branch']
        text_by_record = {item['record_id']: item['text'] for item in task['text_sources']}
        payload = verifier_payload(task['demand'], branch, task['text_sources'])
        call = generate_verifier_output(
            model, tokenizer, compiled_grammar, payload, max_tokens=args.max_new_tokens)
        parsed = call.get('parsed') if isinstance(call.get('parsed'), dict) else {}
        validation = validate_verifier_result(
            call.get('parsed'), branch, text_by_record, demand=task['demand'],
            temporal_context=task['temporal_context'])
        families = sorted(branch_source_families(branch))
        assessments.append({
            'question_id': task['question_id'],
            'demand_id': task['demand_id'],
            'witness_id': branch['witness_id'],
            'record_ids': branch['record_ids'],
            'source_families': families,
            'accepted': validation.get('accepted') is True,
            'error': validation.get('error'),
            'verdict': validation.get('verdict'),
            'reason_code': validation.get('reason_code'),
            'demand_operation': validation.get('demand_operation'),
            'support_mode': parsed.get('support_mode'),
            'shared_property': parsed.get('shared_property'),
            'temporal_context': task['temporal_context'],
            'evidence': validation.get('evidence', []),
            'raw_output': call.get('answer'),
            'input_tokens': call.get('input_tokens'),
            'generated_tokens': call.get('generated_tokens'),
            'verifier_seconds': call.get('seconds'),
        })
        print(f'[{offset}/{len(tasks)}] {task["question_id"]}/{task["demand_id"]} '
              f'{branch["witness_id"]} -> {validation.get("verdict")}/'
              f'{validation.get("reason_code")} accepted={validation.get("accepted")}',
              file=sys.stderr, flush=True)

    grouped, audit_rows = {}, []
    for item in assessments:
        grouped.setdefault((item['question_id'], item['demand_id']), []).append(item)
    for key in sorted(grouped, key=str):
        branches = grouped[key]
        accepted = [{'retrieval_record_ids': item['record_ids'],
                     'witness_id': item['witness_id']}
                    for item in branches if item['accepted']]
        independent_pair = (has_independent_branch_pair(accepted)
                            if len(accepted) >= 2 else False)
        audit_rows.append({
            'question_id': key[0],
            'demand_id': key[1],
            'branch_count': len(branches),
            'accepted_branch_count': len(accepted),
            'semantically_verified_or': independent_pair,
            'reason_counts': dict(sorted(Counter(
                item.get('reason_code') or item.get('error') for item in branches).items())),
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    document = {
        'schema': SCHEMA,
        'source_results_sha256': file_digest(args.results),
        'source_index_sha256': file_digest(args.index / 'index.json'),
        'text_branch_count': len(tasks),
        'visual_branch_skipped_count': visual_skipped,
        'accepted_text_branch_count': sum(item['accepted'] for item in assessments),
        'demands_with_semantically_verified_or': sum(
            item['semantically_verified_or'] for item in audit_rows),
        'seconds': time.perf_counter() - started,
        'verifier_metadata': metadata,
        'per_demand': audit_rows,
        'assessments': assessments,
        'manual_audit_complete': False,
        'answer_path_changed': False,
        'grounding_input_replaced': False,
    }
    write_json(args.output, document)
    print(json.dumps({
        'text_branch_count': len(tasks),
        'visual_branch_skipped_count': visual_skipped,
        'accepted_text_branch_count': document['accepted_text_branch_count'],
        'demands_with_semantically_verified_or': document[
            'demands_with_semantically_verified_or'],
        'seconds': document['seconds'],
    }, ensure_ascii=False), file=sys.stderr)
    return document


if __name__ == '__main__':
    main()
