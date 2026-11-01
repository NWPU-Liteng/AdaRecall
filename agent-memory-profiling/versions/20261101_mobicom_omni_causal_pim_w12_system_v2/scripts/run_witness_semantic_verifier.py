"""Run a cross-series verifier over retained witness branches without changing answers."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time

from omni_multi_witness import CANDIDATE_ONLY
from omni_proof_runtime import file_digest, parse_object, write_json
from omni_witness_verifier import (AUDIT_SCHEMA, VERIFIER_JSON_SCHEMA,
                                   VERIFIER_PROMPT, audit_verified_frontier,
                                   validate_verifier_result, verifier_payload)


def model_series(path):
    value = Path(path).name.lower()
    if 'qwen3___5' in value or 'qwen3.5' in value:
        return 'qwen3.5'
    if 'qwen3-14b' in value or 'qwen3_14b' in value:
        return 'qwen3-14b'
    return value


def resolve_model_series(path, model):
    guessed = model_series(path)
    if guessed != Path(path).name.lower():
        return guessed
    count = sum(parameter.numel() for parameter in model.parameters())
    if 12_000_000_000 <= count <= 16_000_000_000:
        return 'qwen3-14b'
    return guessed


def load_text_model(path, devices):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for the witness verifier')
    if not devices:
        raise ValueError('at least one verifier device is required')
    if len(devices) > torch.cuda.device_count():
        raise ValueError('requested verifier devices exceed the visible CUDA devices')
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    tokenizer.padding_side = 'left'
    max_memory = {index: '22GiB' for index in range(len(devices))}
    max_memory['cpu'] = '64GiB'
    model = AutoModelForCausalLM.from_pretrained(
        str(path), local_files_only=True, dtype=torch.bfloat16,
        attn_implementation='sdpa', device_map='auto', max_memory=max_memory,
        low_cpu_mem_usage=True).eval()
    return tokenizer, model, ','.join(f'cuda:{index}' for index in range(len(devices)))


def compile_verifier_grammar(model, tokenizer):
    import xgrammar as xgr

    stop_tokens = getattr(model.config, 'eos_token_id', tokenizer.eos_token_id)
    stop_tokens = stop_tokens if isinstance(stop_tokens, list) else [stop_tokens]
    info = xgr.TokenizerInfo.from_huggingface(
        tokenizer, vocab_size=int(model.config.vocab_size), stop_token_ids=stop_tokens)
    compiler = xgr.GrammarCompiler(info, max_threads=2)
    return compiler.compile_json_schema(
        VERIFIER_JSON_SCHEMA, strict_mode=True, separators=(',', ':'))


def generate_verifier_output(model, tokenizer, compiled_grammar, payload, *, max_tokens):
    import torch
    from xgrammar.contrib.hf import LogitsProcessor

    messages = [{'role': 'system', 'content': VERIFIER_PROMPT},
                {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer([prompt], return_tensors='pt', add_special_tokens=False)
    device = model.get_input_embeddings().weight.device
    inputs = inputs.to(device)
    stop_tokens = getattr(model.config, 'eos_token_id', tokenizer.eos_token_id)
    stop_tokens = stop_tokens if isinstance(stop_tokens, list) else [stop_tokens]
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **inputs, max_new_tokens=max_tokens, do_sample=False, num_beams=1,
            logits_processor=[LogitsProcessor(compiled_grammar)],
            eos_token_id=stop_tokens, pad_token_id=tokenizer.pad_token_id,
            use_cache=True)
    continuation = generated[0, inputs.input_ids.shape[1]:]
    answer = tokenizer.decode(continuation, skip_special_tokens=True).strip()
    try:
        parsed = parse_object(answer)
    except (ValueError, TypeError, json.JSONDecodeError):
        parsed = None
    return {
        'answer': answer,
        'parsed': parsed,
        'input_tokens': int(inputs.input_ids.shape[1]),
        'generated_tokens': int(continuation.shape[0]),
        'seconds': time.perf_counter() - started,
    }


def load_manifests(path):
    result = {}
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                result[row['question_id']] = row
    return result


def source_payload(record, objects):
    object_id = record.get('primary_object_id')
    source = objects.get(object_id)
    if not isinstance(source, dict) or source.get('kind') != 'text':
        raise ValueError(f'text branch source is missing: {record.get("record_id")}')
    times = source.get('time') or [None, None]
    return {
        'record_id': record['record_id'],
        'role': source.get('role'),
        'session_interval': f'{times[0]}..{times[1]}',
        'text': source.get('raw_text'),
    }


def record_session_time(record, objects):
    source = objects.get(record.get('primary_object_id')) or {}
    times = source.get('time')
    if not times:
        sources = source.get('sources') or []
        times = sources[0].get('time') if sources else None
    times = times or [None, None]
    return times[0], times[1]


def branch_from_descriptor(demand_id, record_id, family):
    return {
        'demand_id': demand_id,
        'record_ids': [record_id],
        'retrieval_record_ids': [record_id],
        'source_family': family,
        'verification_state': CANDIDATE_ONLY,
        'semantic_sufficiency': 'unverified',
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--descriptor-index', type=Path, required=True)
    parser.add_argument('--manifests', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--devices', default='6')
    parser.add_argument('--max-new-tokens', type=int, default=192)
    parser.add_argument('--limit-text-branches', type=int)
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
    if result_document.get('witness_descriptor_index_sha256') != file_digest(args.descriptor_index):
        raise ValueError('results and descriptor index identities differ')
    records = json.loads((args.index / 'records.json').read_text())
    records_by_id = {record['record_id']: record for record in records}
    source = json.loads((args.index / 'source.json').read_text())
    objects = source['objects']
    manifests = load_manifests(args.manifests)

    descriptor = json.loads(args.descriptor_index.read_text())
    descriptor_ids = {item['record_id'] for item in descriptor.get('descriptors', [])}
    if not descriptor_ids <= set(records_by_id):
        raise ValueError('descriptor index contains records outside the proof index')

    tasks = []
    branch_times = {}
    demand_times = {}
    skipped_visual = 0
    for row in result_document['rows']:
        question_id = row['question_id']
        if question_id not in manifests:
            raise ValueError(f'question manifest is missing: {question_id}')
        plan = row.get('proof_plan') or {}
        shadow = plan.get('witness_descriptor_shadow')
        if not shadow:
            raise ValueError(f'question has no witness descriptor shadow: {question_id}')
        demands = {item['demand_id']: item for item in plan.get('demands', [])}
        for frontier in shadow.get('per_demand', []):
            demand_id = frontier['demand_id']
            demand = demands.get(demand_id)
            if demand is None:
                raise ValueError(f'descriptor demand is absent from the proof plan: {demand_id}')
            branches = list(zip(frontier.get('records', []), frontier.get('families', [])))
            if len(branches) != len(frontier.get('records', [])):
                raise ValueError('descriptor families and records are not aligned')
            for record_id, family in branches:
                record = records_by_id.get(record_id)
                if record is None or record_id not in descriptor_ids:
                    raise ValueError(f'descriptor branch record is missing: {record_id}')
                session_start, session_end = record_session_time(record, objects)
                branch_times[(question_id, demand_id, record_id)] = {
                    'session_start': session_start,
                    'session_end': session_end,
                }
                demand_times.setdefault((question_id, demand_id), []).append(
                    (session_start, session_end))
                if record.get('kind') != 'source_text':
                    skipped_visual += 1
                    continue
                text = source_payload(record, objects)
                branch = branch_from_descriptor(demand_id, record_id, family)
                tasks.append({
                    'question_id': question_id,
                    'demand_id': demand_id,
                    'demand': demand,
                    'branch': branch,
                    'text_source': text,
                })
    if args.limit_text_branches is not None:
        if args.limit_text_branches < 1:
            raise ValueError('positive text-branch limit required')
        tasks = tasks[:args.limit_text_branches]
    if not tasks:
        raise ValueError('no text branches are available for verification')
    demand_scope = {}
    for key, times in demand_times.items():
        starts = [start for start, _ in times if start]
        ends = [end for _, end in times if end]
        demand_scope[key] = {
            'earliest_session_start': min(starts) if starts else None,
            'latest_session_end': max(ends) if ends else None,
        }
    for task in tasks:
        key = (task['question_id'], task['demand_id'], task['branch']['record_ids'][0])
        task['temporal_context'] = {
            **branch_times[key],
            **demand_scope[(task['question_id'], task['demand_id'])],
            'session_interval': task['text_source']['session_interval'],
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
        'device_count': len(devices),
        'tensor_dtype': 'bfloat16',
        'generation_backend': 'transformers.generate + xgrammar JSON grammar',
        'input_contract': 'one demand plus one source branch; original question withheld',
        'citation_contract': 'exact substring in the cited source record',
        'independence_boundary': ('different model series than the Reader; same vendor, '
                                  'not a formal independent-evaluator certificate'),
    }

    assessments, started = [], time.perf_counter()
    for offset, task in enumerate(tasks, 1):
        branch = task['branch']
        text_by_record = {task['text_source']['record_id']: task['text_source']['text']}
        payload = verifier_payload(task['demand'], branch, [task['text_source']])
        call = generate_verifier_output(
            model, tokenizer, compiled_grammar, payload,
            max_tokens=args.max_new_tokens)
        validation = validate_verifier_result(
            call.get('parsed'), branch, text_by_record, demand=task['demand'],
            temporal_context=task['temporal_context'])
        assessments.append({
            'question_id': task['question_id'],
            'demand_id': task['demand_id'],
            'record_id': task['branch']['record_ids'][0],
            'source_family': task['branch']['source_family'],
            'accepted': validation.get('accepted') is True,
            'error': validation.get('error'),
            'verdict': validation.get('verdict'),
            'reason_code': validation.get('reason_code'),
            'demand_operation': validation.get('demand_operation'),
            'temporal_context': task['temporal_context'],
            'evidence': validation.get('evidence', []),
            'raw_output': call.get('answer'),
            'input_tokens': call.get('input_tokens'),
            'generated_tokens': call.get('generated_tokens'),
            'verifier_seconds': call.get('seconds'),
        })
        print(f'[{offset}/{len(tasks)}] {task["question_id"]}/{task["demand_id"]} '
              f'{task["branch"]["record_ids"][0]} -> '
              f'{validation.get("verdict")}/{validation.get("reason_code")} '
              f'accepted={validation.get("accepted")}', file=sys.stderr, flush=True)

    grouped = {}
    for item in assessments:
        grouped.setdefault((item['question_id'], item['demand_id']), []).append(item)
    audit = audit_verified_frontier(grouped, verifier_metadata=metadata)
    reason_counts = Counter(item.get('reason_code') or item.get('error') for item in assessments)
    document = {
        'schema': AUDIT_SCHEMA,
        'source_results_sha256': file_digest(args.results),
        'source_index_sha256': file_digest(args.index / 'index.json'),
        'descriptor_index_sha256': file_digest(args.descriptor_index),
        'question_count': len({item['question_id'] for item in assessments}),
        'text_branch_count': len(tasks),
        'visual_branch_skipped_count': skipped_visual,
        'accepted_text_branch_count': sum(item['accepted'] for item in assessments),
        'rejected_text_branch_count': sum(not item['accepted'] for item in assessments),
        'reason_counts': dict(sorted(reason_counts.items())),
        'seconds': time.perf_counter() - started,
        'verifier_metadata': metadata,
        'audit': audit,
        'manual_audit_complete': False,
        'answer_path_changed': False,
        'grounding_input_replaced': False,
        'assessments': assessments,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, document)
    print(json.dumps({
        'text_branch_count': len(tasks),
        'visual_branch_skipped_count': skipped_visual,
        'accepted_text_branch_count': document['accepted_text_branch_count'],
        'demands_with_semantically_verified_or': audit[
            'demands_with_semantically_verified_or'],
        'seconds': document['seconds'],
    }, ensure_ascii=False), file=sys.stderr)
    return document


if __name__ == '__main__':
    main()
