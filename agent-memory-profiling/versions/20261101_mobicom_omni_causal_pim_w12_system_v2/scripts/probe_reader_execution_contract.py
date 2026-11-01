"""Diagnose numerical partition effects without modifying the serving contract."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prefix_state import PrefixStateSnapshot
from reader_single_pass import greedy_decode
from run_mobilemem_text import READER, SYSTEM, atom_text, atomize, file_hash, model_hashes


def compare_outputs(reference, candidate):
    expected, actual = reference['token_ids'], candidate['token_ids']
    divergence = next((index for index, pair in enumerate(zip(expected, actual))
                       if pair[0] != pair[1]), min(len(expected), len(actual)))
    return {'full_greedy_token_ids_equal': expected == actual,
            'stop_reason_equal': reference['stop_reason'] == candidate['stop_reason'],
            'first_divergence': None if expected == actual else divergence}


def prepare_case(processor, case):
    memory = ''.join(atom_text(atom) for atom in case['atoms'])
    first_event = case['atoms'][0]['event_id']
    prefix_memory = ''.join(atom_text(atom) for atom in case['atoms'] if atom['event_id'] == first_event)

    def render(evidence, tail):
        return processor.apply_chat_template(
            [{'role': 'system', 'content': SYSTEM},
             {'role': 'user', 'content': f'原始记忆证据：\n{evidence}\n' + tail}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)

    tail = f"查询时间：{case['timestamp']}\n问题：{case['question']}"
    inputs = processor(text=[render(memory, tail)], return_tensors='pt')
    probe = processor(text=[render(prefix_memory, 'MEMORY_CHECKPOINT_BOUNDARY')], return_tensors='pt')
    tokens, prefix = inputs.input_ids[0].tolist(), probe.input_ids[0].tolist()
    boundary = next((index for index, pair in enumerate(zip(tokens, prefix)) if pair[0] != pair[1]),
                    min(len(tokens), len(prefix)))
    boundary = min(boundary, len(tokens) - 1)
    if boundary < 1:
        raise ValueError('empty checkpoint boundary')
    return inputs, boundary


def synthetic_cases():
    cases = []
    for count in (8, 32, 64):
        atoms = []
        for index in range(count):
            content = (f'记录编号{index}：设备检查于{index % 24:02d}:15完成。'
                       f'电池余量{100 - index % 71}%，采样温度{20 + index % 9}度。'
                       '测量来自设备日志，未启动数据上传。下次检查前保留本地结果。')
            atoms.append({'event_id': f'synthetic_{index}', 'start': 0, 'end': len(content),
                          'content': content, 'timestamp': '2026-01-01T12:00:00+00:00',
                          'source': 'synthetic-device', 'role': 'user'})
        cases.append({'identity': f'synthetic_history_{count}', 'atoms': atoms,
                      'timestamp': '2026-01-02T00:00:00+00:00',
                      'question': '按记录编号顺序列出最先三条与最后三条的电池余量和温度，保留编号，不加建议。'})
    return cases


def replay_cases(result_path, prepared, count, question_ids, processor):
    result = json.loads(result_path.read_text())
    if not result.get('complete') or len(result['rows']) != len(result['selected_question_ids']):
        raise ValueError('complete registered execution required')
    if [row['question_id'] for row in result['rows']] != result['selected_question_ids']:
        raise ValueError('registration mismatch')
    audit_path = prepared / 'audit.json'
    if file_hash(audit_path) != result['provenance']['prepared_audit_sha256']:
        raise ValueError('prepared audit mismatch')
    for name, expected in json.loads(audit_path.read_text())['artifacts_sha256'].items():
        if file_hash(prepared / name) != expected:
            raise ValueError(f'prepared artifact changed: {name}')
    if model_hashes(READER) != result['provenance']['reader_weights']:
        raise ValueError('reader weights changed')
    manifests = {row['question_id']: row for row in map(json.loads,
                 (prepared / 'manifests.jsonl').read_text().splitlines())}
    graphs = json.loads((prepared / 'mobilemem_graph.json').read_text())
    atom_maps = {}
    cases = []
    if question_ids:
        by_id = {row['question_id']: row for row in result['rows']}
        if any(identity not in by_id for identity in question_ids):
            raise ValueError('requested replay ID is absent from the registered run')
        selected_rows = [by_id[identity] for identity in question_ids]
    else:
        selected_rows = result['rows'][:count]
    for row in selected_rows:
        question = manifests[row['question_id']]
        if question['split'] != 'development' or row['status'] != 'ok':
            raise ValueError('only successful development executions can be replayed')
        person = question['person_id']
        if person not in atom_maps:
            graph = next(graph for graph in graphs if graph['person_id'] == person)
            atoms, _ = atomize(graph, processor.tokenizer)
            atom_maps[person] = {atom['identity']: atom for atom in atoms}
        chosen = [atom_maps[person][atom['identity']] for atom in row['selected_atoms']]
        if not chosen or any(atom['timestamp'] > question['effective_timestamp'] for atom in chosen):
            raise ValueError('nonempty temporally valid source evidence required')
        cases.append({'identity': row['question_id'], 'atoms': chosen,
                      'timestamp': question['effective_timestamp'], 'question': question['question']})
    return cases


def run_case(model, processor, case, args):
    inputs, boundary = prepare_case(processor, case)
    inputs = inputs.to(args.device)
    snapshot = None
    first_logits = {}
    modes = {}
    with torch.inference_mode():
        for mode in ('unsplit', 'split_cold', 'snapshot_restored', 'split_repeat'):
            model.model.rope_deltas = None
            torch.cuda.synchronize(args.device)
            started = time.perf_counter()
            if mode == 'unsplit':
                output = model(**inputs, use_cache=True, logits_to_keep=1)
            elif mode == 'snapshot_restored':
                cache, rope = snapshot.restore(args.device)
                model.model.rope_deltas = rope
                output = model(input_ids=inputs.input_ids[:, boundary:], past_key_values=cache,
                               use_cache=True, logits_to_keep=1)
            else:
                prefix_inputs = {name: value[:, :boundary] for name, value in inputs.items()
                                 if name in {'input_ids', 'attention_mask', 'mm_token_type_ids'}}
                prefix = model(**prefix_inputs, use_cache=True, logits_to_keep=1)
                if mode == 'split_cold':
                    snapshot = PrefixStateSnapshot(prefix.past_key_values, prefix_inputs['input_ids'],
                                                   getattr(model.model, 'rope_deltas', None),
                                                   capacity_bytes=2 * 1024**3)
                output = model(input_ids=inputs.input_ids[:, boundary:], past_key_values=prefix.past_key_values,
                               use_cache=True, logits_to_keep=1)
                del prefix
            torch.cuda.synchronize(args.device)
            prefill_seconds = time.perf_counter() - started
            first_logits[mode] = output.logits[:, -1, :].float().cpu().clone()
            answer, generated, stop = greedy_decode(model, output, processor.tokenizer,
                                                     args.gen_tokens, args.device)
            torch.cuda.synchronize(args.device)
            modes[mode] = {'token_ids': generated, 'stop_reason': stop, 'answer': answer,
                           'prefill_seconds': prefill_seconds, 'total_seconds': time.perf_counter() - started}
            del output
    comparisons = {}
    for reference, candidate in (('unsplit', 'split_cold'), ('split_cold', 'snapshot_restored'),
                                  ('split_cold', 'split_repeat')):
        comparisons[f'{reference}_vs_{candidate}'] = {
            **compare_outputs(modes[reference], modes[candidate]),
            'first_logits_equal': torch.equal(first_logits[reference], first_logits[candidate]),
            'first_logits_max_absolute_difference': float(
                (first_logits[reference] - first_logits[candidate]).abs().max())}
    return {'identity': case['identity'], 'prompt_tokens': int(inputs.input_ids.shape[1]),
            'prefix_tokens': boundary, 'snapshot_bytes': snapshot.bytes,
            'modes': modes, 'comparisons': comparisons,
            'boundary': 'greedy outputs are only compared between execution modes; split_cold is not an oracle'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--result', type=Path)
    parser.add_argument('--prepared', type=Path)
    parser.add_argument('--replay-count', type=int, default=4)
    parser.add_argument('--replay-question-id', action='append', default=[])
    parser.add_argument('--gen-tokens', type=int, default=256)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.output.exists() or args.replay_count < 1 or args.gen_tokens < 1:
        raise ValueError('fresh output and positive limits required')
    if bool(args.result) != bool(args.prepared):
        raise ValueError('result and prepared must be supplied together')
    torch.set_num_threads(2)
    from transformers import AutoModelForImageTextToText, AutoProcessor
    import transformers
    frozen = {name: file_hash(Path(__file__).parent / name) for name in
              (Path(__file__).name, 'reader_single_pass.py', 'prefix_state.py', 'run_mobilemem_text.py')}
    processor = AutoProcessor.from_pretrained(str(READER), local_files_only=True)
    cases = synthetic_cases()
    if args.result:
        cases += replay_cases(args.result, args.prepared, args.replay_count,
                              args.replay_question_id, processor)
    provenance = {'reader_weights': model_hashes(READER), 'code_sha256': frozen,
                  'torch': torch.__version__, 'transformers': transformers.__version__,
                  'device': torch.cuda.get_device_name(args.device), 'dtype': 'bfloat16',
                  'attention': 'sdpa', 'gen_tokens': args.gen_tokens,
                  'source_result_sha256': file_hash(args.result) if args.result else None}
    model = AutoModelForImageTextToText.from_pretrained(str(READER), local_files_only=True,
             dtype=torch.bfloat16, attn_implementation='sdpa').to(args.device).eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {'complete': False, 'provenance': provenance, 'registered_cases': [case['identity'] for case in cases],
               'rows': [], 'serving_contract_changed': False, 'publication_ready': False}
    for case in cases:
        row = run_case(model, processor, case, args)
        payload['rows'].append(row)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        print(json.dumps({'case': row['identity'], 'comparisons': row['comparisons']},
                         ensure_ascii=False), flush=True)
    if any(file_hash(Path(__file__).parent / name) != expected for name, expected in frozen.items()):
        raise ValueError('probe source changed while running')
    payload['complete'] = True
    payload['boundary'] = 'diagnostic path equivalence only; no replacement of unsplit greedy audit, no quality or device speedup claim'
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
