"""Unsplit/split/restored numerical qualification on one label-free natural prefix."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import zipfile

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from audit_closure_prefixes import longest_prefix, reconstruct
from prefix_state import PrefixStateSnapshot


def file_digest(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            result.update(block)
    return result.hexdigest()


def decode(reader, output, limit, stop_ids):
    result = []
    first = output.logits[:, -1, :].detach().float().cpu()
    for index in range(limit):
        token = output.logits[:, -1, :].argmax(-1, keepdim=True)
        result.append(int(token.item()))
        if result[-1] in stop_ids:
            return result, 'eos', first
        if index + 1 < limit:
            output = reader(input_ids=token, past_key_values=output.past_key_values,
                            use_cache=True, logits_to_keep=1)
    return result, 'max_new_tokens', first


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-new-tokens', type=int, default=32)
    args = parser.parse_args()
    if args.output.exists() or args.max_new_tokens < 1:
        raise ValueError('fresh output and positive qualification length required')
    run = json.loads(args.run.read_text())
    if not run.get('complete'):
        raise ValueError('complete serving reference required')
    audit = json.loads((args.prepared / 'audit.json').read_text())
    if file_digest(args.prepared / 'audit.json') != run['prepared_audit_sha256']:
        raise ValueError('prepared reference differs')
    if file_digest(args.prepared / 'manifests.jsonl') != audit['artifacts_sha256']['manifests.jsonl']:
        raise ValueError('serving manifest changed')
    for name, expected in run['visual_state_identity_files_sha256'].items():
        if file_digest(Path(run['reader']) / name) != expected:
            raise ValueError('frozen Reader changed')
    manifests = {row['question_id']: row for row in map(json.loads, (args.prepared / 'manifests.jsonl').read_text().splitlines())}
    torch.set_num_threads(2)
    torch.manual_seed(0)
    processor = AutoProcessor.from_pretrained(run['reader'], local_files_only=True)
    processor.image_processor.size['longest_edge'] = run['image_max_pixels']
    candidates = []
    previous = []
    with zipfile.ZipFile(args.archive) as archive:
        for row in run['rows']:
            if row['images']:
                continue
            inputs = reconstruct(processor, archive, run, row, manifests[row['question_id']])
            tokens = inputs.input_ids[0].tolist()
            common = min(longest_prefix(previous, tokens), len(tokens) - 1)
            if common:
                candidates.append((common, row, inputs))
            previous.append(tokens)
    if not candidates:
        raise ValueError('no naturally occurring exact text-prefix opportunity')
    boundary, original, inputs = max(candidates, key=lambda item: item[0])
    registration = {'run_sha256': file_digest(args.run), 'script_sha256': file_digest(Path(__file__)),
        'prefix_source_sha256': file_digest(Path(__file__).with_name('prefix_state.py')),
        'reconstruct_source_sha256': file_digest(Path(__file__).with_name('audit_closure_prefixes.py')),
        'question_id': original['question_id'], 'selection': 'longest prior identical text-only prefix; no labels',
        'prefix_tokens': boundary, 'input_tokens': original['input_tokens'],
        'max_new_tokens': args.max_new_tokens, 'evaluation_label_files_read': []}
    with args.output.open('x') as handle:
        json.dump({'complete': False, 'registration': registration}, handle, indent=2)
    reader = AutoModelForImageTextToText.from_pretrained(run['reader'], local_files_only=True,
        dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda:0').eval()
    inputs = inputs.to('cuda:0')
    eos = reader.config.text_config.eos_token_id
    stop_ids = set(eos if isinstance(eos, list) else [eos])
    results = {}
    logits = {}
    snapshot = None
    with torch.inference_mode():
        for mode in ('unsplit', 'split_cold', 'snapshot_restored'):
            reader.model.rope_deltas = None
            torch.cuda.synchronize()
            started = time.perf_counter()
            restore_seconds = 0.0
            if mode == 'unsplit':
                output = reader(**inputs, use_cache=True, logits_to_keep=1)
            elif mode == 'split_cold':
                prefix_inputs = {name: value[:, :boundary] for name, value in inputs.items()
                                 if name in ('input_ids', 'attention_mask', 'mm_token_type_ids')}
                prefix = reader(**prefix_inputs, use_cache=True, logits_to_keep=1)
                snapshot = PrefixStateSnapshot(prefix.past_key_values, prefix_inputs['input_ids'],
                    getattr(reader.model, 'rope_deltas', None), capacity_bytes=512 * 1024**2)
                output = reader(input_ids=inputs.input_ids[:, boundary:], past_key_values=prefix.past_key_values,
                                use_cache=True, logits_to_keep=1)
            else:
                restore_started = time.perf_counter()
                cache, rope = snapshot.restore('cuda:0')
                reader.model.rope_deltas = rope
                torch.cuda.synchronize()
                restore_seconds = time.perf_counter() - restore_started
                output = reader(input_ids=inputs.input_ids[:, boundary:], past_key_values=cache,
                                use_cache=True, logits_to_keep=1)
            torch.cuda.synchronize()
            prefill_seconds = time.perf_counter() - started
            tokens, stop, first = decode(reader, output, args.max_new_tokens, stop_ids)
            torch.cuda.synchronize()
            results[mode] = {'token_ids': tokens, 'stop_reason': stop,
                             'diagnostic_prefill_seconds': prefill_seconds,
                             'restore_seconds': restore_seconds,
                             'diagnostic_total_seconds': time.perf_counter() - started}
            logits[mode] = first
            del output
    comparisons = {}
    for left, right in (('unsplit', 'split_cold'), ('split_cold', 'snapshot_restored'),
                        ('unsplit', 'snapshot_restored')):
        comparisons[left + '_vs_' + right] = {
            'token_ids_equal': results[left]['token_ids'] == results[right]['token_ids'],
            'stop_equal': results[left]['stop_reason'] == results[right]['stop_reason'],
            'first_logits_bitwise_equal': torch.equal(logits[left], logits[right]),
            'first_logits_max_error': float((logits[left] - logits[right]).abs().max())}
    result = {'complete': True, 'registration': registration, 'modes': results,
        'comparisons': comparisons, 'snapshot_bytes': snapshot.bytes,
        'snapshot_seconds': snapshot.snapshot_seconds,
        'qualified_on_this_case': all(value['token_ids_equal'] and value['stop_equal']
                                      and value['first_logits_bitwise_equal'] for value in comparisons.values()),
        'production_state_reuse_enabled': False,
        'boundary': 'single development natural text-only prefix and limited generated suffix; no global state-equivalence, memory-policy benefit, visual-prefix restore or E2E claim'}
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
