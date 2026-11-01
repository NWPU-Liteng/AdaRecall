"""Teacher-forced drift analysis for a BF16/W8A16 Reader pair.

The script consumes diagnostic-only processor tensors dumped by the serving
runner and the reference arm's generated token ids.  It never reads evaluation
labels.  BF16 and W8A16 are loaded sequentially on one device, so the two
models are compared on exactly the same prompt and teacher-forced answer.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import json
import math
from pathlib import Path
import sys

import torch
from transformers import AutoModelForImageTextToText

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pim_backend.functional_decode import materialize_w8a16_primary


def digest(path):
    import hashlib
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            result.update(block)
    return result.hexdigest()


def load_reference_tokens(paths):
    tokens = {}
    provenance = {}
    for path in paths:
        path = Path(path)
        run = json.loads(path.read_text())
        if not run.get('complete') or run.get('evaluation_label_files_read') != []:
            raise ValueError('complete label-isolated reference run required')
        for row in run['rows']:
            question_id = row['question_id']
            if question_id in tokens:
                raise ValueError('duplicate reference question id')
            tokens[question_id] = [int(value) for value in row['answer_token_ids']]
        provenance[str(path)] = digest(path)
    return tokens, provenance


def load_reader_inputs(path):
    value = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(value, dict) or 'input_ids' not in value:
        raise ValueError('processor input tensor dictionary required')
    tensors = {name: tensor for name, tensor in value.items()
               if isinstance(tensor, torch.Tensor)}
    if tensors['input_ids'].ndim != 2 or tensors['input_ids'].shape[0] != 1:
        raise ValueError('batch-one processor input required')
    return tensors


def extend_with_answer(inputs, answer_tokens):
    answer = torch.tensor([answer_tokens], dtype=inputs['input_ids'].dtype)
    extended = dict(inputs)
    extended['input_ids'] = torch.cat([inputs['input_ids'], answer], dim=1)
    if 'attention_mask' in inputs:
        extended['attention_mask'] = torch.cat([
            inputs['attention_mask'],
            torch.ones_like(answer, dtype=inputs['attention_mask'].dtype)], dim=1)
    for name in ('mm_token_type_ids', 'token_type_ids'):
        if name in inputs:
            extended[name] = torch.cat([inputs[name], torch.zeros_like(answer)], dim=1)
    prompt_tokens = int(inputs['input_ids'].shape[1])
    if prompt_tokens < 1 or not answer_tokens:
        raise ValueError('nonempty prompt and answer required')
    return extended, prompt_tokens


def load_model(path, device, quantization='bf16', group_size=128,
               fallback_scope='none', residual_bits=0):
    model = AutoModelForImageTextToText.from_pretrained(
        str(path), local_files_only=True, dtype=torch.bfloat16,
        attn_implementation='sdpa').to(device).eval()
    conversion = None
    if quantization == 'bf16':
        if fallback_scope != 'none':
            raise ValueError('BF16 arm cannot declare a W8A16 fallback scope')
    elif quantization == 'w8a16':
        with torch.inference_mode():
            conversion = materialize_w8a16_primary(
                model, group_size=group_size, fallback_scope=fallback_scope,
                residual_bits=residual_bits)
    else:
        raise ValueError('only bf16 and w8a16 diagnostic arms are supported')
    return model, conversion


def forward_with_cache(model, inputs, answer_tokens, device):
    extended, prompt_tokens = extend_with_answer(inputs, answer_tokens)
    moved = {name: tensor.to(device) for name, tensor in extended.items()}
    answer_start = prompt_tokens
    answer_end = answer_start + len(answer_tokens)
    prediction_positions = torch.arange(
        answer_start - 1, answer_end - 1, device=device)
    with torch.inference_mode():
        output = model(
            **moved, use_cache=False, output_hidden_states=True,
            logits_to_keep=prediction_positions)
    hidden = tuple(value.detach().to('cpu') for value in output.hidden_states)
    logits = output.logits.detach().to('cpu') if output.logits is not None else None
    if logits is not None:
        if logits.shape[1] != len(prediction_positions):
            raise ValueError('logit position selection changed')
    return {
        'prompt_tokens': prompt_tokens,
        'answer_tokens': list(answer_tokens),
        'hidden_states': hidden,
        'logits': logits,
        'prediction_positions': prediction_positions.cpu(),
    }


def compare_arms(reference, candidate):
    hidden_reference = reference['hidden_states']
    hidden_candidate = candidate['hidden_states']
    if len(hidden_reference) != len(hidden_candidate):
        raise ValueError('layer count differs between arms')
    prompt_tokens = reference['prompt_tokens']
    answer_tokens = len(reference['answer_tokens'])
    if candidate['prompt_tokens'] != prompt_tokens:
        raise ValueError('prompt token count differs between arms')
    start = prompt_tokens - 1
    stop = prompt_tokens + answer_tokens - 1
    layers = []
    for index, (left, right) in enumerate(zip(hidden_reference, hidden_candidate)):
        if left.shape != right.shape:
            raise ValueError('hidden-state shape differs between arms')
        left_rows = left[0, start:stop].float()
        right_rows = right[0, start:stop].float()
        difference = left_rows - right_rows
        reference_norm = torch.linalg.vector_norm(left_rows, dim=-1)
        difference_norm = torch.linalg.vector_norm(difference, dim=-1)
        relative = difference_norm / reference_norm.clamp_min(1e-12)
        cosine = torch.nn.functional.cosine_similarity(
            left_rows, right_rows, dim=-1)
        layers.append({
            'layer': index,
            'relative_l2_mean': float(relative.mean()),
            'relative_l2_p95': float(torch.quantile(relative, 0.95)),
            'relative_l2_max': float(relative.max()),
            'cosine_mean': float(cosine.mean()),
            'max_abs': float(difference.abs().max()),
        })
    thresholds = (1e-4, 1e-3, 1e-2, 5e-2)
    first_layer = {}
    for threshold in thresholds:
        matches = [row['layer'] for row in layers
                   if row['relative_l2_mean'] >= threshold]
        first_layer[f'{threshold:g}'] = matches[0] if matches else None

    logits_reference = reference['logits']
    logits_candidate = candidate['logits']
    if logits_reference is None or logits_candidate is None:
        raise ValueError('teacher-forced logits required')
    top_reference = logits_reference[0].float().topk(2, dim=-1)
    top_candidate = logits_candidate[0].float().topk(2, dim=-1)
    answer = torch.tensor(reference['answer_tokens'])
    prediction_answer = answer
    if len(prediction_answer) != top_reference.values.shape[0]:
        raise ValueError('answer/logit alignment changed')
    reference_ids = top_reference.indices[:, 0]
    candidate_ids = top_candidate.indices[:, 0]
    first_flip = (reference_ids != candidate_ids).nonzero(as_tuple=False)
    margins_reference = top_reference.values[:, 0] - top_reference.values[:, 1]
    margins_candidate = top_candidate.values[:, 0] - top_candidate.values[:, 1]
    logit_cosine = torch.nn.functional.cosine_similarity(
        logits_reference[0].float(), logits_candidate[0].float(), dim=-1)
    return {
        'layers': layers,
        'first_layer_relative_l2_mean_at_least': first_layer,
        'logits': {
            'positions': int(reference_ids.numel()),
            'top1_agreement': int((reference_ids == candidate_ids).sum()),
            'top1_agreement_rate': float((reference_ids == candidate_ids).float().mean()),
            'reference_top1_matches_answer': int((reference_ids == prediction_answer).sum()),
            'candidate_top1_matches_answer': int((candidate_ids == prediction_answer).sum()),
            'first_top1_flip_position': (int(first_flip[0, 0])
                                         if first_flip.numel() else None),
            'reference_margin_mean': float(margins_reference.mean()),
            'candidate_margin_mean': float(margins_candidate.mean()),
            'margin_ratio_candidate_over_reference': float(
                margins_candidate.mean() / margins_reference.mean().clamp_min(1e-12)),
            'logit_cosine_mean': float(logit_cosine.mean()),
        },
    }


def analyze_request(inputs_path, answer_tokens, model_path, device,
                    quantization, group_size, fallback_scope, residual_bits=0):
    inputs = load_reader_inputs(inputs_path)
    model, conversion = load_model(
        model_path, device, quantization, group_size, fallback_scope,
        residual_bits)
    result = forward_with_cache(model, inputs, answer_tokens, device)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    result['conversion'] = conversion
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reader', type=Path, required=True)
    parser.add_argument('--inputs-dir', type=Path, required=True)
    parser.add_argument('--reference-run', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--group-size', type=int, default=128)
    parser.add_argument('--fallback-scope',
                        choices=('none', 'lm_head', 'output_sensitive'),
                        default='none')
    parser.add_argument('--residual-bits', type=int, choices=(0, 2, 4), default=0)
    parser.add_argument('--max-requests', type=int, default=0,
                        help='zero means all dumped requests after sharding')
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--shard-count', type=int, default=1)
    parser.add_argument('--max-answer-tokens', type=int, default=256)
    args = parser.parse_args()
    if (args.shard_count < 1 or args.shard_index < 0
            or args.shard_index >= args.shard_count):
        raise ValueError('valid shard index/count required')
    if args.output.exists():
        raise ValueError('fresh drift output required')
    reference_tokens, provenance = load_reference_tokens(args.reference_run)
    paths = sorted(args.inputs_dir.glob('*.reader_inputs.pt'))
    if not paths:
        raise ValueError('no diagnostic Reader inputs found')
    requests = []
    for path in paths:
        question_id = path.name.removesuffix('.reader_inputs.pt')
        if question_id not in reference_tokens:
            raise ValueError('missing reference tokens for ' + question_id)
        requests.append((question_id, path, reference_tokens[question_id]))
    requests = requests[args.shard_index::args.shard_count]
    if args.max_requests:
        requests = requests[:args.max_requests]

    rows = []
    for question_id, path, tokens in requests:
        tokens = tokens[:args.max_answer_tokens]
        reference = analyze_request(
            path, tokens, args.reader, args.device, 'bf16', args.group_size,
            'none', 0)
        candidate = analyze_request(
            path, tokens, args.reader, args.device, 'w8a16',
            args.group_size, args.fallback_scope, args.residual_bits)
        rows.append({
            'question_id': question_id,
            'inputs_sha256': digest(path),
            'reference_run_tokens_sha256': digest(
                Path(args.reference_run[0])),
            'prompt_tokens': reference['prompt_tokens'],
            'answer_tokens_used': len(tokens),
            'reference_conversion': reference['conversion'],
            'candidate_conversion': candidate['conversion'],
            **compare_arms(reference, candidate),
        })
        print(json.dumps({'question_id': question_id,
                          'first_layer': rows[-1]['first_layer_relative_l2_mean_at_least'],
                          'logits': rows[-1]['logits']}, ensure_ascii=False), flush=True)

    result = {
        'schema': 'reader-quantization-teacher-forced-drift-v1',
        'complete': True,
        'reader': str(args.reader),
        'candidate_quantization': 'w8a16',
        'candidate_fallback_scope': args.fallback_scope,
        'candidate_residual_bits': args.residual_bits,
        'group_size': args.group_size,
        'requests': len(rows),
        'shard': {'index': args.shard_index, 'count': args.shard_count},
        'reference_runs': provenance,
        'inputs_dir': str(args.inputs_dir),
        'rows': rows,
        'boundary': ('teacher-forced BF16/W8A16 hidden-state and logit drift on exact '
                     'post-retrieval processor inputs; no evaluation labels and no PIM '
                     'timing claim'),
        'performance_comparable': False,
        'actual_physical_pim_executed': False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'requests': len(rows), 'output': str(args.output)},
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
