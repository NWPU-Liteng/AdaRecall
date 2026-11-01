"""Replay captured, label-free MobileMem Reader inputs through a fixed GPU queue."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as functional


SEQUENCE_KEYS = {'input_ids', 'attention_mask', 'token_type_ids', 'mm_token_type_ids'}
VISION_KEYS = {'pixel_values', 'image_grid_thw'}


def load_capture_rows(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        run = json.loads(path.read_text())
        if run.get('schema') != 'proof-mainline-reader-input-capture-v1':
            raise ValueError(f'not a Reader input capture: {path}')
        if not run.get('complete') or run.get('capacity_capture_only') is not True:
            raise ValueError(f'incomplete or non-capture run: {path}')
        if run.get('evaluation_label_files_read'):
            raise ValueError(f'capture read evaluation labels: {path}')
        for row in run.get('rows', []):
            if row.get('evaluation_label_files_read'):
                raise ValueError('captured request read evaluation labels')
            capture = Path(row['input_path'])
            if not capture.is_file():
                raise FileNotFoundError(capture)
            rows.append({**row, 'capture_run': str(path)})
    identities = [(row['person_id'], row['question_id']) for row in rows]
    if not rows or len(identities) != len(set(identities)):
        raise ValueError('at least one unique captured request is required')
    return sorted(rows, key=lambda row: (int(row['request_index']), int(row['person_id']),
                                         row['question_id']))


def collate_inputs(items: list[dict[str, torch.Tensor]], pad_token_id: int) -> dict[str, torch.Tensor]:
    if not items:
        raise ValueError('cannot collate an empty batch')
    keys = set.union(*(set(item) for item in items))
    if not {'input_ids', 'attention_mask'} <= keys:
        raise ValueError('captured inputs require input_ids and attention_mask')
    missing_nonvision = {
        key for key in keys - VISION_KEYS
        if any(key not in item for item in items)
    }
    if missing_nonvision:
        raise ValueError('captured Reader tensor keys differ within a batch: '
                         + ', '.join(sorted(missing_nonvision)))

    result = {}
    for key in sorted(keys):
        tensors = [item[key] for item in items if key in item]
        if key in SEQUENCE_KEYS:
            lengths = [int(item['input_ids'].shape[-1]) for item in items]
            tensors = [item.get(key, torch.zeros_like(item['input_ids'])) for item in items]
            width = max(lengths)
            fill = pad_token_id if key == 'input_ids' else 0
            padded = [functional.pad(tensor, (width - length, 0), value=fill)
                      for tensor, length in zip(tensors, lengths)]
            result[key] = torch.cat(padded, dim=0)
        elif key in VISION_KEYS:
            if any(tensor.ndim == 0 for tensor in tensors):
                raise ValueError(f'invalid captured vision tensor: {key}')
            if any(tensor.shape[1:] != tensors[0].shape[1:] for tensor in tensors):
                raise ValueError(f'incompatible captured vision tensor shapes: {key}')
            result[key] = torch.cat(tensors, dim=0)
        else:
            if any(tensor.shape[1:] != tensors[0].shape[1:] for tensor in tensors):
                raise ValueError(f'incompatible captured tensor shapes: {key}')
            result[key] = torch.cat(tensors, dim=0)
    return result


def cache_tensor_bytes(value, seen: set[int] | None = None, device: str | None = None) -> int:
    if seen is None:
        seen = set()
    identity = id(value)
    if identity in seen:
        return 0
    seen.add(identity)
    if torch.is_tensor(value):
        if device is not None and value.device.type != device:
            return 0
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(cache_tensor_bytes(item, seen, device) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(cache_tensor_bytes(item, seen, device) for item in value)
    total = 0
    for name in ('layers', 'key_cache', 'value_cache', 'keys', 'values',
                 'conv_states', 'recurrent_states', 'state'):
        if hasattr(value, name):
            try:
                total += cache_tensor_bytes(getattr(value, name), seen, device)
            except (AttributeError, RuntimeError, TypeError):
                pass
    if hasattr(value, 'to_legacy_cache'):
        try:
            total += cache_tensor_bytes(value.to_legacy_cache(), seen, device)
        except (AttributeError, RuntimeError, TypeError):
            pass
    return total


def decode_batch(model, processor, inputs, rows, *, max_new_tokens, interarrival_ms,
                 scheduler_state, cache_mode):
    inputs = {name: value.to('cuda:0') for name, value in inputs.items()}
    batch_size = len(rows)
    started = time.perf_counter()
    arrival_offsets = [scheduler_state['arrival_index'] + index
                       for index in range(batch_size)]
    arrival_times = [offset * interarrival_ms / 1000.0 for offset in arrival_offsets]
    earliest_start = max(scheduler_state['finish_time'], arrival_times[-1])
    elapsed = time.perf_counter() - scheduler_state['wall_start']
    if earliest_start > elapsed:
        time.sleep(earliest_start - elapsed)
    batch_start = time.perf_counter()

    torch.cuda.reset_peak_memory_stats(0)
    model.model.rope_deltas = None
    initial_cache = None
    if cache_mode == 'host_offload':
        from transformers.cache_utils import DynamicCache
        initial_cache = DynamicCache(config=model.config.get_text_config(decoder=True),
                                     offloading=True, offload_only_non_sliding=False)
    with torch.inference_mode():
        output = model(**inputs, past_key_values=initial_cache,
                       use_cache=True, logits_to_keep=1)
    torch.cuda.synchronize()
    prefill_seconds = time.perf_counter() - batch_start
    cache = output.past_key_values
    state_bytes = cache_tensor_bytes(cache)
    cache_cuda_bytes = cache_tensor_bytes(cache, device='cuda')
    cache_host_bytes = cache_tensor_bytes(cache, device='cpu')
    next_tokens = output.logits[:, -1, :].argmax(-1, keepdim=True)
    attention_mask = inputs['attention_mask']
    eos_ids = getattr(model.config.text_config, 'eos_token_id', processor.tokenizer.eos_token_id)
    eos_ids = set(eos_ids if isinstance(eos_ids, list) else [eos_ids])
    pad_id = processor.tokenizer.pad_token_id
    pad_id = processor.tokenizer.eos_token_id if pad_id is None else pad_id
    generated = [[] for _ in rows]
    finished = [False] * batch_size
    finish_seconds = [None] * batch_size

    for position in range(max_new_tokens):
        token_values = next_tokens[:, 0].tolist()
        active_before = [not value for value in finished]
        for index, token_id in enumerate(token_values):
            if finished[index]:
                continue
            generated[index].append(int(token_id))
            if token_id in eos_ids:
                finished[index] = True
                finish_seconds[index] = time.perf_counter() - batch_start
        if all(finished) or position + 1 == max_new_tokens:
            break
        active_after = torch.tensor([not value for value in finished], device='cuda:0',
                                    dtype=attention_mask.dtype).unsqueeze(1)
        next_tokens = torch.where(active_after.bool(), next_tokens,
                                  torch.full_like(next_tokens, pad_id))
        attention_mask = torch.cat((attention_mask, active_after), dim=1)
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids = position_ids.masked_fill(attention_mask == 0, 0)[:, -1:]
        rope_deltas = model.model.rope_deltas
        if rope_deltas is not None:
            position_ids = position_ids + rope_deltas.to(position_ids.device).repeat_interleave(
                batch_size // rope_deltas.shape[0], dim=0)
        position_ids = position_ids.unsqueeze(0)
        with torch.inference_mode():
            output = model(input_ids=next_tokens, attention_mask=attention_mask,
                           position_ids=position_ids, past_key_values=cache,
                           use_cache=True, logits_to_keep=1)
        cache = output.past_key_values
        next_tokens = output.logits[:, -1, :].argmax(-1, keepdim=True)
        torch.cuda.synchronize()
    torch.cuda.synchronize()
    batch_finish = time.perf_counter()
    batch_seconds = batch_finish - batch_start
    peak_bytes = torch.cuda.max_memory_allocated(0)
    scheduler_state['finish_time'] = batch_finish - scheduler_state['wall_start']
    scheduler_state['arrival_index'] += batch_size

    results = []
    for index, row in enumerate(rows):
        completion = finish_seconds[index] if finish_seconds[index] is not None else batch_seconds
        answer = processor.tokenizer.decode(generated[index], skip_special_tokens=True).strip()
        results.append({
            'person_id': row['person_id'],
            'question_id': row['question_id'],
            'input_tokens': row['input_tokens'],
            'image_count': row['image_count'],
            'queue_wait_seconds': max(0.0, batch_start - scheduler_state['wall_start'] - arrival_times[index]),
            'reader_ttft_seconds': prefill_seconds,
            'reader_service_seconds': completion,
            'generated_tokens': len(generated[index]),
            'answer_sha256': hashlib.sha256(answer.encode()).hexdigest(),
            'cache_mode': cache_mode,
            'evaluation_label_files_read': [],
        })
    batch_result = {
        'batch_size': batch_size,
        'input_tokens': [row['input_tokens'] for row in rows],
        'prefill_seconds': prefill_seconds,
        'reader_batch_seconds': batch_seconds,
        'cache_tensor_bytes': state_bytes,
        'cache_cuda_bytes_after_prefill': cache_cuda_bytes,
        'cache_host_bytes_after_prefill': cache_host_bytes,
        'cache_mode': cache_mode,
        'peak_allocated_bytes': peak_bytes,
        'peak_reserved_bytes': torch.cuda.max_memory_reserved(0),
        'oom': False,
    }
    del output, cache, next_tokens, attention_mask
    return batch_result, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--capture-runs', type=Path, nargs='+', required=True)
    parser.add_argument('--reader', type=Path, required=True)
    parser.add_argument('--embedder', type=Path, required=True)
    parser.add_argument('--reranker', type=Path, required=True)
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 4, 8, 12])
    parser.add_argument('--interarrival-ms', type=float, default=0.0)
    parser.add_argument('--cache-mode', choices=['resident', 'host_offload'], default='resident')
    parser.add_argument('--max-new-tokens', type=int, default=128)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.interarrival_ms < 0 or args.max_new_tokens < 1:
        raise ValueError('fresh output, non-negative arrival interval, and positive decode limit required')
    if any(size < 1 for size in args.batch_sizes):
        raise ValueError('batch sizes must be positive')
    rows = load_capture_rows(args.capture_runs)

    from transformers import AutoModel, AutoTokenizer
    from omni_proof_runtime import load_reader
    from omni_provenance_optimizer import FrozenRelevanceModel

    processor, model = load_reader(args.reader)
    tokenizer = AutoTokenizer.from_pretrained(str(args.embedder), local_files_only=True,
                                               padding_side='left')
    embedder = AutoModel.from_pretrained(str(args.embedder), local_files_only=True,
        dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda:0').eval()
    reranker = FrozenRelevanceModel(args.reranker, 'cuda:0')
    torch.cuda.synchronize()
    resident_bytes = torch.cuda.memory_allocated(0)

    output = {
        'schema': 'mobilemem-reader-capacity-queue-v1',
        'complete': False,
        'workload_boundary': 'label-free captured final Reader inputs; fixed micro-batch queue; synthetic open-loop arrival interval',
        'capture_runs': [str(path) for path in args.capture_runs],
        'request_count': len(rows),
        'interarrival_ms': args.interarrival_ms,
        'cache_mode': args.cache_mode,
        'max_new_tokens': args.max_new_tokens,
        'resident_reader_embedder_reranker_bytes': resident_bytes,
        'gpu_name': torch.cuda.get_device_name(0),
        'runtime': {'torch': torch.__version__},
        'sweeps': [],
        'evaluation_label_files_read': [],
    }
    for batch_size in args.batch_sizes:
        sweep = {'configured_batch_size': batch_size, 'batches': [], 'requests': [],
                 'oom': False}
        subset = rows[:min(len(rows), 128)]
        state = {'arrival_index': 0, 'finish_time': 0.0, 'wall_start': time.perf_counter()}
        for offset in range(0, len(subset), batch_size):
            batch_rows = subset[offset:offset + batch_size]
            tensors = [torch.load(Path(row['input_path']), map_location='cpu', weights_only=True)
                       for row in batch_rows]
            inputs = collate_inputs(tensors, processor.tokenizer.pad_token_id
                                    if processor.tokenizer.pad_token_id is not None
                                    else processor.tokenizer.eos_token_id)
            try:
                batch_result, request_results = decode_batch(
                    model, processor, inputs, batch_rows,
                    max_new_tokens=args.max_new_tokens,
                    interarrival_ms=args.interarrival_ms,
                    scheduler_state=state, cache_mode=args.cache_mode)
                sweep['batches'].append(batch_result)
                sweep['requests'].extend(request_results)
            except torch.cuda.OutOfMemoryError as error:
                torch.cuda.empty_cache()
                sweep['oom'] = True
                sweep['oom_error'] = str(error)[:400]
                break
            finally:
                del inputs, tensors
        completed = len(sweep['requests'])
        elapsed = state['finish_time']
        sweep['completed_requests'] = completed
        sweep['queue_goodput_qps'] = completed / elapsed if elapsed > 0 else None
        sweep['ttft_p50_seconds'] = _percentile(
            [row['queue_wait_seconds'] + row['reader_ttft_seconds'] for row in sweep['requests']], 0.50)
        sweep['ttft_p95_seconds'] = _percentile(
            [row['queue_wait_seconds'] + row['reader_ttft_seconds'] for row in sweep['requests']], 0.95)
        sweep['reader_service_p95_seconds'] = _percentile(
            [row['reader_service_seconds'] for row in sweep['requests']], 0.95)
        output['sweeps'].append(sweep)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
        if sweep['oom']:
            break
    output['complete'] = True
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')


def _percentile(values, quantile):
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * quantile)
    return ordered[index]


if __name__ == '__main__':
    main()
