"""Qualify exact cached Reader prefill logits on captured raw-image inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from omni_proof_runtime import load_reader, write_json
from omni_visual_feature_store import VisualFeatureStore
from omni_visual_feature_cache_runtime import (DEFAULT_CAPACITY_BYTES,
                                               VisualFeatureCache)


def load_capture(capture_path):
    capture_path = Path(capture_path)
    metadata_path = capture_path.with_suffix('.json')
    if not capture_path.is_file() or not metadata_path.is_file():
        raise ValueError(f'capture and metadata are required: {capture_path}')
    metadata = json.loads(metadata_path.read_text())
    if (metadata.get('schema') != 'proof-mainline-reader-input-capture-metadata-v1'
            or Path(metadata.get('input_path', '')).resolve() != capture_path.resolve()):
        raise ValueError(f'capture metadata identity differs: {metadata_path}')
    inputs = torch.load(capture_path, map_location='cpu', weights_only=True)
    if (not isinstance(inputs, dict) or not inputs
            or any(not isinstance(name, str) or not isinstance(value, torch.Tensor)
                   for name, value in inputs.items())):
        raise ValueError('captured Reader input must be a non-empty tensor mapping')
    if 'input_ids' not in inputs or 'pixel_values' not in inputs or 'image_grid_thw' not in inputs:
        raise ValueError('capture requires input_ids, pixel_values, and image_grid_thw')
    input_ids_sha256 = hashlib.sha256(inputs['input_ids'].contiguous().numpy().tobytes()).hexdigest()
    if input_ids_sha256 != metadata.get('input_ids_sha256'):
        raise ValueError('captured input_ids digest differs from metadata')
    image_sha256s = metadata.get('image_sha256s')
    if (not isinstance(image_sha256s, list) or not image_sha256s
            or any(not isinstance(value, str) or len(value) != 64 for value in image_sha256s)
            or inputs['image_grid_thw'].shape != (len(image_sha256s), 3)):
        raise ValueError('captured ordered image identities differ from processor grids')
    return inputs, metadata


def prefill_last_logits(model, inputs, device):
    with torch.inference_mode():
        output = model(**inputs, use_cache=True, logits_to_keep=1)
    logits = output.logits.detach().to(device='cpu', copy=True).contiguous()
    del output
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)
    return logits


def max_absolute_difference(left, right):
    if left.shape != right.shape or not left.is_floating_point() or not right.is_floating_point():
        raise ValueError('logit comparison requires equal floating tensors')
    return float((left - right).abs().max().item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reader', type=Path, required=True)
    parser.add_argument('--capture', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--capacity-bytes', type=int, default=DEFAULT_CAPACITY_BYTES)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f'refusing to overwrite qualification report: {args.output}')
    started = time.perf_counter()
    processor, model = load_reader(args.reader)
    cache = VisualFeatureCache(
        args.reader, processor, model, capacity_bytes=args.capacity_bytes)
    rows = []
    for capture_path in args.capture:
        capture_started = time.perf_counter()
        inputs, metadata = load_capture(capture_path)
        device_inputs = {name: value.to(args.device) for name, value in inputs.items()}
        before = cache.store.snapshot()
        reference = prefill_last_logits(model, device_inputs, args.device)
        with cache.activate(model, metadata['image_sha256s']):
            cold = prefill_last_logits(model, device_inputs, args.device)
        after_cold = cache.store.snapshot()
        with cache.activate(model, metadata['image_sha256s']):
            warm = prefill_last_logits(model, device_inputs, args.device)
        after_warm = cache.store.snapshot()
        cold_equal = torch.equal(reference, cold)
        warm_equal = torch.equal(reference, warm)
        cold_delta = {
            name: int(after_cold[name]) - int(before[name])
            for name in ('hits', 'misses', 'bypasses', 'integrity_failures')}
        warm_delta = {
            name: int(after_warm[name]) - int(after_cold[name])
            for name in ('hits', 'misses', 'bypasses', 'integrity_failures')}
        cold_exercised = (cold_delta['misses'] == 1 and cold_delta['hits'] == 0
                          and cold_delta['bypasses'] == 0
                          and cold_delta['integrity_failures'] == 0)
        warm_exercised = (warm_delta['hits'] == 1 and warm_delta['misses'] == 0
                          and warm_delta['bypasses'] == 0
                          and warm_delta['integrity_failures'] == 0)
        row = {
            'person_id': metadata['person_id'],
            'question_id': metadata['question_id'],
            'capture_path': str(capture_path.resolve()),
            'input_tokens': int(device_inputs['input_ids'].shape[1]),
            'image_count': len(metadata['image_sha256s']),
            'image_sha256s': metadata['image_sha256s'],
            'prefill_logit_scope': ('production-equivalent last-position next-token logits '
                                    'from use_cache=True, logits_to_keep=1'),
            'reference_logits_sha256': VisualFeatureStore.tensor_digest(reference),
            'cold_logits_sha256': VisualFeatureStore.tensor_digest(cold),
            'warm_logits_sha256': VisualFeatureStore.tensor_digest(warm),
            'uncached_vs_cold_torch_equal': cold_equal,
            'uncached_vs_warm_torch_equal': warm_equal,
            'cold_cache_exercised': cold_exercised,
            'warm_cache_exercised': warm_exercised,
            'uncached_vs_warm_max_abs_difference': (
                max_absolute_difference(reference, warm) if not warm_equal else 0.0),
            'cold_cache_delta': cold_delta,
            'warm_cache_delta': warm_delta,
            'cached_feature_sha256': cache.store.last_feature_hashes,
            'seconds': time.perf_counter() - capture_started,
        }
        rows.append(row)
        print(json.dumps({'question_id': row['question_id'],
                          'images': row['image_count'],
                          'cold_bitwise_equal': cold_equal,
                          'warm_bitwise_equal': warm_equal,
                          'cold_cache_exercised': cold_exercised,
                          'warm_cache_exercised': warm_exercised,
                          'warm_cache_delta': row['warm_cache_delta'],
                          'seconds': row['seconds']}, ensure_ascii=False), flush=True)
        del device_inputs, reference, cold, warm
    complete = all(row['uncached_vs_cold_torch_equal']
                   and row['uncached_vs_warm_torch_equal']
                   and row['cold_cache_exercised']
                   and row['warm_cache_exercised'] for row in rows)
    report = {
        'schema': 'visual-feature-cache-prefill-qualification-v1',
        'complete': complete,
        'reader': str(args.reader),
        'device': args.device,
        'capacity_bytes': args.capacity_bytes,
        'comparison': 'torch.equal on last-position prefill logits; cache off vs cold miss vs warm hit',
        'model_identity': cache.model_identity,
        'processor_identity': cache.processor_identity,
        'rows': rows,
        'cache_snapshot': cache.snapshot(),
        'elapsed_seconds': time.perf_counter() - started,
        'boundary': ('captured processor tensors are replayed to compare the exact Reader '
                     'prefill path; this qualification does not rerun retrieval, generation, '
                     'or include retrieval/decode latency'),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    if not complete:
        raise RuntimeError('cached visual features changed prefill logits bitwise')


if __name__ == '__main__':
    main()
