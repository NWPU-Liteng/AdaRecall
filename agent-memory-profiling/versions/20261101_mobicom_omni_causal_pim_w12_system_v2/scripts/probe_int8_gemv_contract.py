"""Qualify the declared INT8xINT8->INT32 GEMV contract on CUDA."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch


def digest(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def check_case(device, *, rows=256, inputs=256, batch=32, seed=20261023):
    generator = torch.Generator(device='cpu').manual_seed(seed)
    weights = torch.randint(-8, 8, (rows, inputs), generator=generator, dtype=torch.int8)
    activations = torch.randint(-8, 8, (batch, inputs), generator=generator, dtype=torch.int8)
    weights = weights.to(device)
    activations = activations.to(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    full = torch._int_mm(activations, weights.t())
    torch.cuda.synchronize(device)
    full_seconds = time.perf_counter() - started
    reference = activations.cpu().to(torch.int64) @ weights.cpu().t().to(torch.int64)
    output_shards = []
    shard_seconds = 0.0
    for start in range(0, rows, 32):
        torch.cuda.synchronize(device)
        shard_started = time.perf_counter()
        output_shards.append(torch._int_mm(activations, weights[start:start + 32].t()))
        torch.cuda.synchronize(device)
        shard_seconds += time.perf_counter() - shard_started
    sharded = torch.cat(output_shards, dim=1)
    expected_bound = inputs * 128 * 128
    return {
        'shape': {'batch': batch, 'inputs': inputs, 'rows': rows},
        'full_matches_int64': bool(torch.equal(full.cpu().to(torch.int64), reference)),
        'sharded_matches_int64': bool(torch.equal(sharded.cpu().to(torch.int64), reference)),
        'full_matches_sharded': bool(torch.equal(full, sharded)),
        'dtype': str(full.dtype),
        'max_abs_reference': int(reference.abs().max().item()),
        'conservative_abs_bound': expected_bound,
        'full_seconds': full_seconds,
        'sharded_seconds': shard_seconds,
        'weight_sha256': digest(weights),
        'activation_sha256': digest(activations),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('fresh probe output required')
    result = {'schema_version': 'int8-gemv-contract-probe-v1', 'complete': False,
              'declared_contract': 'INT8 x INT8 -> INT32; output rows may be independently sharded',
              'historical_failure': {'batch': 8, 'error': 'self.size(0) needs to be greater than 16, but got 8'},
              'hardware': torch.cuda.get_device_name(args.device) if torch.cuda.is_available() else None}
    try:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable')
        result['case'] = check_case(torch.device(args.device))
        result['complete'] = True
        result['qualification'] = ('exact int32 functional arithmetic for the tested shapes; '
                                    'not a quantized Reader, complete generation, or PIM timing result')
    except Exception as error:
        result['error_type'] = type(error).__name__
        result['error'] = str(error)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
