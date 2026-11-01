#!/usr/bin/env python3
"""Shard teacher-forced drift analysis for development-only quantization flips."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            result.update(block)
    return result.hexdigest()


def run_job(bits: int, shard: int, shard_count: int, gpu: str, *,
            reader: Path, inputs: Path, reference_runs: list[Path],
            output_root: Path) -> dict:
    output = output_root / f'r{bits}_shard_{shard}.json'
    log = output_root / 'logs' / f'r{bits}_shard_{shard}.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, str(ROOT / 'scripts/analyze_reader_quantization_drift.py'),
        '--reader', str(reader), '--inputs-dir', str(inputs),
        '--reference-run', *map(str, reference_runs),
        '--output', str(output), '--device', 'cuda:0',
        '--group-size', '128', '--fallback-scope', 'none',
        '--residual-bits', str(bits), '--max-answer-tokens', '256',
        '--shard-index', str(shard), '--shard-count', str(shard_count),
    ]
    environment = dict(os.environ)
    environment['CUDA_VISIBLE_DEVICES'] = gpu
    environment['OMP_NUM_THREADS'] = '2'
    environment['MKL_NUM_THREADS'] = '2'
    environment['OPENBLAS_NUM_THREADS'] = '2'
    environment['TOKENIZERS_PARALLELISM'] = 'false'
    started = time.time()
    with log.open('w') as handle:
        completed = subprocess.run(command, cwd=ROOT, env=environment,
                                   stdout=handle, stderr=subprocess.STDOUT)
    return {
        'bits': bits, 'shard': shard, 'shard_count': shard_count, 'gpu': gpu,
        'returncode': completed.returncode, 'seconds': time.time() - started,
        'output': str(output), 'output_sha256': digest(output) if output.exists() else None,
        'log': str(log),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dump-root', type=Path, required=True)
    parser.add_argument('--reference-root', type=Path, required=True)
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--gpus', nargs='+', default=['0', '1', '3', '4', '5', '6'])
    parser.add_argument('--bits', nargs='+', type=int, choices=(0, 2, 4), default=[2])
    args = parser.parse_args()
    if args.output_root.exists():
        raise ValueError('fresh drift output root required')
    registration = json.loads(args.registration.read_text())
    reader = Path(registration['arguments']['reader'])
    dump_manifest = json.loads((args.dump_root / 'manifest.json').read_text())
    if dump_manifest.get('complete') is not True:
        raise ValueError('complete development input dump required')
    inputs = args.dump_root / 'inputs'
    cohort = json.loads((args.dump_root / 'cohort.json').read_text())
    expected = set().union(*(set(cohort['question_ids_by_user'][user])
                             for user in ('0', '1', '2')))
    observed = {path.name.removesuffix('.reader_inputs.pt')
                for path in inputs.glob('*.reader_inputs.pt')}
    if not expected <= observed:
        raise ValueError('diagnostic input directory is incomplete')
    reference_runs = sorted(path for path in args.reference_root.glob('baseline_user_*.json')
                            if path.name.removeprefix('baseline_user_').removesuffix('.json').isdigit())
    if len(reference_runs) != 3:
        raise ValueError('three development BF16 reference runs required')
    if len(args.gpus) < 2:
        raise ValueError('at least two isolated GPUs required')
    args.output_root.mkdir(parents=True)
    shard_count = min(len(args.gpus), 6)
    jobs = [(bits, shard) for bits in args.bits for shard in range(shard_count)]
    results = []
    with ThreadPoolExecutor(max_workers=shard_count) as pool:
        futures = {}
        for index, (bits, shard) in enumerate(jobs):
            gpu = args.gpus[index % len(args.gpus)]
            futures[pool.submit(
                run_job, bits, shard, shard_count, gpu,
                reader=reader, inputs=inputs, reference_runs=reference_runs,
                output_root=args.output_root)] = (bits, shard)
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    failed = [result for result in results if result['returncode'] != 0]
    if failed:
        raise SystemExit(f'development drift shard failed: {failed}')
    for result in results:
        document = json.loads(Path(result['output']).read_text())
        if document.get('complete') is not True:
            raise ValueError(f"incomplete drift shard: {result['output']}")
    manifest = {
        'schema': 'hyperpim-sc-development-flip-drift-manifest-v1',
        'complete': True,
        'dump_root': str(args.dump_root),
        'dump_manifest_sha256': digest(args.dump_root / 'manifest.json'),
        'reference_root': str(args.reference_root),
        'reference_runs': {str(path): digest(path) for path in reference_runs},
        'shard_count': shard_count,
        'bits': args.bits,
        'jobs': sorted(results, key=lambda item: (item['bits'], item['shard'])),
        'boundary': ('teacher-forced development-only mechanism diagnosis; no holdout users; '
                     'no new quality-gate result'),
    }
    (args.output_root / 'manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'complete': True, 'jobs': len(results),
                      'output_root': str(args.output_root)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
