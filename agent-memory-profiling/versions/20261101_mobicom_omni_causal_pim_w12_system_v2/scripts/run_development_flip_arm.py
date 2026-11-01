#!/usr/bin/env python3
"""Run one development-only quantization arm on the frozen flip cohort."""
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


def build_command(registration: dict, cohort: dict, user: str, *, cohort_path: Path,
                  output_root: Path, quantization: str, residual_bits: int,
                  fallback_scope: str) -> list[str]:
    arguments = dict(registration['arguments'])
    arguments.update({
        'output': str(output_root / 'serving' / f'{user}.json'),
        'user_id': user,
        'limit': len(cohort['question_ids_by_user'][user]),
        'cohort': str(cohort_path),
        'device': 'cuda:0',
        'reader_quantization': quantization,
        'w8a16_group_size': 128,
        'w8a16_residual_bits': residual_bits,
        'w8a16_fallback_scope': fallback_scope,
        'w12a16_group_size': 128,
        'diagnostic_input_dump_dir': None,
        'hardware_identity': 'RTX3090:development-flip-quality-diagnostic',
    })
    command = [sys.executable, str(ROOT / 'scripts/run_omni_raw_image_smoke.py')]
    for key, value in arguments.items():
        if value is None or value is False:
            continue
        if value is True:
            command.append(f'--{key.replace("_", "-")}')
        else:
            command.extend([f'--{key.replace("_", "-")}', str(value)])
    return command


def run_user(registration: dict, cohort: dict, user: str, cohort_path: Path,
             output_root: Path, gpu: str, quantization: str,
             residual_bits: int, fallback_scope: str) -> dict:
    output = output_root / 'serving' / f'{user}.json'
    log = output_root / 'logs' / f'{user}.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    command = build_command(
        registration, cohort, user, cohort_path=cohort_path,
        output_root=output_root, quantization=quantization,
        residual_bits=residual_bits, fallback_scope=fallback_scope)
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
        'user': user, 'gpu': gpu, 'returncode': completed.returncode,
        'seconds': time.time() - started, 'output': str(output),
        'output_sha256': digest(output) if output.exists() else None,
        'log': str(log),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--cohort', type=Path, required=True)
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--arm-name', required=True)
    parser.add_argument('--reader-quantization', choices=('bf16', 'w8a16', 'w12a16'), required=True)
    parser.add_argument('--residual-bits', type=int, choices=(0, 2, 4), default=0)
    parser.add_argument('--fallback-scope', choices=('none', 'lm_head', 'output_sensitive'),
                        default='none')
    parser.add_argument('--gpus', nargs='+', default=['0', '1', '3'])
    args = parser.parse_args()
    if len(args.gpus) < 3:
        raise ValueError('three isolated GPUs required')
    if args.output_root.exists():
        raise ValueError('fresh output root required')
    cohort = json.loads(args.cohort.read_text())
    if cohort.get('diagnostic_only') is not True or cohort.get('requests') != 13:
        raise ValueError('frozen 13-request development diagnostic cohort required')
    if sorted(cohort['question_ids_by_user'], key=int) != ['0', '1', '2']:
        raise ValueError('only development users 0,1,2 are allowed')
    registration = json.loads(args.registration.read_text())
    args.output_root.mkdir(parents=True)
    jobs = list(zip(cohort['question_ids_by_user'], args.gpus))
    results = []
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        futures = {
            pool.submit(run_user, registration, cohort, user, args.cohort,
                        args.output_root, gpu, args.reader_quantization,
                        args.residual_bits, args.fallback_scope): (user, gpu)
            for user, gpu in jobs
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    failed = [result for result in results if result['returncode'] != 0]
    if failed:
        raise SystemExit(f'development flip arm failed: {failed}')
    for user in cohort['question_ids_by_user']:
        path = args.output_root / 'serving' / f'{user}.json'
        run = json.loads(path.read_text())
        if not run.get('complete') or run.get('evaluation_label_files_read') != []:
            raise ValueError(f'incomplete or label-reading output: {path}')
        if [row['question_id'] for row in run['rows']] != cohort['question_ids_by_user'][user]:
            raise ValueError(f'output order differs: {path}')
        if run.get('reader_quantization') != args.reader_quantization:
            raise ValueError(f'quantization differs: {path}')
        if (args.reader_quantization == 'w8a16'
                and run.get('reader_w8a16_residual_bits') != args.residual_bits):
            raise ValueError(f'residual bits differ: {path}')
        if (args.reader_quantization == 'w12a16'
                and (run.get('reader_weight_bits') != 12
                     or run.get('reader_w12a16_group_size') != 128)):
            raise ValueError(f'W12A16 contract differs: {path}')
    manifest = {
        'schema': 'hyperpim-sc-development-flip-arm-v1',
        'complete': True,
        'arm': args.arm_name,
        'reader_quantization': args.reader_quantization,
        'reader_weight_bits': 12 if args.reader_quantization == 'w12a16' else 8 if args.reader_quantization == 'w8a16' else None,
        'residual_bits': args.residual_bits,
        'fallback_scope': args.fallback_scope,
        'cohort': str(args.cohort),
        'cohort_sha256': digest(args.cohort),
        'registration': str(args.registration),
        'registration_sha256': digest(args.registration),
        'jobs': sorted(results, key=lambda item: int(item['user'])),
        'evaluation_label_files_read': [],
        'boundary': ('development-only flip recovery experiment; 13 requests are not a '
                     'quality gate and never include holdout users'),
    }
    (args.output_root / 'serving_manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'complete': True, 'arm': args.arm_name,
                      'output_root': str(args.output_root)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
