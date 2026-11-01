#!/usr/bin/env python3
"""Run one frozen precision arm on the 192-request development cohort."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
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


def command_for(registration: dict, cohort: dict, cohort_path: Path,
                user: str, output: Path, quantization: str,
                weight_bits: int | None, group_size: int) -> list[str]:
    arguments = dict(registration['arguments'])
    arguments.update({
        'output': str(output),
        'user_id': user,
        'limit': len(cohort['question_ids_by_user'][user]),
        'cohort': str(cohort_path),
        'device': 'cuda:0',
        'reader_quantization': quantization,
        'w12a16_group_size': group_size,
        'diagnostic_input_dump_dir': None,
        'hardware_identity': 'RTX3090:development-192-arm',
        'concurrency_contract': 'one isolated worker per development user',
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


def run_user(registration: dict, cohort: dict, cohort_path: Path, user: str,
             output_root: Path, gpu: str, quantization: str,
             weight_bits: int | None, group_size: int) -> dict:
    output = output_root / f'baseline_user_{user}.json'
    log = output_root / 'logs' / f'{user}.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    command = command_for(registration, cohort, cohort_path, user, output,
                          quantization, weight_bits, group_size)
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
    parser.add_argument('--preregistration', type=Path, required=True)
    parser.add_argument('--cohort', type=Path, required=True)
    parser.add_argument('--registration-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--reader-quantization', choices=('bf16', 'w8a16', 'w12a16'),
                        required=True)
    parser.add_argument('--weight-bits', type=int, choices=(8, 12), default=None)
    parser.add_argument('--group-size', type=int, choices=(8, 16, 32, 64, 128, 256),
                        default=128)
    parser.add_argument('--gpus', nargs='+', default=['0', '1', '3'])
    args = parser.parse_args()
    if args.output_root.exists():
        raise ValueError('fresh output root required')
    prereg = json.loads(args.preregistration.read_text())
    if prereg.get('status') != 'registered_before_generation':
        raise ValueError('frozen preregistration required')
    if digest(args.cohort) != prereg['cohort']['sha256']:
        raise ValueError('cohort differs from preregistration')
    cohort = json.loads(args.cohort.read_text())
    users = sorted(cohort['question_ids_by_user'], key=int)
    if users != prereg['cohort']['users']:
        raise ValueError('development users differ from preregistration')
    if any(len(cohort['question_ids_by_user'][user]) != prereg['cohort']['requests_per_user']
           for user in users):
        raise ValueError('request count per user differs from preregistration')
    if len(args.gpus) < len(users):
        raise ValueError('one isolated GPU per development user required')
    if args.reader_quantization != prereg['candidate_arm']['reader_quantization']:
        raise ValueError('reader quantization differs from preregistration')
    if args.weight_bits != prereg['candidate_arm']['reader_weight_bits']:
        raise ValueError('weight bits differ from preregistration')
    if args.group_size != prereg['candidate_arm']['group_size']:
        raise ValueError('group size differs from preregistration')
    args.output_root.mkdir(parents=True)
    jobs = []
    for user, gpu in zip(users, args.gpus):
        registration = json.loads(
            (args.registration_root / f'baseline_user_{user}.registration.json').read_text())
        jobs.append((user, gpu, registration))
    results = []
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        futures = {
            pool.submit(run_user, registration, cohort, args.cohort, user,
                        args.output_root, gpu, args.reader_quantization,
                        args.weight_bits, args.group_size): (user, gpu)
            for user, gpu, registration in jobs
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    failed = [result for result in results if result['returncode'] != 0]
    if failed:
        raise SystemExit(f'development arm failed: {failed}')
    for user in users:
        path = args.output_root / f'baseline_user_{user}.json'
        run = json.loads(path.read_text())
        if not run.get('complete') or run.get('evaluation_label_files_read') != []:
            raise ValueError(f'incomplete or label-reading output: {path}')
        if [row['question_id'] for row in run['rows']] != cohort['question_ids_by_user'][user]:
            raise ValueError(f'output order differs: {path}')
        if run.get('reader_quantization') != args.reader_quantization:
            raise ValueError(f'quantization differs: {path}')
        if args.weight_bits is not None and run.get('reader_weight_bits') != args.weight_bits:
            raise ValueError(f'weight bits differ: {path}')
    manifest = {
        'schema': 'hyperpim-sc-development-arm-v1',
        'complete': True,
        'preregistration': str(args.preregistration),
        'preregistration_sha256': digest(args.preregistration),
        'cohort': str(args.cohort),
        'cohort_sha256': digest(args.cohort),
        'reader_quantization': args.reader_quantization,
        'reader_weight_bits': args.weight_bits,
        'group_size': args.group_size,
        'requests': sum(len(ids) for ids in cohort['question_ids_by_user'].values()),
        'jobs': sorted(results, key=lambda item: int(item['user'])),
        'evaluation_label_files_read': [],
        'boundary': 'development-only 192-request quality generation; concurrent timings are not latency evidence',
    }
    (args.output_root / 'serving_manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'complete': True, 'output_root': str(args.output_root)},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
