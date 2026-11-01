#!/usr/bin/env python3
"""Dump exact Reader inputs for development-only quantization flips."""
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


def load_diagnostic_cohort(decomposition: dict, source: dict) -> dict:
    flips = decomposition.get('flips', [])
    if decomposition.get('complete') is not True or len(flips) != 13:
        raise ValueError('complete 13-flip development decomposition required')
    if sorted({str(row['person_id']) for row in flips}, key=int) != ['0', '1', '2']:
        raise ValueError('only development users 0,1,2 are permitted')
    requested = {user: set() for user in ('0', '1', '2')}
    for row in flips:
        requested[str(row['person_id'])].add(row['question_id'])
    question_ids_by_user = {}
    for user in ('0', '1', '2'):
        source_ids = source['question_ids_by_user'][user]
        selected = [question_id for question_id in source_ids if question_id in requested[user]]
        if set(selected) != requested[user]:
            raise ValueError('flip question is absent from the registered development cohort')
        question_ids_by_user[user] = selected
    return {
        'complete': True,
        'diagnostic_only': True,
        'selection': 'development-only flip questions in original registered cohort order',
        'serving_splits': ['development'],
        'prepared_audit_sha256': source['prepared_audit_sha256'],
        'question_ids_by_user': question_ids_by_user,
        'requests': sum(len(values) for values in question_ids_by_user.values()),
        'boundary': 'mechanism diagnosis only; no holdout users; no evaluation labels in generation',
    }


def build_command(registration: dict, user: str, count: int, cohort: Path,
                  output_root: Path) -> list[str]:
    arguments = dict(registration['arguments'])
    output = output_root / 'serving' / f'bf16_user_{user}.json'
    dump = output_root / 'inputs'
    replacements = {
        'output': str(output), 'user_id': user, 'limit': count, 'cohort': str(cohort),
        'device': 'cuda:0', 'reader_quantization': 'bf16',
        'w8a16_group_size': 128, 'w8a16_residual_bits': 0,
        'w8a16_fallback_scope': 'none', 'diagnostic_input_dump_dir': str(dump),
        'hardware_identity': 'RTX3090:development-flip-diagnostic',
    }
    arguments.update(replacements)
    command = [sys.executable, str(ROOT / 'scripts/run_omni_raw_image_smoke.py')]
    for key, value in arguments.items():
        if value is None or value is False:
            continue
        if value is True:
            command.append(f'--{key.replace("_", "-")}')
        else:
            command.extend([f'--{key.replace("_", "-")}', str(value)])
    return command


def run_user(registration: dict, user: str, cohort: dict, cohort_path: Path,
             output_root: Path, gpu: str) -> dict:
    count = len(cohort['question_ids_by_user'][user])
    command = build_command(registration, user, count, cohort_path, output_root)
    log = output_root / 'logs' / f'bf16_user_{user}.log'
    log.parent.mkdir(parents=True, exist_ok=True)
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
    output = output_root / 'serving' / f'bf16_user_{user}.json'
    return {
        'user': user, 'gpu': gpu, 'requests': count,
        'returncode': completed.returncode, 'seconds': time.time() - started,
        'output': str(output), 'output_sha256': digest(output) if output.exists() else None,
        'log': str(log),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--decomposition', type=Path, required=True)
    parser.add_argument('--source-cohort', type=Path, required=True)
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--gpus', nargs='+', default=['0', '1', '3'])
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if len(args.gpus) < 3:
        raise ValueError('three isolated GPUs required')
    decomposition = json.loads(args.decomposition.read_text())
    source = json.loads(args.source_cohort.read_text())
    registration = json.loads(args.registration.read_text())
    if registration.get('arguments', {}).get('reader_quantization') != 'bf16':
        raise ValueError('BF16 diagnostic registration required')
    cohort = load_diagnostic_cohort(decomposition, source)
    if args.dry_run:
        commands = {
            user: build_command(registration, user, len(cohort['question_ids_by_user'][user]),
                                args.output_root / 'cohort.json', args.output_root)
            for user in ('0', '1', '2')
        }
        print(json.dumps(commands, ensure_ascii=False, indent=2))
        return
    if args.output_root.exists():
        raise ValueError('fresh diagnostic output root required')
    args.output_root.mkdir(parents=True)
    cohort_path = args.output_root / 'cohort.json'
    cohort_path.write_text(json.dumps(cohort, ensure_ascii=False, indent=2) + '\n')
    jobs = [
        (user, gpu) for user, gpu in zip(cohort['question_ids_by_user'], args.gpus)
        if cohort['question_ids_by_user'][user]
    ]
    results = []
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        futures = {
            pool.submit(run_user, registration, user, cohort, cohort_path,
                        args.output_root, gpu): (user, gpu)
            for user, gpu in jobs
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    failed = [result for result in results if result['returncode'] != 0]
    if failed:
        raise SystemExit(f'development diagnostic dump failed: {failed}')
    for user in cohort['question_ids_by_user']:
        path = args.output_root / 'serving' / f'bf16_user_{user}.json'
        run = json.loads(path.read_text())
        if not run.get('complete') or run.get('evaluation_label_files_read') != []:
            raise ValueError(f'incomplete or label-reading output: {path}')
        if [row['question_id'] for row in run['rows']] != cohort['question_ids_by_user'][user]:
            raise ValueError(f'diagnostic output order differs: {path}')
        expected = set(cohort['question_ids_by_user'][user])
        observed = {path.name.removesuffix('.reader_inputs.pt')
                    for path in (args.output_root / 'inputs').glob('*.reader_inputs.pt')}
        if not expected <= observed:
            raise ValueError(f'missing diagnostic inputs for user {user}')
    manifest = {
        'schema': 'hyperpim-sc-development-flip-input-dump-v1',
        'complete': True,
        'decomposition': str(args.decomposition),
        'decomposition_sha256': digest(args.decomposition),
        'source_cohort': str(args.source_cohort),
        'source_cohort_sha256': digest(args.source_cohort),
        'diagnostic_cohort': str(cohort_path),
        'diagnostic_cohort_sha256': digest(cohort_path),
        'registration': str(args.registration),
        'registration_sha256': digest(args.registration),
        'jobs': sorted(results, key=lambda item: int(item['user'])),
        'evaluation_label_files_read': [],
        'boundary': ('development-only mechanism diagnosis; exact post-retrieval Reader inputs; '
                     'not a new quality result and not physical PIM execution'),
    }
    (args.output_root / 'manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'complete': True, 'requests': cohort['requests'],
                      'output_root': str(args.output_root)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
