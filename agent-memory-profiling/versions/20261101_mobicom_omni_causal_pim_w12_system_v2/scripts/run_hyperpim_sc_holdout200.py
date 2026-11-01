#!/usr/bin/env python3
"""Run the preregistered 200-request holdout pair on isolated workers."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def command_for(protocol: dict, job: dict, output: Path) -> list[str]:
    common = protocol['common_arguments']
    arm = protocol['arms'][job['arm']]['arguments']
    command = [
        sys.executable, str(ROOT / 'scripts/run_omni_raw_image_smoke.py'),
        '--prepared', common['prepared'],
        '--image-archive', common['image_archive'],
        '--reader', common['reader'],
        '--embedder', common['embedder'],
        '--reranker', common['reranker'],
        '--output', str(output),
        '--user-id', str(job['user']),
        '--limit', str(common['per_user']),
        '--cohort', protocol['cohort']['path'],
        '--allowed-splits', *common['allowed_splits'],
        '--screening-run',
        '--device', 'cuda:0',
        '--candidate-sessions', str(common['candidate_sessions']),
        '--max-images', str(common['max_images']),
        '--max-new-tokens', str(common['max_new_tokens']),
        '--session-text-tokens', str(common['session_text_tokens']),
        '--total-text-tokens', str(common['total_text_tokens']),
        '--evidence-policy', common['evidence_policy'],
        '--disable-edge-reuse',
        '--image-max-pixels', str(common['image_max_pixels']),
        '--planner-time-limit', str(common['planner_time_limit']),
        '--reader-quantization', arm['reader_quantization'],
        '--w8a16-group-size', str(arm['w8a16_group_size']),
        '--w8a16-residual-bits', str(arm['w8a16_residual_bits']),
        '--w8a16-fallback-scope', arm['w8a16_fallback_scope'],
        '--hardware-identity', 'RTX3090:isolated-holdout-worker',
        '--concurrency-contract', common['concurrency_contract'],
        '--startup-policy', common['startup_policy'],
    ]
    if arm['reader_quantization'] == 'w12a16':
        if 'w12a16_group_size' not in arm:
            raise ValueError('W12A16 holdout arm requires explicit group size')
        command.extend(['--w12a16-group-size', str(arm['w12a16_group_size'])])
    return command


def run_job(protocol: dict, job: dict, gpu: str, serving: Path) -> dict:
    arm_root = serving / job['arm']
    arm_root.mkdir(parents=True, exist_ok=True)
    output = arm_root / f"baseline_user_{job['user']}.json"
    log = serving / 'logs' / f"{job['arm']}_user_{job['user']}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        return {'arm': job['arm'], 'user': str(job['user']), 'gpu': gpu,
                'returncode': 0, 'skipped_existing': True}
    command = command_for(protocol, job, output)
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
    return {'arm': job['arm'], 'user': str(job['user']), 'gpu': gpu,
            'returncode': completed.returncode, 'seconds': time.time() - started,
            'output': str(output), 'output_sha256': digest(output) if output.exists() else None,
            'log': str(log)}


def run_queued(protocol: dict, job: dict, serving: Path,
               gpu_queue: queue.Queue[str]) -> dict:
    gpu = gpu_queue.get()
    try:
        return run_job(protocol, job, gpu, serving)
    finally:
        gpu_queue.put(gpu)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    if protocol.get('status') != 'registered_before_generation':
        raise ValueError('pre-generation protocol required')
    cohort = Path(protocol['cohort']['path'])
    if digest(cohort) != protocol['cohort']['sha256']:
        raise ValueError('cohort identity differs from registration')
    cohort_document = json.loads(cohort.read_text())
    gate = Path(protocol['quality_dependency']['evidence'])
    if digest(gate) != protocol['quality_dependency']['evidence_sha256']:
        raise ValueError('quality dependency identity differs from registration')
    jobs = protocol['jobs']
    if sorted((job['arm'], str(job['user'])) for job in jobs) != sorted(
            (arm, str(user)) for arm in protocol['arms'] for user in protocol['cohort']['users']):
        raise ValueError('job matrix differs from registered arms/users')
    if args.dry_run:
        print(json.dumps([command_for(protocol, job, ROOT / 'dry-run.json') for job in jobs],
                         ensure_ascii=False, indent=2))
        return
    serving = Path(protocol['serving_root'])
    if serving.exists():
        raise ValueError('fresh serving root required')
    serving.mkdir(parents=True)
    gpu_queue: queue.Queue[str] = queue.Queue()
    for gpu in protocol['gpus']:
        gpu_queue.put(str(gpu))
    results = []
    with ThreadPoolExecutor(max_workers=len(protocol['gpus'])) as pool:
        futures = {pool.submit(run_queued, protocol, job, serving, gpu_queue): job
                   for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                result = future.result()
            except Exception as error:
                result = {'arm': job['arm'], 'user': str(job['user']),
                          'returncode': 999, 'error': repr(error)}
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    failed = [result for result in results if result.get('returncode') != 0]
    manifest = {
        'schema': 'hyperpim-sc-holdout200-serving-manifest-v1',
        'protocol': str(args.protocol),
        'protocol_sha256': digest(args.protocol),
        'cohort_sha256': digest(cohort),
        'jobs': sorted(results, key=lambda item: (item.get('arm', ''), item.get('user', ''))),
        'complete': not failed and len(results) == len(jobs),
        'evaluation_label_files_read': [],
        'boundary': 'concurrent holdout quality generation only; timings are not latency evidence',
    }
    (serving / 'serving_manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    if failed:
        raise SystemExit(f'holdout serving failed: {failed}')
    for arm in protocol['arms']:
        for user in protocol['cohort']['users']:
            path = serving / arm / f'baseline_user_{user}.json'
            run = json.loads(path.read_text())
            if not run.get('complete') or run.get('evaluation_label_files_read') != []:
                raise SystemExit(f'incomplete or label-reading output: {path}')
            if [row['question_id'] for row in run['rows']] != cohort_document['question_ids_by_user'][str(user)]:
                raise SystemExit(f'cohort order differs: {path}')
    print(json.dumps({'complete': True, 'serving_root': str(serving), 'jobs': len(jobs)},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
