"""Build all per-user proof indexes after the visual locator source is sealed."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def build_one(user_id: str, gpu: str, args) -> tuple[str, int]:
    output = args.output_root / f'proof_index_user_{user_id}'
    log = args.output_root / 'logs' / f'user_{user_id}.log'
    if (output / 'index.json').is_file():
        payload = json.loads((output / 'index.json').read_text())
        if payload.get('complete'):
            return user_id, 0
    command = [sys.executable, str(ROOT / 'scripts/build_omni_proof_index.py'),
               '--prepared', str(args.prepared), '--locator-root', str(args.locator_root),
               '--model', str(args.model), '--version-root', str(ROOT), '--user-id', user_id,
               '--output', str(output), '--batch-size', str(args.batch_size),
               '--allowed-splits', 'development', 'validation', 'holdout']
    env = dict(os.environ)
    env['CUDA_VISIBLE_DEVICES'] = gpu
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('w') as handle:
        code = subprocess.run(command, env=env, cwd=ROOT, stdout=handle,
                              stderr=subprocess.STDOUT).returncode
    return user_id, code


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--locator-root', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--users', nargs='+', required=True)
    parser.add_argument('--gpus', nargs='+', required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()
    if not (args.locator_root / 'status.json').is_file():
        raise ValueError('sealed visual locator root is required')
    args.output_root.mkdir(parents=True, exist_ok=True)
    failures = []
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        futures = {pool.submit(build_one, user, args.gpus[i % len(args.gpus)], args): user
                   for i, user in enumerate(args.users)}
        for future in as_completed(futures):
            user, code = future.result()
            print(json.dumps({'user_id': user, 'returncode': code}), flush=True)
            if code:
                failures.append(user)
    if failures:
        raise SystemExit(f'index jobs failed: {sorted(failures, key=int)}')


if __name__ == '__main__':
    main()
