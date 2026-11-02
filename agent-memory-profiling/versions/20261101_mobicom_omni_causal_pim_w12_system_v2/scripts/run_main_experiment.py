"""Prepare, freeze, and launch the single main MobileMem-Omni serving path."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml

from build_main_cohort import build as build_cohort
from extensions_config import load_extensions
from omni_prepare import prepare


ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict):
        raise ValueError('main experiment configuration must be a mapping')
    load_extensions(config)
    return config


def prepare_data(config: dict) -> dict:
    paths = config['paths']
    output = Path(paths['prepared'])
    if output.exists():
        return json.loads((output / 'audit.json').read_text())
    return prepare(Path(paths['data']), Path(paths['questions']), output,
                   Path(paths['image_archive']),
                   Path(paths['split_reference']) if paths.get('split_reference') else None)


def freeze_cohort(config: dict) -> dict:
    paths = config['paths']
    output = Path(paths['cohort'])
    if output.exists():
        return json.loads(output.read_text())
    return build_cohort(Path(paths['prepared']), output,
                        set(config['serving']['splits']))


def user_ids(config: dict, cohort: dict, requested: list[str] | None) -> list[str]:
    if requested:
        result = requested
    else:
        result = sorted(cohort['question_ids_by_user'], key=int)
    unknown = sorted(set(result) - set(cohort['question_ids_by_user']))
    if unknown:
        raise ValueError(f'unknown users in cohort: {unknown}')
    return result


def serve_one(config: dict, user_id: str, gpu: str, *, shard_index: int = 0,
              shard_count: int = 1) -> int:
    paths, models, serving = config['paths'], config['models'], config['serving']
    pim = config.get('pim', {})
    w12 = pim.get('w12', {}) if isinstance(pim, dict) else {}
    analysis = config.get('analysis', {})
    extensions_json = load_extensions(config).to_json()
    run_root = Path(paths['run_root'])
    run_root.mkdir(parents=True, exist_ok=True)
    output = run_root / f'user_{user_id}_shard_{shard_index}.json'
    log_path = run_root / 'logs' / f'user_{user_id}_shard_{shard_index}.log'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and json.loads(output.read_text()).get('complete'):
        return 0
    index = Path(paths['index_root']) / f'proof_index_user_{user_id}'
    command = [sys.executable, str(ROOT / 'scripts/run_proof_mainline.py'),
               '--prepared', paths['prepared'], '--image-archive', paths['image_archive'],
               '--reader', models['reader'], '--embedder', models['embedder'],
               '--reranker', models['reranker'], '--index', str(index),
               '--cohort', paths['cohort'], '--user-id', user_id, '--output', str(output),
               '--max-new-tokens', str(serving['max_new_tokens']),
               '--input-budget', str(serving['input_budget']),
               '--total-text-tokens', str(serving['total_text_tokens']),
               '--max-images', str(serving['max_images']),
               '--planner-time-limit', str(serving['planner_time_limit']),
               '--shard-index', str(shard_index), '--shard-count', str(shard_count),
               '--allowed-splits', *serving['splits'], '--query-contract',
               '--extensions-json', extensions_json]
    if serving.get('question_ids_file'):
        command.extend(['--question-ids-file', str(serving['question_ids_file'])])
    if analysis.get('multi_witness_shadow', False):
        command.append('--multi-witness-shadow')
    if analysis.get('independent_witness_rounds') is not None:
        command.extend(['--independent-witness-rounds',
                        str(int(analysis['independent_witness_rounds']))])
    if analysis.get('independent_witness_candidate_limit') is not None:
        command.extend(['--independent-witness-candidate-limit',
                        str(int(analysis['independent_witness_candidate_limit']))])
    if analysis.get('initial_witness_per_demand_limit') is not None:
        command.extend(['--initial-witness-per-demand-limit',
                        str(int(analysis['initial_witness_per_demand_limit']))])
    if analysis.get('witness_descriptor_index'):
        command.extend(['--witness-descriptor-index',
                        str(analysis['witness_descriptor_index'])])
    if analysis.get('witness_descriptor_shadow', False):
        command.append('--witness-descriptor-shadow')
    if analysis.get('witness_descriptor_per_demand_limit') is not None:
        command.extend(['--witness-descriptor-per-demand-limit',
                        str(int(analysis['witness_descriptor_per_demand_limit']))])
    if analysis.get('witness_descriptor_prerank_family_limit') is not None:
        command.extend(['--witness-descriptor-prerank-family-limit',
                        str(int(analysis['witness_descriptor_prerank_family_limit']))])
    functional_route = str(w12.get('functional_route', 'none'))
    if functional_route != 'none':
        command.extend([
            '--w12-functional-route', functional_route,
            '--reader-quantization', 'w12a16',
            '--w12-numeric-mode', str(w12.get('numeric_mode', 'bf16_reference')),
            '--w12-bank-count', str(int(w12.get('bank_count', 256))),
            '--w12-tile-rows', str(int(w12.get('tile_rows', 16))),
            '--w12-host-contract-rows', str(int(w12.get(
                'host_contract_rows', int(w12.get('bank_count', 256))
                * int(w12.get('tile_rows', 16))))),
            '--w12a16-group-size', str(int(w12.get('group_size', 128))),
        ])
        if w12.get('release_source', False):
            command.append('--w12-release-source')
        if w12.get('allow_decode_fallback', False):
            command.append('--w12-allow-decode-fallback')
    environment = dict(os.environ)
    environment['CUDA_VISIBLE_DEVICES'] = str(gpu)
    environment['PYTHONPATH'] = str(ROOT) + (
        ':' + environment['PYTHONPATH'] if environment.get('PYTHONPATH') else '')
    with log_path.open('a') as log:
        completed = subprocess.run(command, env=environment, stdout=log,
                                   stderr=subprocess.STDOUT, cwd=ROOT)
    return completed.returncode


def launch(config: dict, requested_users: list[str] | None) -> int:
    cohort = json.loads(Path(config['paths']['cohort']).read_text())
    users = user_ids(config, cohort, requested_users)
    gpus = [str(value) for value in config['execution']['gpus']]
    if not gpus:
        raise ValueError('at least one GPU is required')
    shard_count = int(config['execution'].get('shard_count', 1))
    jobs = [(user, gpus[index % len(gpus)]) for index, user in enumerate(users)]
    failures = [user for user, _ in jobs]
    retry_count = max(0, int(config['execution'].get('retries', 1)))
    for attempt in range(retry_count + 1):
        current = [(user, gpus[index % len(gpus)]) for index, user in enumerate(failures)]
        if not current:
            break
        failures = []
        with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            futures = {pool.submit(serve_one, config, user, gpu,
                                   shard_count=shard_count): (user, gpu)
                       for user, gpu in current}
            for future in as_completed(futures):
                user, gpu = futures[future]
                code = future.result()
                print(json.dumps({'user_id': user, 'gpu': gpu,
                                  'returncode': code, 'attempt': attempt}), flush=True)
                if code:
                    failures.append(user)
    if failures:
        raise SystemExit(f'main serving jobs failed: {sorted(failures, key=int)}')
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--mode', choices=['prepare', 'cohort', 'serve', 'launch', 'all'], required=True)
    parser.add_argument('--users', nargs='*')
    parser.add_argument('--user-id')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--shard-count', type=int, default=1)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.mode in {'prepare', 'all'}:
        print(json.dumps(prepare_data(config), ensure_ascii=False, indent=2))
    if args.mode in {'cohort', 'all'}:
        print(json.dumps(freeze_cohort(config), ensure_ascii=False, indent=2))
    if args.mode == 'serve':
        if not args.user_id:
            raise ValueError('--user-id is required for serve mode')
        raise SystemExit(serve_one(config, args.user_id, args.gpu,
                                   shard_index=args.shard_index, shard_count=args.shard_count))
    if args.mode == 'launch':
        raise SystemExit(launch(config, args.users))
    if args.mode == 'all':
        raise SystemExit(launch(config, args.users))


if __name__ == '__main__':
    main()
