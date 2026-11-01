"""Freeze one candidate, check six label-free requests, then run the full cohort."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from omni_proof_runtime import file_digest, write_json


def run_jobs(jobs):
    processes, logs = [], []
    try:
        for command, gpu, log_path in jobs:
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                       OPENBLAS_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false')
            handle = log_path.open('x')
            logs.append(handle)
            processes.append((subprocess.Popen(command, env=env, stdout=handle, stderr=subprocess.STDOUT), log_path))
        errors = []
        for process, path in processes:
            status = process.wait()
            print(json.dumps({'log': str(path), 'returncode': status}), flush=True)
            if status:
                errors.append(str(path))
        if errors:
            raise RuntimeError('staged execution failed; preserve all outputs: ' + ', '.join(errors))
    finally:
        for handle in logs:
            handle.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--stage', choices=['smoke', 'full'], required=True)
    parser.add_argument('--gpus', default='0,1,2,3,4,5,6')
    args = parser.parse_args()
    version = Path(__file__).resolve().parents[1]
    root = args.root.resolve()
    prepared = version / 'artifacts/omni_paper_scope_v3_20261021/prepared'
    previous = version / 'artifacts/proof_recovery_dev3x64_contract_v2_20261025'
    locator = previous / 'visual_merged'
    archive = version / '../../../MobileMem-main/data/omni/image.zip'
    model_root = Path('/data2/user25263093/.cache/modelscope/hub/models/Qwen')
    reader, embedder, reranker = (model_root / name for name in
        ('Qwen3___5-4B', 'Qwen3-Embedding-0.6B', 'Qwen3-Reranker-0.6B'))
    gpus = args.gpus.split(',')
    if len(gpus) < 3 or len(set(gpus)) != len(gpus):
        raise ValueError('at least three distinct available GPUs required')
    code = root / 'runtime_mainline/scripts'

    def command(script, *arguments):
        return [sys.executable, '-u', str(code / script), *map(str, arguments)]

    if args.stage == 'smoke':
        root.mkdir(parents=True, exist_ok=False)
        code.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(version / 'scripts', code, ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copytree(version / 'pim_backend', code.parent / 'pim_backend', ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copyfile(previous / 'cohort.json', root / 'cohort.json')
        shutil.copyfile(version / 'reports/PROOF_ATOM_REGION_V3_PROTOCOL_20261025_ZH.md', root / 'design_protocol.md')
        subprocess.run(command('register_proof_recovery.py', '--root', root, '--prepared', prepared,
            '--reader', reader, '--embedder', embedder, '--reranker', reranker,
            '--candidate', root.name), check=True)
        cohort = json.loads((root / 'cohort.json').read_text())
        for user, identities in cohort['question_ids_by_user'].items():
            write_json(root / f'smoke_ids_user_{user}.json', identities[:2])
        write_json(root / 'staging_protocol.json', {
            'smoke_selection': 'first two original-order IDs per registered development user; no labels',
            'smoke_question_ids_by_user': {user: identities[:2]
                for user, identities in cohort['question_ids_by_user'].items()},
            'design_protocol_sha256': file_digest(root / 'design_protocol.md'),
            'prior_locator_root': str(locator), 'prior_protocol_sha256': file_digest(previous / 'protocol.json'),
            'full_cohort_includes_smoke_requests': True, 'gpus': gpus,
            'parameter_selection_from_smoke_judge': False, 'baseline_or_ablation_launched': False})
        jobs = [(command('build_omni_proof_index.py', '--prepared', prepared, '--locator-root', locator,
                  '--model', embedder, '--version-root', version, '--user-id', user,
                  '--output', root / f'proof_index_user_{user}'), gpus[int(user)],
                 root / f'proof_index_user_{user}.log') for user in ('0', '1', '2')]
        run_jobs(jobs)
        subprocess.run(command('register_proof_index_manifest.py', '--root', root, '--locator-root', locator,
                               '--archive', archive), check=True)
        partitions = [(user, 0, 1, gpus[int(user)]) for user in ('0', '1', '2')]
    else:
        audit = json.loads((root / 'smoke_structural_audit.json').read_text())
        if not audit.get('structural_checks_passed') or audit.get('requests') != 6:
            raise ValueError('six-request label-free structural audit required before full generation')
        protocol = json.loads((root / 'protocol.json').read_text())
        for relative, digest in protocol['source_files_sha256'].items():
            if file_digest(root / 'runtime_mainline' / relative) != digest:
                raise ValueError('frozen candidate source changed')
        index_manifest = json.loads((root / 'index_manifest.json').read_text())
        if not index_manifest.get('complete') or file_digest(archive) != index_manifest['image_archive_sha256']:
            raise ValueError('frozen index manifest or source image archive changed')
        for user, entry in index_manifest['indexes'].items():
            index = root / f'proof_index_user_{user}'
            for name, field in [('index.json', 'index_sha256'), ('records.json', 'records_sha256'),
                                ('vectors.npy', 'vectors_sha256'), ('source.json', 'source_sha256'),
                                ('image_costs.json', 'image_costs_sha256')]:
                if file_digest(index / name) != entry[field]:
                    raise ValueError(f'frozen index changed for user {user}: {name}')
        partitions = []
        for gpu_index, gpu in enumerate(gpus):
            user, shard = str(gpu_index % 3), gpu_index // 3
            shard_count = sum(index % 3 == int(user) for index in range(len(gpus)))
            partitions.append((user, shard, shard_count, gpu))
    jobs = []
    for user, shard, count, gpu in partitions:
        name = f'smoke_user_{user}' if args.stage == 'smoke' else f'proof_user_{user}_shard_{shard}'
        invocation = command('run_proof_mainline.py', '--prepared', prepared, '--image-archive', archive,
            '--reader', reader, '--embedder', embedder, '--reranker', reranker,
            '--index', root / f'proof_index_user_{user}', '--cohort', root / 'cohort.json',
            '--user-id', user, '--shard-index', shard, '--shard-count', count,
            '--output', root / f'{name}.json', '--input-budget', 8192, '--total-text-tokens', 3600,
            '--max-images', 4, '--max-new-tokens', 512, '--planner-time-limit', 2)
        if args.stage == 'smoke':
            invocation += ['--question-ids-file', str(root / f'smoke_ids_user_{user}.json')]
        jobs.append((invocation, gpu, root / f'{name}.log'))
    run_jobs(jobs)
    if args.stage == 'full':
        subprocess.run(command('merge_proof_recovery_runs.py', '--root', root), check=True)
    subprocess.run(command('check_proof_atom_region.py', '--root', root,
                           '--prepared', prepared, '--stage', args.stage), check=True)
    if args.stage == 'full':
        run_jobs([(command('judge_omni_closure.py', '--prepared', prepared,
            '--runs', root / f'proof_user_{user}.json',
            '--prompt-source', version / '../../../MobileMem-main/omni/eval/eval/question_answering_and_judge_prompts.txt',
            '--output', root / f'judge_user_{user}.json'), gpus[int(user)], root / f'judge_user_{user}.log')
            for user in ('0', '1', '2')])
        workers = [(root / f'judge_user_{user}.json', json.loads((root / f'judge_user_{user}.json').read_text()))
                   for user in ('0', '1', '2')]
        if any(not worker.get('complete') for _, worker in workers):
            raise ValueError('all disjoint fixed-Judge partitions must complete')
        provenance = dict(workers[0][1]['provenance'])
        for _, worker in workers[1:]:
            for key in provenance:
                if key != 'run_sha256' and worker['provenance'][key] != provenance[key]:
                    raise ValueError('fixed-Judge protocol differs across workers: ' + key)
        provenance['run_sha256'] = {key: value for _, worker in workers
                                    for key, value in worker['provenance']['run_sha256'].items()}
        write_json(root / 'judge.json', {'complete': True,
            'rows': [row for _, worker in workers for row in worker['rows']],
            'summaries': {key: value for _, worker in workers for key, value in worker['summaries'].items()},
            'provenance': provenance,
            'compute_shards': [{'path': str(path), 'sha256': file_digest(path)} for path, _ in workers]})
        subprocess.run(command('audit_proof_recovery_closure.py', '--root', root,
                               '--prepared', prepared, '--cohort', root / 'cohort.json'), check=True)


if __name__ == '__main__':
    main()
