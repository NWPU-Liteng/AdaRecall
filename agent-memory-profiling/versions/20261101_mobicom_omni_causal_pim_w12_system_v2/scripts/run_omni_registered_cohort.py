"""Freeze a label-free multi-user development cohort before launching GPU workers."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            result.update(block)
    return result.hexdigest()


def freeze_cohort(manifests, development_people, limit):
    if limit < 1 or not development_people or len(set(development_people)) != len(development_people):
        raise ValueError('positive per-user size and unique development identities required')
    cohorts = []
    for person_id in development_people:
        rows = [row for row in manifests if row['person_id'] == person_id
                and row['split'] == 'development'][:limit]
        if len(rows) != limit or any(row['visibility_protocol'] != 'history_end' for row in rows):
            raise ValueError('insufficient authorized development history-end requests')
        identities = [row['question_id'] for row in rows]
        if len(set(identities)) != limit:
            raise ValueError('duplicate development question identity')
        cohorts.append({'person_id': person_id, 'question_ids': identities})
    return cohorts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--image-archive', type=Path, required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--gpus', nargs='+', required=True)
    parser.add_argument('--per-user', type=int, default=16)
    parser.add_argument('--image-edges-per-session', type=int, default=0)
    parser.add_argument('--image-novelty-weight', type=float, default=0.0)
    parser.add_argument('--visual-cache-bytes', type=int, default=0)
    parser.add_argument('--edge-cache-bytes', type=int, default=64 * 1024**2)
    parser.add_argument('--disable-edge-reuse', action='store_true')
    parser.add_argument('--evidence-policy', choices=('local_hyperedge', 'evidence_compiler'), default='local_hyperedge')
    parser.add_argument('--joint-index-root', type=Path)
    parser.add_argument('--joint-embedding-model', type=Path,
                        default=ROOT.parents[1] / 'models/Qwen3-VL-Embedding-2B')
    parser.add_argument('--joint-recall-count', type=int, default=64)
    args = parser.parse_args()
    if (args.run_root.exists() or len(set(args.gpus)) != len(args.gpus)
            or args.per_user < 1 or args.image_edges_per_session < 0 or args.visual_cache_bytes < 0
            or args.edge_cache_bytes <= 0 or args.joint_recall_count < 1
            or (args.evidence_policy == 'evidence_compiler' and args.joint_index_root is None)
            or not math.isfinite(args.image_novelty_weight) or args.image_novelty_weight < 0):
        raise ValueError('fresh run root, distinct GPUs, positive size and nonnegative image quota required')
    prepared = args.prepared.resolve()
    split = json.loads((prepared / 'split.json').read_text())
    audit = json.loads((prepared / 'audit.json').read_text())
    for name in ('manifests.jsonl', 'split.json', 'omni_graph.json'):
        if digest(prepared / name) != audit['artifacts_sha256'][name]:
            raise ValueError('prepared serving input changed')
    manifests = [json.loads(line) for line in (prepared / 'manifests.jsonl').read_text().splitlines()]
    cohorts = freeze_cohort(manifests, split['development_people'], args.per_user)
    if len(args.gpus) != len(cohorts):
        raise ValueError('one dedicated GPU per development user required')
    args.run_root.mkdir(parents=True, exist_ok=False)
    source_names = ('scripts/run_omni_registered_cohort.py', 'scripts/run_omni_raw_image_smoke.py',
                    'scripts/omni_evidence_planner.py', 'scripts/omni_embedding_store.py',
                    'scripts/omni_embedding_producer.py', 'scripts/omni_service_measurement.py',
                    'scripts/omni_visual_feature_store.py',
                    'scripts/omni_image_selection.py', 'pim_backend/decode_gemv.py',
                    'pim_backend/lpddr_bank_pim.py', 'scripts/omni_evidence_compiler.py',
                    'scripts/omni_joint_index.py')
    source_hashes = {name: digest(ROOT / name) for name in source_names}
    hardware = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name',
                                        '--format=csv,noheader'], text=True)
    gpu_hardware = {line.split(',', 1)[0].strip(): line.strip() for line in hardware.splitlines()}
    if any(gpu not in gpu_hardware for gpu in args.gpus):
        raise ValueError('every assigned physical GPU must be identified before launch')
    cohort_registration_sha256 = hashlib.sha256(json.dumps({
        'prepared_audit_sha256': digest(prepared / 'audit.json'),
        'serving_manifest_sha256': digest(prepared / 'manifests.jsonl'),
        'image_archive_sha256': digest(args.image_archive),
        'cohorts': cohorts,
        'embedding_producer_policy': 'session-local-canonical-batch-v1',
    }, sort_keys=True).encode()).hexdigest()
    jobs = []
    for cohort, gpu in zip(cohorts, args.gpus):
        output = args.run_root.resolve() / f"user_{cohort['person_id']}.json"
        command = [sys.executable, str(ROOT / 'scripts/run_omni_raw_image_smoke.py'),
                   '--prepared', str(prepared), '--image-archive', str(args.image_archive.resolve()),
                   '--output', str(output), '--user-id', cohort['person_id'],
                   '--limit', str(args.per_user), '--evidence-policy', args.evidence_policy,
                   '--candidate-sessions', '4', '--total-text-tokens', '3600',
                   '--max-images', '2', '--image-max-pixels', '262144',
                   '--max-new-tokens', '512', '--max-hyperedges', '64']
        if args.evidence_policy == 'evidence_compiler':
            command.extend(['--relevance-modality', 'joint_index',
                            '--compiler-retrieval', 'evidence_units',
                            '--joint-index', str((args.joint_index_root / f'index_user_{cohort["person_id"]}').resolve()),
                            '--joint-embedding-model', str(args.joint_embedding_model.resolve()),
                            '--joint-recall-count', str(args.joint_recall_count)])
        command.extend(['--edge-cache-bytes', str(args.edge_cache_bytes),
                        '--cohort-registration-sha256', cohort_registration_sha256,
                        '--hardware-identity', gpu_hardware[gpu],
                        '--concurrency-contract', 'three_independent_user_workers_one_per_gpu',
                        '--startup-policy', 'model_load_outside_request_per_worker'])
        if args.disable_edge_reuse:
            command.append('--disable-edge-reuse')
        if args.image_edges_per_session:
            command.extend(['--image-edges-per-session', str(args.image_edges_per_session)])
        if args.image_novelty_weight:
            command.extend(['--image-novelty-weight', str(args.image_novelty_weight)])
        if args.visual_cache_bytes:
            command.extend(['--visual-cache-bytes', str(args.visual_cache_bytes), '--verify-generation'])
        jobs.append({**cohort, 'gpu': gpu, 'output': str(output), 'command': command})
    plan = {'schema_version': 'omni-registered-development-cohort-v1', 'jobs': jobs,
            'selection': 'first N in frozen serving-manifest order for every development user; no labels',
            'source_files_sha256': source_hashes, 'prepared_audit_sha256': digest(prepared / 'audit.json'),
            'serving_manifest_sha256': digest(prepared / 'manifests.jsonl'),
            'image_archive_sha256': digest(args.image_archive),
            'evaluator_label_files_read': [], 'max_new_tokens': 512,
            'image_edges_per_session': args.image_edges_per_session,
            'image_novelty_weight': args.image_novelty_weight,
            'visual_cache_bytes': args.visual_cache_bytes,
            'edge_cache_bytes': args.edge_cache_bytes,
            'disable_edge_reuse': args.disable_edge_reuse,
            'gpu_hardware': {gpu: gpu_hardware[gpu] for gpu in args.gpus},
            'embedding_producer_policy': 'session-local-canonical-batch-v1',
            'evidence_policy': args.evidence_policy,
            'joint_index_root': str(args.joint_index_root.resolve()) if args.joint_index_root else None,
            'joint_recall_count': args.joint_recall_count,
            'cohort_registration_sha256': cohort_registration_sha256,
            'visual_generation_validation_required': bool(args.visual_cache_bytes),
            'output_budget_reason': 'uniform 512-token ceiling for all users after observed 256-token truncations; changed protocol, not matched speedup evidence',
            'boundary': 'static history-end QA; per-user independent cold stores; user0 already development-exposed; users are not cold shards of one stream; no holdout authorization'}
    with (args.run_root / 'cohort_plan.json').open('x') as handle:
        json.dump(plan, handle, ensure_ascii=False, indent=2)
    running = []
    started = time.perf_counter()
    try:
        for job in jobs:
            log_path = args.run_root / f"user_{job['person_id']}.log"
            log = log_path.open('x')
            environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': job['gpu'], 'OMP_NUM_THREADS': '2',
                           'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2'}
            process = subprocess.Popen(job['command'], env=environment, stdout=log, stderr=subprocess.STDOUT)
            running.append((job, process, log))
        statuses = []
        for job, process, log in running:
            status = process.wait()
            log.close()
            statuses.append({'person_id': job['person_id'], 'exit_code': status, 'output': job['output']})
    finally:
        for _, process, log in running:
            if process.poll() is None:
                process.terminate()
                process.wait()
            if not log.closed:
                log.close()
    sources_unchanged = all(digest(ROOT / name) == value for name, value in source_hashes.items())
    result = {'workers': statuses, 'sources_unchanged': sources_unchanged,
              'all_workers_exited_zero': all(item['exit_code'] == 0 for item in statuses),
              'wall_seconds': time.perf_counter() - started,
              'boundary': 'worker orchestration only; quality and execution integrity require separate audit'}
    with (args.run_root / 'worker_status.json').open('x') as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result), flush=True)
    return int(not result['all_workers_exited_zero'] or not sources_unchanged)


if __name__ == '__main__':
    raise SystemExit(main())
