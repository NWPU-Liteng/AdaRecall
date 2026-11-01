"""One frozen development main comparison, followed by the unchanged 8B judge."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from audit_omni_recall_width import paired_quality
from audit_omni_smoke_closure import digest, empirical_latency
from diagnose_omni_cohort_evidence import diagnose_row

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def environment(gpu):
    return {**os.environ, 'CUDA_VISIBLE_DEVICES': gpu, 'OMP_NUM_THREADS': '2',
            'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2', 'TOKENIZERS_PARALLELISM': 'false'}


def summarize(root, prepared):
    cohort = json.loads((root / 'cohort.json').read_text())
    judge = json.loads((root / 'judge.json').read_text())
    labels = {row['question_id']: row for row in map(json.loads, (prepared / 'labels.jsonl').read_text().splitlines())}
    graphs = {item['person_id']: item for item in json.loads((prepared / 'omni_graph.json').read_text())}
    judged = {(row['run'], row['question_id']): row for row in judge['rows']}
    paired, policies, inputs = [], {}, {}
    correctness = {}
    for policy in ('reference', 'candidate'):
        rows, diagnostics, indices = [], [], []
        for user, question_ids in cohort['question_ids_by_user'].items():
            path = root / f'{policy}_user_{user}.json'
            run = json.loads(path.read_text())
            if not run.get('complete') or [row['question_id'] for row in run['rows']] != question_ids:
                raise ValueError('whole-cohort source order and denominator required')
            if judge['provenance']['run_sha256'][str(path)] != digest(path):
                raise ValueError('Judge does not bind current saved answers')
            inputs[str(path)] = digest(path)
            atoms = {atom['atom_id']: atom for atom in graphs[user]['atoms']}
            for row in run['rows']:
                judgment = judged[(str(path), row['question_id'])]
                if judgment['prediction_sha256'] != row['answer_sha256']:
                    raise ValueError('judged prediction differs')
                correct = judgment['parse_valid'] and judgment['judge_label'] == 'CORRECT'
                correctness[(policy, row['question_id'])] = bool(correct)
                rows.append({**row, 'judge_correct': bool(correct), 'judge_valid': judgment['parse_valid']})
                diagnostics.append(diagnose_row(row, labels[row['question_id']], atoms))
            indices.append({'person_id': user, 'model_load_seconds': run['model_load_seconds'],
                            'construction_seconds': run['construction_seconds'],
                            'joint_index': run['joint_index_metadata'],
                            'model_loaded_cuda_allocated_bytes': run['model_loaded_cuda_allocated_bytes']})
        stages = {name: sum(row[name] for row in rows) for name in
                  ('query_retrieval_seconds', 'edge_embedding_seconds', 'evidence_planning_seconds',
                   'prefill_seconds', 'decode_seconds')}
        coverage = {}
        for name in ('source_session_recall_candidates', 'source_session_recall_selected',
                     'label_image_recall_candidates', 'label_image_recall_selected'):
            values = [row[name] for row in diagnostics if row[name] is not None]
            coverage[name] = sum(values) / len(values) if values else None
        policies[policy] = {'requests': len(rows), 'correct': sum(row['judge_correct'] for row in rows),
            'strict_eos_correct': sum(row['judge_correct'] and row['stop_reason'] == 'eos' for row in rows),
            'invalid_judgments': sum(not row['judge_valid'] for row in rows),
            'service_latency': empirical_latency([row['service_seconds'] for row in rows]),
            'ttft_latency': empirical_latency([row['ttft_seconds'] for row in rows]),
            'stage_seconds': stages, 'output_tokens': sum(len(row['answer_token_ids']) for row in rows),
            'truncated_question_ids': [row['question_id'] for row in rows if row['stop_reason'] != 'eos'],
            'peak_cuda_allocated_bytes': max(row['service_peak_cuda_allocated_bytes'] for row in rows),
            'coverage': coverage, 'startup_and_indices': indices, 'diagnostic_rows': diagnostics}
    for user, identities in cohort['question_ids_by_user'].items():
        paired.extend({'person_id': user, 'question_id': identity,
                       'reference_correct': correctness[('reference', identity)],
                       'expanded_correct': correctness[('candidate', identity)]} for identity in identities)
    before = policies['reference']['service_latency']['sum_seconds']
    after = policies['candidate']['service_latency']['sum_seconds']
    return {'complete': True, 'policies': policies, 'quality': paired_quality(paired),
            'per_user_quality': {user: paired_quality([row for row in paired if row['person_id'] == user])
                                 for user in cohort['question_ids_by_user']},
            'screening_service_ratio': before / after, 'measured_causal_end_to_end_speedup': None,
            'full_system_pim_speedup': None, 'main_experiment_closed': False,
            'inputs_sha256': inputs, 'judge_sha256': digest(root / 'judge.json'),
            'boundary': 'one frozen development comparison; concurrent GPU-isolated/shared-host screening; 8B proxy Judge unchanged; no ablations, validation, holdout, measured PIM or publication-readiness claim'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--index-root', type=Path, required=True)
    parser.add_argument('--cohort', type=Path, required=True)
    parser.add_argument('--gpus', nargs=6, default=['0', '1', '2', '3', '4', '5'])
    parser.add_argument('--judge-gpu', default='6')
    parser.add_argument('--compiler-retrieval', choices=('exchange', 'evidence_units'), default='evidence_units')
    args = parser.parse_args()
    root = args.run_root.resolve()
    if root.exists() or len(set([*args.gpus, args.judge_gpu])) != 7:
        raise ValueError('fresh main comparison directory and disjoint GPUs required')
    root.mkdir(parents=True)
    runtime = root / 'runtime'
    for folder in ('scripts', 'pim_backend'):
        shutil.copytree(ROOT / folder, runtime / folder, ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copyfile(args.cohort, root / 'cohort.json')
    cohort = json.loads(args.cohort.read_text())
    prepared = ROOT / 'artifacts/omni_paper_scope_v3_20261021/prepared'
    if cohort['prepared_audit_sha256'] != digest(prepared / 'audit.json'):
        raise ValueError('registered cohort refers to another prepared release')
    jobs = []
    gpu_index = 0
    for policy in ('reference', 'candidate'):
        for user, identities in cohort['question_ids_by_user'].items():
            output = root / f'{policy}_user_{user}.json'
            command = [sys.executable, str(runtime / 'scripts/run_omni_raw_image_smoke.py'),
                '--prepared', str(prepared), '--image-archive', str(ROOT.parents[2] / 'MobileMem-main/data/omni/image.zip'),
                '--output', str(output), '--user-id', user, '--limit', str(len(identities)),
                '--cohort', str(root / 'cohort.json'), '--screening-run', '--reader-patch-projection', 'linear',
                '--reader-quantization', 'bf16', '--max-images', '4', '--image-max-pixels', '262144',
                '--max-new-tokens', '512', '--input-budget', '8192', '--total-text-tokens', '3600',
                '--hardware-identity', 'RTX3090:gpu' + args.gpus[gpu_index],
                '--concurrency-contract', 'six independent GPU workers, shared CPU/storage; screening only',
                '--query-local-token-cost-reuse']
            if policy == 'candidate':
                index_path = (args.index_root / f'index_user_{user}').resolve()
                if not json.loads((index_path / 'index.json').read_text()).get('complete'):
                    raise ValueError('complete offline evidence index required')
                command.extend(['--evidence-policy', 'evidence_compiler', '--relevance-modality', 'joint_index',
                    '--joint-index', str(index_path), '--joint-embedding-model', str(ROOT.parents[1] / 'models/Qwen3-VL-Embedding-2B'),
                    '--joint-recall-count', '64', '--planner-time-limit', '5'])
                command.extend(['--compiler-retrieval', args.compiler_retrieval])
            else:
                command.extend(['--evidence-policy', 'local_hyperedge', '--candidate-sessions', '4'])
            jobs.append({'policy': policy, 'user': user, 'gpu': args.gpus[gpu_index],
                         'output': str(output), 'command': command})
            gpu_index += 1
    write(root / 'protocol.json', {'registered_before_generation': True, 'jobs': jobs,
        'cohort_sha256': digest(root / 'cohort.json'), 'prepared_audit_sha256': digest(prepared / 'audit.json'),
        'source_files_sha256': {str(path.relative_to(runtime)): digest(path) for path in runtime.rglob('*')
                               if path.is_file() and path.suffix in {'.py', '.json'}},
        'judge': 'Qwen3-VL-8B-Instruct, unchanged strict JSON prompt',
        'treatment': 'offline evidence retrieval plus budgeted provenance-closure compilation',
        'compiler_retrieval': args.compiler_retrieval,
        'validation_accessed': False, 'holdout_accessed': False, 'measurement': 'shared-host screening only'})
    running = []
    started = time.perf_counter()
    for job in jobs:
        log = (root / f'{job["policy"]}_user_{job["user"]}.log').open('x')
        process = subprocess.Popen(job['command'], stdout=log, stderr=subprocess.STDOUT,
                                   env=environment(job['gpu']))
        running.append((job, process, log))
    statuses = []
    for job, process, log in running:
        status = process.wait()
        log.close()
        statuses.append({'output': job['output'], 'exit_code': status})
    write(root / 'worker_status.json', {'workers': statuses, 'wall_seconds': time.perf_counter() - started})
    if any(item['exit_code'] for item in statuses):
        raise RuntimeError('incomplete main comparison; inspect worker logs, no partial scoring')
    judge_command = [sys.executable, str(runtime / 'scripts/judge_omni_closure.py'),
        '--prepared', str(prepared), '--runs', *[job['output'] for job in jobs],
        '--prompt-source', str(ROOT.parents[2] / 'MobileMem-main/omni/eval/eval/question_answering_and_judge_prompts.txt'),
        '--output', str(root / 'judge.json')]
    with (root / 'judge.log').open('x') as log:
        subprocess.run(judge_command, stdout=log, stderr=subprocess.STDOUT,
                       env=environment(args.judge_gpu), check=True)
    summary = summarize(root, prepared)
    write(root / 'summary.json', summary)
    print(json.dumps({name: summary[name] for name in ('complete', 'quality', 'screening_service_ratio',
                                                      'main_experiment_closed')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
