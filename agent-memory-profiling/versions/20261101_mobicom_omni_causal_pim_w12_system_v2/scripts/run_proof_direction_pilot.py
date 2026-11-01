"""Freeze and run one source-fact/operation diagnostic, not an accuracy search."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from audit_proof_direction import selected_jsonl
from check_proof_atom_region import check_run
from omni_proof_runtime import file_digest


ROOT = Path(__file__).resolve().parents[1]
SELECTION = {'0': ['0_q_83', '0_q_95', '0_q_123', '0_q_662'],
             '1': ['1_q_72', '1_q_78', '1_q_270', '1_q_663'],
             '2': ['2_q_116', '2_q_715', '2_q_747', '2_q_799']}


def save(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def environment(gpu):
    return {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu), 'OMP_NUM_THREADS': '2',
            'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2',
            'TOKENIZERS_PARALLELISM': 'false', 'HF_HUB_OFFLINE': '1'}


def register(root):
    if root.exists():
        raise ValueError('fresh pilot root required')
    previous = ROOT / 'artifacts/proof_atom_region_v4_schema_20261025'
    prepared = ROOT / 'artifacts/omni_paper_scope_v3_20261021/prepared'
    previous_protocol = json.loads((previous / 'protocol.json').read_text())
    cohort = json.loads((previous / 'cohort.json').read_text())
    identities = {identity for values in SELECTION.values() for identity in values}
    manifests = selected_jsonl(prepared / 'manifests.jsonl', identities)
    if (len(identities) != 12 or any(row['split'] != 'development' for row in manifests.values())
            or file_digest(prepared / 'audit.json') != previous_protocol['prepared_audit_sha256']):
        raise ValueError('development selection/prepared identity mismatch')
    for user, selected in SELECTION.items():
        if ([identity for identity in cohort['question_ids_by_user'][user] if identity in selected] != selected
                or any(manifests[identity]['person_id'] != user for identity in selected)):
            raise ValueError('source order and user partition must be preserved')
    for model in previous_protocol['models'].values():
        for filename, expected in model['files_sha256'].items():
            if file_digest(Path(model['path']) / filename) != expected:
                raise ValueError('previously registered model changed')
    root.mkdir(parents=True)
    runtime = root / 'runtime_mainline'
    for folder in ('scripts', 'pim_backend'):
        shutil.copytree(ROOT / folder, runtime / folder, ignore=shutil.ignore_patterns('__pycache__'))
    save(root / 'cohort.json', {**cohort, 'question_ids_by_user': SELECTION, 'per_user': 4,
         'selection': 'question-only diagnostic subset; targeted mechanism coverage, not representative accuracy'})
    jobs = []
    for user, identities_for_user in SELECTION.items():
        for shard in range(2):
            gpu = int(user) * 2 + shard
            output = root / f'user_{user}_shard_{shard}.json'
            command = [sys.executable, str(runtime / 'scripts/run_proof_mainline.py'),
                '--prepared', str(prepared), '--image-archive', str(ROOT.parents[2] / 'MobileMem-main/data/omni/image.zip'),
                '--reader', previous_protocol['models']['reader']['path'],
                '--embedder', previous_protocol['models']['embedder']['path'],
                '--reranker', previous_protocol['models']['reranker']['path'],
                '--index', str(previous / f'proof_index_user_{user}'), '--cohort', str(root / 'cohort.json'),
                '--user-id', user, '--shard-index', str(shard), '--shard-count', '2', '--output', str(output),
                '--query-contract', '--input-budget', '8192', '--total-text-tokens', '3600',
                '--max-images', '4', '--max-new-tokens', '512', '--planner-time-limit', '2']
            jobs.append({'gpu': gpu, 'user': user, 'shard': shard, 'output': str(output),
                         'question_ids': identities_for_user[shard::2], 'command': command})
    save(root / 'protocol.json', {**{key: value for key, value in previous_protocol.items()
                                   if key not in {'source_files_sha256', 'sharding', 'cohort_sha256'}},
        'candidate': 'source_fact_operation_pilot_v1', 'kind': 'targeted_development_pilot',
        'previous_protocol_sha256': file_digest(previous / 'protocol.json'),
        'query_only_diagnostic_sha256': file_digest(ROOT / 'artifacts/proof_direction_semantic_v1_20261025/summary.json'),
        'cohort_sha256': file_digest(root / 'cohort.json'), 'question_ids_by_user': SELECTION,
        'cases': [manifests[identity] for values in SELECTION.values() for identity in values],
        'prepared': str(prepared), 'index_root': str(previous), 'jobs': jobs,
        'source_files_sha256': {str(path.relative_to(runtime)): file_digest(path)
                              for path in sorted(runtime.rglob('*')) if path.is_file()},
        'index_sha256_by_user': {user: file_digest(previous / f'proof_index_user_{user}/index.json')
                                for user in SELECTION},
        'operations_contract': 'topologically ordered derived operations never become source obligations; full question retained',
        'evidence_query_and_or_contract': 'only source observations are conjunctive witness obligations; derived operation DAG remains a query plan, not evidence',
        'source_contract_repair': 'source observations versus derived operations; complete question in embedding, reranking and grounding; identical exact-render and Reader prompt',
        'contextual_policy': 'explicit nonproof budget-feasible package; no source-sufficiency certificate',
        'selection_uses_gold_or_judge': False, 'accuracy_generalization_claimed': False,
        'automatic_full_192_run': False, 'judge_after_structural_audit': True,
        'method_effectiveness_established': False,
        'measurement': 'shared server screening only; no matched comparison, physical PIM or causal speedup'})
    print(json.dumps({'registered': str(root), 'requests': 12, 'workers': len(jobs)}), flush=True)


def audit(root, protocol):
    manifests = {row['question_id']: row['question'] for row in protocol['cases']}
    summaries, rows = [], []
    for user, identities in protocol['question_ids_by_user'].items():
        index = Path(protocol['index_root']) / f'proof_index_user_{user}'
        if file_digest(index / 'index.json') != protocol['index_sha256_by_user'][user]:
            raise ValueError('registered offline index changed')
        records = json.loads((index / 'records.json').read_text())
        objects = json.loads((index / 'source.json').read_text())['objects']
        for job in (item for item in protocol['jobs'] if item['user'] == user):
            run = json.loads(Path(job['output']).read_text())
            summaries.extend(check_run(run, job['question_ids'], records, objects, manifests, protocol))
            rows.extend(run['rows'])
    expected = {identity for values in protocol['question_ids_by_user'].values() for identity in values}
    if len(rows) != len(expected) or {row['question_id'] for row in rows} != expected:
        raise ValueError('full pilot denominator required')
    result = {'complete': True, 'structural_checks_passed': True, 'requests': len(rows),
        'query_status': dict(Counter(row['parse_status'] for row in summaries)),
        'selection_modes': dict(Counter(row['selection_mode'] for row in summaries)),
        'operation_queries': sum(bool(row['operations']) for row in summaries),
        'multi_fact_queries': sum(row['demands'] > 1 for row in summaries),
        'all_reader_eos': all(row['stop_reason'] == 'eos' for row in rows),
        'run_sha256': {job['output']: file_digest(Path(job['output'])) for job in protocol['jobs']},
        'rows': summaries, 'evaluation_label_files_read': [], 'main_experiment_closed': False,
        'boundary': 'identity/derivation/processor-budget correctness only; no semantic sufficiency or gain certificate'}
    save(root / 'structural_audit.json', result)
    return rows, result


def run(root):
    protocol = json.loads((root / 'protocol.json').read_text())
    if (root / 'worker_status.json').exists():
        raise ValueError('pilot already ran; do not repeat or select by outcomes')
    if file_digest(root / 'cohort.json') != protocol['cohort_sha256']:
        raise ValueError('cohort changed')
    for relative, expected in protocol['source_files_sha256'].items():
        if file_digest(root / 'runtime_mainline' / relative) != expected:
            raise ValueError('frozen source changed: ' + relative)
    started = time.perf_counter()
    running = []
    for job in protocol['jobs']:
        log = (root / (Path(job['output']).stem + '.log')).open('x')
        process = subprocess.Popen(job['command'], stdout=log, stderr=subprocess.STDOUT,
                                   env=environment(job['gpu']))
        running.append((job, process, log))
    statuses = []
    for job, process, log in running:
        status = process.wait()
        log.close()
        statuses.append({'output': job['output'], 'exit_code': status})
    save(root / 'worker_status.json', {'workers': statuses, 'wall_seconds': time.perf_counter() - started})
    if any(item['exit_code'] for item in statuses):
        raise RuntimeError('incomplete pilot; no survivor-only scoring')
    rows, structure = audit(root, protocol)
    command = [sys.executable, str(root / 'runtime_mainline/scripts/judge_omni_closure.py'),
        '--prepared', protocol['prepared'], '--runs', *[job['output'] for job in protocol['jobs']],
        '--prompt-source', str(ROOT.parents[2] / 'MobileMem-main/omni/eval/eval/question_answering_and_judge_prompts.txt'),
        '--output', str(root / 'judge.json')]
    with (root / 'judge.log').open('x') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=environment(6), check=True)
    judge = json.loads((root / 'judge.json').read_text())
    if not judge['complete'] or len(judge['rows']) != len(rows):
        raise ValueError('Judge denominator mismatch')
    by_id = {row['question_id']: row for row in rows}
    if (len({item['question_id'] for item in judge['rows']}) != len(rows)
            or judge['provenance']['run_sha256'] != structure['run_sha256']):
        raise ValueError('Judge prediction/run binding failed')
    judged_rows = []
    for judgment in judge['rows']:
        row = by_id[judgment['question_id']]
        if judgment['prediction_sha256'] != row['answer_sha256']:
            raise ValueError('Judge did not score the saved answer')
        judged_rows.append({'question_id': row['question_id'],
            'correct': bool(judgment['parse_valid'] and judgment['judge_label'] == 'CORRECT'),
            'judge_valid': judgment['parse_valid'], 'answer': row['answer'],
            'source_facts': row['proof_plan']['demands'],
            'operations': row['proof_plan']['query_operations'],
            'selection_mode': row['proof_plan']['selection_mode']})
    correct = sum(row['correct'] for row in judged_rows)
    summary = {'complete': True, 'requests': len(rows), 'correct': correct,
        'accuracy': correct / len(rows), 'invalid_judgments': sum(not row['judge_valid'] for row in judged_rows),
        'structural_audit_sha256': file_digest(root / 'structural_audit.json'),
        'judge_sha256': file_digest(root / 'judge.json'), 'query_status': structure['query_status'],
        'multi_fact_queries': structure['multi_fact_queries'], 'operation_queries': structure['operation_queries'],
        'service_seconds': sum(row['service_seconds'] for row in rows),
        'source_grounding_seconds': sum(row['proof_plan']['grounding']['seconds'] for row in rows),
        'reader_decode_seconds': sum(row['decode_seconds'] for row in rows),
        'method_effectiveness_established': False, 'main_experiment_closed': False,
        'baseline_or_ablation_launched': False, 'validation_accessed': False, 'holdout_accessed': False,
        'rows': judged_rows,
        'boundary': 'targeted development diagnostic, no matched gain estimate or independent validation'}
    save(root / 'summary.json', summary)
    print(json.dumps({key: value for key, value in summary.items() if key != 'rows'},
                     ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--mode', choices=['register', 'run'], required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    register(root) if args.mode == 'register' else run(root)


if __name__ == '__main__':
    main()
