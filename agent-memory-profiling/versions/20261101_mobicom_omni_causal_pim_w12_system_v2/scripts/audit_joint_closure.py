"""Paired development uncertainty, producer amortization, and capacity accounting."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from omni_joint_index import digest


def paired_uncertainty(reference, candidate, seed=0, repeats=20000):
    if not reference or set(reference) != set(candidate):
        raise ValueError('nonempty fully paired requests required')
    identities = sorted(reference)
    differences = np.array([int(candidate[identity] == 'CORRECT') - int(reference[identity] == 'CORRECT')
                            for identity in identities], dtype=np.float64)
    invalid = sum(reference[identity] not in ('CORRECT', 'WRONG')
                  or candidate[identity] not in ('CORRECT', 'WRONG') for identity in identities)
    wins, losses = int((differences > 0).sum()), int((differences < 0).sum())
    generator = np.random.default_rng(seed)
    bootstrap = differences[generator.integers(0, len(differences), size=(repeats, len(differences)))].mean(axis=1)
    return {'requests': len(differences), 'accuracy_difference': float(differences.mean()),
            'wins': wins, 'losses': losses, 'invalid_pairs': invalid,
            'question_iid_bootstrap_95pct': np.quantile(bootstrap, [.025, .975]).tolist(),
            'mcnemar_exact_two_sided_p': float(binomtest(wins, wins + losses).pvalue) if wins + losses else 1.,
            'population_noninferiority_established': False,
            'boundary': 'exploratory question-IID interval; only three development users; not user-cluster generalization; invalid judge retained as not correct; no noninferiority margin registered'}


def amortization(baseline_total, candidate_total, baseline_startup, candidate_startup,
                 candidate_producer, requests):
    if requests < 1 or any(value < 0 for value in (baseline_total, candidate_total,
                          baseline_startup, candidate_startup, candidate_producer)):
        raise ValueError('positive request denominator and nonnegative measured costs required')
    saving = (baseline_total - candidate_total) / requests
    extra = candidate_startup + candidate_producer - baseline_startup
    break_even = max(0, math.ceil(extra / saving)) if saving > 0 else None
    return {'online_saving_per_query_seconds_screening_only': saving,
            'additional_covered_initialization_seconds': extra,
            'candidate_producer_seconds': candidate_producer,
            'covered_lifecycle_baseline_seconds': baseline_startup + baseline_total,
            'covered_lifecycle_candidate_seconds': candidate_startup + candidate_producer + candidate_total,
            'linear_extrapolation_break_even_queries': break_even,
            'boundary': 'sum of covered serial phase times, not measured cold wall-clock; excludes graph loading, registration and uninstrumented startup; extrapolation assumes unchanged workload/marginal cost; shared-host screening not causal performance'}


def audit(root, judge_path):
    root = root.resolve()
    judge = json.loads(judge_path.read_text())
    if not judge.get('complete'):
        raise ValueError('complete judge required')
    judged = {(str(Path(row['run']).resolve()), row['question_id']): row for row in judge['rows']}
    verdicts = {policy: {} for policy in ('local_hyperedge', 'provenance_milp')}
    totals = {policy: {'service': 0., 'startup_covered': 0.} for policy in verdicts}
    runs, producer_total, index_bytes = [], 0., 0
    model_paths = {}
    for policy in verdicts:
        for person in ('0', '1', '2'):
            path = root / f'{policy}_user_{person}.json'
            run = json.loads(path.read_text())
            expected_hash = judge['provenance']['run_sha256'].get(str(path))
            if not run.get('complete') or expected_hash != digest(path):
                raise ValueError('changed, incomplete or unjudged run')
            registration = json.loads(path.with_suffix('.registration.json').read_text())
            if [row['question_id'] for row in run['rows']] != registration['question_ids']:
                raise ValueError('request denominator changed')
            for row in run['rows']:
                verdict = judged[(str(path), row['question_id'])]
                if verdict['prediction_sha256'] != row['answer_sha256']:
                    raise ValueError('changed answer')
                verdicts[policy][row['question_id']] = verdict['judge_label']
            service = sum(row['service_seconds'] for row in run['rows'])
            startup = run['model_load_seconds'] + run['visual_state_identity_seconds'] + run['construction_seconds']
            totals[policy]['service'] += service
            totals[policy]['startup_covered'] += startup
            if policy == 'provenance_milp':
                if run.get('relevance_modality') != 'joint_index':
                    raise ValueError('joint-index candidate required')
                index = Path(registration['arguments']['joint_index'])
                metadata = json.loads((index / 'index.json').read_text())
                if digest(index / 'index.json') != run['joint_index_metadata']['index_json_sha256']:
                    raise ValueError('changed producer metadata')
                producer_total += metadata['construction_seconds']
                index_bytes += metadata['vector_bytes']
                model_paths.update({role: Path(registration['arguments'][role]) for role in
                                    ('reader', 'embedder', 'reranker', 'joint_embedding_model', 'joint_reranker_model')})
            runs.append({'policy': policy, 'user': person, 'service_seconds': service,
                         'startup_covered_seconds': startup,
                         'model_loaded_cuda_allocated_bytes': run['model_loaded_cuda_allocated_bytes'],
                         'peak_request_cuda_allocated_bytes': max(row['service_peak_cuda_allocated_bytes']
                                                                  for row in run['rows']),
                         'truncated_outputs': sum(row['stop_reason'] == 'max_new_tokens' for row in run['rows'])})
    weights = {role: sum(path.stat().st_size for path in directory.glob('*.safetensors'))
               for role, directory in model_paths.items()}
    return {'complete': True, 'judge_sha256': digest(judge_path),
            'paired_quality': paired_uncertainty(verdicts['local_hyperedge'], verdicts['provenance_milp']),
            'amortization': amortization(totals['local_hyperedge']['service'], totals['provenance_milp']['service'],
                totals['local_hyperedge']['startup_covered'], totals['provenance_milp']['startup_covered'],
                producer_total, len(verdicts['local_hyperedge'])),
            'candidate_checkpoint_file_bytes_by_role': weights,
            'candidate_index_vector_bytes_three_users': index_bytes,
            'capacity_boundary': 'checkpoint serialized bytes and observed server CUDA peaks, not full phone/Jetson footprint; excludes runtime/OS/other applications; no device feasibility qualification',
            'per_run': runs, 'actual_pim_end_to_end_speedup': None,
            'main_experiment_closed': False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--judge', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('fresh audit output required')
    result = audit(args.run_root, args.judge)
    with args.output.open('x') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
