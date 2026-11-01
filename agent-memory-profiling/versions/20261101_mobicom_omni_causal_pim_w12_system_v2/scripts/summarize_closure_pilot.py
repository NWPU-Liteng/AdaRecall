"""Paired full-denominator pilot accounting; never certifies isolated E2E gains."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(run_root, judge_path):
    judge = json.loads(judge_path.read_text())
    if not judge.get('complete'):
        raise ValueError('complete judge output required')
    policies = {}
    cohort_hashes = set()
    candidate_modalities = set()
    judged = {(row['run'], row['question_id']): row for row in judge['rows']}
    for policy in ('local_hyperedge', 'provenance_milp'):
        paths = [run_root / f'{policy}_user_{user}.json' for user in ('0', '1', '2')]
        rows = []
        for path in paths:
            run = json.loads(path.read_text())
            if not run.get('complete') or digest(path) != judge['provenance']['run_sha256'].get(str(path)):
                raise ValueError('incomplete or changed judged serving output')
            registration = json.loads(path.with_suffix('.registration.json').read_text())
            cohort_hashes.add(registration.get('cohort_sha256'))
            if policy == 'provenance_milp':
                candidate_modalities.add(run.get('relevance_modality', 'text'))
            if [row['question_id'] for row in run['rows']] != registration['question_ids']:
                raise ValueError('frozen request denominator changed')
            for row in run['rows']:
                evaluation = judged[(str(path), row['question_id'])]
                if evaluation['prediction_sha256'] != row['answer_sha256']:
                    raise ValueError('judge/prediction mismatch')
                rows.append({**row, 'judge_label': evaluation['judge_label']})
        identities = [(row['person_id'], row['question_id']) for row in rows]
        if len(identities) != len(set(identities)):
            raise ValueError('duplicate request within policy')
        latency = [row['service_seconds'] for row in rows]
        correct = sum(row['judge_label'] == 'CORRECT' for row in rows)
        valid = sum(row['judge_label'] is not None for row in rows)
        ledger = Counter()
        for row in rows:
            ledger.update(row['service_stage_ledger']['disjoint_seconds'])
        total = sum(latency)
        if not np.isclose(sum(ledger.values()), total, rtol=1e-8, atol=1e-7):
            raise ValueError('service ledger does not reconcile')
        policies[policy] = {'requests': len(rows), 'correct': correct, 'invalid_judge': len(rows) - valid,
            'full_denominator_accuracy': correct / len(rows),
            'valid_judge_accuracy': correct / valid if valid else None,
            'eos': sum(row['stop_reason'] == 'eos' for row in rows),
            'max_new_tokens': sum(row['stop_reason'] == 'max_new_tokens' for row in rows),
            'total_service_seconds': total, 'mean_seconds': float(np.mean(latency)),
            'p50_seconds': float(np.percentile(latency, 50)), 'p95_seconds': float(np.percentile(latency, 95)),
            'disjoint_stage_seconds': dict(ledger),
            'vision_nested_seconds': sum(row['service_stage_ledger']['nested_seconds']['vision_in_prefill_seconds'] for row in rows),
            'image_occurrences': sum(len(row['images']) for row in rows),
            'requests_with_images': sum(bool(row['images']) for row in rows),
            'input_tokens': sum(row['input_tokens'] for row in rows),
            'generation_tokens': sum(len(row['answer_token_ids']) for row in rows),
            'solver_status_counts': dict(Counter(str(row['hypergraph_evidence_plan'].get('solver_status')) for row in rows)),
            'max_solver_gap': max(row['hypergraph_evidence_plan'].get('mip_gap', 0) for row in rows),
            'rows': {row['question_id']: {'person_id': row['person_id'], 'judge_label': row['judge_label'],
                     'stop_reason': row['stop_reason'], 'service_seconds': row['service_seconds'],
                     'images': len(row['images'])} for row in rows}}
    reference = policies['local_hyperedge']['rows']
    candidate = policies['provenance_milp']['rows']
    if set(reference) != set(candidate):
        raise ValueError('unpaired policy request identities')
    if len(cohort_hashes) != 1 or len(candidate_modalities) != 1:
        raise ValueError('inconsistent cohort or candidate relevance protocol')
    wins = [identity for identity in reference if reference[identity]['judge_label'] != 'CORRECT'
            and candidate[identity]['judge_label'] == 'CORRECT']
    losses = [identity for identity in reference if reference[identity]['judge_label'] == 'CORRECT'
              and candidate[identity]['judge_label'] != 'CORRECT']
    no_invalid = all(policy['invalid_judge'] == 0 for policy in policies.values())
    return {'complete': True, 'policies': policies,
        'cohort_sha256': next(iter(cohort_hashes)),
        'candidate_relevance_modality': next(iter(candidate_modalities)),
        'pairing': {'requests': len(reference), 'wins': wins, 'losses': losses,
            'screening_service_ratio': policies['local_hyperedge']['total_service_seconds'] /
                                       policies['provenance_milp']['total_service_seconds']},
        'quality_screen': {'no_proxy_score_loss': no_invalid and len(wins) >= len(losses),
                          'no_invalid_judge': no_invalid,
                          'population_noninferiority_established': False},
        'isolated_causal_e2e_speedup': None, 'pim_end_to_end_speedup': None,
        'main_experiment_closed': False,
        'boundary': f'{len(reference)} development requests per policy; separate GPUs, simultaneous shared CPU/storage; frozen candidate relevance and optional images; proxy judge only; no isolated repeat, population noninferiority, visual grounding, state-reuse or PIM benefit established',
        'judge_sha256': digest(judge_path), 'script_sha256': digest(Path(__file__))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--judge', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('fresh summary output required')
    result = summarize(args.run_root, args.judge)
    with args.output.open('x') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps({name: {key: value for key, value in stats.items() if key != 'rows'}
                      for name, stats in result['policies'].items()}, ensure_ascii=False, indent=2))
    print(json.dumps(result['pairing'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
