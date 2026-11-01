"""Audit label-free phase diagnostics without qualifying quality or speedup."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(root):
    root = Path(root)
    protocol = json.loads((root / 'protocol.json').read_text())
    cohort = json.loads((root / 'cohort.json').read_text())
    if not protocol.get('diagnostic_only') or protocol.get('judge_invoked') is not False:
        raise ValueError('phase-only protocol required')
    if protocol.get('joint_patch_projection', 'conv') not in {'conv', 'linear'}:
        raise ValueError('unknown patch projection')
    if digest(root / 'cohort.json') != protocol['cohort_sha256']:
        raise ValueError('diagnostic cohort changed')
    if protocol.get('evaluation_label_files_read') or cohort.get('evaluation_label_files_read'):
        raise ValueError('diagnostic must not read labels')
    for name, expected in protocol['sources_sha256'].items():
        if digest(root / 'runtime' / name) != expected:
            raise ValueError('runtime snapshot changed')
    rows = []
    for user in protocol['gpu_by_user']:
        source = root / f'provenance_milp_user_{user}.json'
        data = json.loads(source.read_text())
        if not data.get('complete') or data.get('evaluation_label_files_read'):
            raise ValueError('incomplete or label-reading execution')
        if [row['question_id'] for row in data['rows']] != cohort['question_ids_by_user'][user]:
            raise ValueError('diagnostic request identity mismatch')
        for row in data['rows']:
            plan = row['hypergraph_evidence_plan']
            timing = plan['multimodal_scoring_timing']
            if timing.get('patch_projection', 'conv') != protocol.get('joint_patch_projection', 'conv'):
                raise ValueError('execution patch projection differs from registered protocol')
            phases = {name: timing[name] for name in (
                'image_load_seconds', 'prompt_build_seconds', 'processor_transfer_seconds',
                'model_forward_seconds', 'residual_seconds')}
            if any(not math.isfinite(value) or value < 0 for value in phases.values()):
                raise ValueError('invalid phase timing')
            if not math.isclose(sum(phases.values()), timing['total_seconds'], abs_tol=1e-6):
                raise ValueError('phase costs do not close')
            vision = timing.get('vision_in_forward_seconds')
            if vision is not None and (vision < 0 or vision > timing['model_forward_seconds']):
                raise ValueError('nested vision exceeds forward time')
            rows.append({'user': user, 'question_id': row['question_id'],
                'source_sha256': digest(source), 'service_seconds_diagnostic_only': row['service_seconds'],
                'query_retrieval_seconds': row['query_retrieval_seconds'],
                'evidence_planning_seconds': row['evidence_planning_seconds'],
                'image_cost_seconds': plan['image_cost_seconds'],
                'reranker': timing,
                'patch_projection': timing.get('patch_projection', 'conv'),
                'forward_fraction_of_scoring': timing['model_forward_seconds'] / timing['total_seconds'],
                'forward_fraction_of_service': timing['model_forward_seconds'] / row['service_seconds']})
    return {'complete': True, 'diagnostic_only': True, 'rows': rows,
            'actual_pim_end_to_end_speedup': None, 'main_experiment_closed': False,
            'boundary': 'synchronized phase composition, eight-token decode, no quality or causal E2E claim'}


def compare_rows(reference, candidate):
    if (reference['question_id'], reference['person_id']) != (candidate['question_id'], candidate['person_id']):
        raise ValueError('different requests cannot be compared')
    old_plan, new_plan = (row['hypergraph_evidence_plan'] for row in (reference, candidate))
    old_scores, new_scores = ({item['edge_id']: item['relevance_log_odds']
                              for item in plan['candidate_replay']} for plan in (old_plan, new_plan))
    if old_scores.keys() != new_scores.keys():
        raise ValueError('different candidate sets')
    old_rank = sorted(old_scores, key=lambda key: (-old_scores[key], key))
    new_rank = sorted(new_scores, key=lambda key: (-new_scores[key], key))
    comparison = {
        'question_id': reference['question_id'],
        'scores_bitwise_equal': old_scores == new_scores,
        'rank_order_equal': old_rank == new_rank,
        'top_16_equal': old_rank[:16] == new_rank[:16],
        'max_score_absolute_difference': max(abs(old_scores[key] - new_scores[key]) for key in old_scores),
        'accepted_atoms_equal': old_plan['accepted_atom_ids'] == new_plan['accepted_atom_ids'],
        'selected_images_equal': old_plan['images'] == new_plan['images'],
        'processor_inputs_equal': reference['processor_inputs_sha256'] == candidate['processor_inputs_sha256'],
        'generated_tokens_equal': reference['answer_token_ids'] == candidate['answer_token_ids'],
        'stop_reason_equal': reference['stop_reason'] == candidate['stop_reason'],
        'performance_qualified': False,
        'boundary': 'single-request numerical diagnostic, not full-budget paired latency validation'}
    old_timing = old_plan.get('multimodal_scoring_timing', {})
    new_timing = new_plan.get('multimodal_scoring_timing', {})
    if old_timing and new_timing:
        comparison['reranker_total_seconds_reference_batch'] = old_timing.get('total_seconds')
        comparison['reranker_total_seconds_candidate_batch'] = new_timing.get('total_seconds')
        comparison['reranker_forward_seconds_reference_batch'] = old_timing.get('model_forward_seconds')
        comparison['reranker_forward_seconds_candidate_batch'] = new_timing.get('model_forward_seconds')
        old_vision, new_vision = old_timing.get('vision_in_forward_seconds'), new_timing.get('vision_in_forward_seconds')
        if old_vision is not None and new_vision is not None:
            comparison['vision_forward_seconds_reference_batch'] = old_vision
            comparison['vision_forward_seconds_candidate_batch'] = new_vision
    return comparison


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--reference-root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.run_root)
    if args.reference_root:
        audit(args.reference_root)
        result['reference_comparisons'] = []
        for row in result['rows']:
            name = f"provenance_milp_user_{row['user']}.json"
            reference = json.loads((args.reference_root / name).read_text())['rows']
            candidate = json.loads((args.run_root / name).read_text())['rows']
            result['reference_comparisons'].extend(compare_rows(old, new)
                                                   for old, new in zip(reference, candidate))
    if args.output.exists():
        raise ValueError('fresh audit output required')
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({'complete': True, 'requests': len(result['rows']), 'output': str(args.output)}))


if __name__ == '__main__':
    main()
