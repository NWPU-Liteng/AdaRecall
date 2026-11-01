"""Archive the frozen comparison and model PIM only on its executed BF16 trace."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from audit_omni_smoke_closure import digest
from pim_backend.decode_gemv import project_trace
from pim_backend.lpddr_bank_pim import BankPimConfig
from run_evidence_compiler_mainline import summarize, write


def project_candidate(runs, config):
    if (not runs or any(not run.get('complete') or run.get('reader_quantization') != 'bf16'
                        or run.get('evidence_policy_name') != 'evidence_compiler' for run in runs)
            or len({run['reader_config_sha256'] for run in runs}) != 1):
        raise ValueError('complete same-Reader BF16 candidate runs required')
    operators = [operator for run in runs for operator in run['operator_trace']['operators']]
    if any(operator['element_bytes'] != 2 for operator in operators
           if operator['phase'] == 'decode' and operator['rows'] == 1):
        raise ValueError('BF16 trace cannot be projected as an integer arithmetic path')
    if not any(operator['phase'] == 'decode' and operator['rows'] == 1 for operator in operators):
        raise ValueError('executed batch-one decode trace required')
    result = project_trace(operators, config)
    result.update({
        'questions': sum(len(run['rows']) for run in runs),
        'decode_tokens_observed': sum(len(row['answer_token_ids']) for run in runs for row in run['rows']),
        'reader_config_sha256': runs[0]['reader_config_sha256'],
        'reader_weight_bytes': runs[0]['reader_weight_bytes'],
        'placement_charge': 'once per resident Reader replica over the concatenated candidate cohort',
        'host_linear_seconds_scenario': result['host_linear_cycles'] / config.clock_hz,
        'warm_hybrid_linear_seconds_scenario': result['warm_hybrid_linear_cycles'] / config.clock_hz,
        'cold_hybrid_linear_seconds_scenario': result['cold_hybrid_linear_cycles'] / config.clock_hz,
        'full_model_residency_validated': False,
        'measured_gpu_seconds_replaced': False,
        'operator_trace_coverage_by_run': [run['operator_trace']['coverage'] for run in runs],
        'arithmetic': 'BF16 operands with custom bank-local FP32 accumulation assumption; no W8A8 conversion',
    })
    return result


def evidence_statistics(runs):
    rows = [row for run in runs for row in run['rows']]
    plans = [row['hypergraph_evidence_plan'] for row in rows]
    gaps = [float(plan['mip_gap']) for plan in plans if plan.get('mip_gap') is not None]
    return {
        'requests': len(rows),
        'requests_with_images': sum(bool(row['images']) for row in rows),
        'image_inputs': sum(len(row['images']) for row in rows),
        'input_tokens': sum(row['input_tokens'] for row in rows),
        'solver_status_counts': dict(Counter(str(plan['solver_status']) for plan in plans
                                             if 'solver_status' in plan)),
        'max_reported_mip_gap': max(gaps) if gaps else None,
        'render_feasibility_cut_count': sum(len(plan.get('render_feasibility_cuts', [])) for plan in plans),
        'boundary': 'offline diagnostics only; no labels or answer outcomes feed serving selection',
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--profile', type=Path, default=ROOT / 'pim_backend/mobile_lpddr5_bank_profile.json')
    args = parser.parse_args()
    root = args.run_root.resolve()
    if (root / 'closure.json').exists() or (root / 'closure.md').exists():
        raise ValueError('do not overwrite a completed comparison archive')
    cohort = json.loads((root / 'cohort.json').read_text())
    judge = json.loads((root / 'judge.json').read_text())
    if not judge.get('complete'):
        raise ValueError('whole-cohort Judge required')
    prepared = ROOT / 'artifacts/omni_paper_scope_v3_20261021/prepared'
    summary = summarize(root, prepared)
    runs = {policy: [json.loads((root / f'{policy}_user_{user}.json').read_text())
                     for user in cohort['question_ids_by_user']] for policy in ('reference', 'candidate')}
    profile = json.loads(args.profile.read_text())
    projection = project_candidate(runs['candidate'], BankPimConfig(**profile['config']))
    projection.update({'profile_name': profile['name'], 'profile_sha256': digest(args.profile),
                       'profile_calibration_status': profile['calibration_status'],
                       'inputs_sha256': {str(root / f'candidate_user_{user}.json'):
                                         digest(root / f'candidate_user_{user}.json')
                                         for user in cohort['question_ids_by_user']}})
    indices = []
    protocol = json.loads((root / 'protocol.json').read_text())
    for job in protocol['jobs']:
        if job['policy'] != 'candidate':
            continue
        index_path = Path(job['command'][job['command'].index('--joint-index') + 1])
        metadata = json.loads((index_path / 'index.json').read_text())
        indices.append({'person_id': metadata['person_id'],
            'index_json_sha256': digest(index_path / 'index.json'),
            'exchanges': len(metadata['edges']), 'vector_bytes': metadata['vector_bytes'],
            'index_artifact_bytes': sum((index_path / name).stat().st_size for name in ('index.json', 'vectors.npy')),
            'evidence_view_count': metadata.get('evidence_view_count', 0),
            'evidence_view_vector_bytes': metadata.get('evidence_view_vector_bytes', 0),
            'evidence_view_artifact_bytes': ((index_path / 'evidence_view_vectors.npy').stat().st_size
                                             if metadata.get('evidence_views') else 0),
            'construction_seconds': metadata['construction_seconds'],
            'incremental_construction_seconds': metadata['incremental_construction_seconds'],
            'inherited_visual_construction_seconds': metadata['inherited_visual_construction_seconds'],
            'image_cost_preparation_seconds': metadata['image_cost_preparation_seconds'],
            'reader_image_cost_count': len(metadata['reader_image_token_costs'])})
    delta = summary['quality']['paired_accuracy_delta']
    result = {'development_comparison_complete': True, 'summary': summary,
        'evidence_statistics': {policy: evidence_statistics(values) for policy, values in runs.items()},
        'offline_indices': indices, 'same_candidate_trace_pim': projection,
        'decision': ('candidate_quality_regressed_on_frozen_development' if delta < 0 else
                     'candidate_quality_tied_on_frozen_development' if delta == 0 else
                     'candidate_quality_improved_on_frozen_development_not_yet_independently_validated'),
        'main_experiment_closed': False,
        'source_files_sha256': {name: digest(ROOT / name) for name in (
            'scripts/close_evidence_compiler_mainline.py', 'scripts/run_evidence_compiler_mainline.py',
            'pim_backend/decode_gemv.py', 'pim_backend/lpddr_bank_pim.py')},
        'boundary': 'full development accounting, not a guaranteed positive result; no hardware speedup, mobile latency, or publication acceptance claim'}
    write(root / 'closure.json', result)
    quality = summary['quality']
    report = [
        '# Frozen development comparison', '',
        f"- Decision: `{result['decision']}`.",
        f"- Full paired denominator: {quality['requests']} requests per policy; unchanged Qwen3-VL-8B Judge.",
        f"- Correct: reference {quality['reference_correct']}; candidate {quality['expanded_correct']}.",
        f"- Paired accuracy delta: {delta:+.4%}; conditional bootstrap CI95 {quality['paired_delta_bootstrap_ci95']}.",
        f"- Shared-host service ratio (reference/candidate): {summary['screening_service_ratio']:.4f}; not causal E2E speedup.",
        f"- Images: reference {result['evidence_statistics']['reference']['image_inputs']}; candidate {result['evidence_statistics']['candidate']['image_inputs']}.",
        f"- Same BF16 candidate trace, modeled decode-Linear warm/cold ratios: {projection['warm_linear_speedup_scenario']:.4f} / {projection['cold_linear_speedup_scenario']:.4f}.",
        '- PIM uses the fixed uncalibrated LPDDR5 profile, one replica placement charge, and shared-capacity accounting.',
        '- No PIM E2E extrapolation: attention, recurrent state, vision/prefill, total device residency and thermals remain outside this model.',
        '- No algorithm/threshold tuning after this frozen result; validation and holdout remain sealed.',
        '- `development_comparison_complete=true`; `main_experiment_closed=false`.',
    ]
    with (root / 'closure.md').open('x') as handle:
        handle.write('\n'.join(report) + '\n')
    print(json.dumps({'decision': result['decision'], 'quality': quality,
                      'closure': str(root / 'closure.json')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
