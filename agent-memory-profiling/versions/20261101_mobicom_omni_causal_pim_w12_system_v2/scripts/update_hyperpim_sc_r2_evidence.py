#!/usr/bin/env python3
"""Bind the 192-request r2 quality gate and cycle timeline into one evidence file."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--timeline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    evidence = json.loads(args.evidence.read_text())
    audit = json.loads(args.audit.read_text())
    evaluation = json.loads(args.evaluation.read_text())
    timeline = json.loads(args.timeline.read_text())

    if not audit.get('complete') or audit.get('requests') != 192:
        raise ValueError('complete 192-request paired audit required')
    if not evaluation.get('complete') or evaluation.get('requests') != 192:
        raise ValueError('complete 192-request evaluation required')
    if evaluation['serving_pair'] != {
            'reference': 'bf16', 'candidate': 'w8a16_r2',
            'candidate_quantization': 'w8a16',
            'candidate_fallback_scope': 'none',
            'candidate_residual_bits': 2,
            'single_factor': True, 'fallback_scope_verified': True,
            'residual_bits_verified': True, 'upstream_identical': True}:
        raise ValueError('evaluation serving contract differs from registered r2 arm')
    if timeline.get('requests') != 200 or timeline.get('calibration', {}).get('passed') is not True:
        raise ValueError('calibrated 200-request cycle timeline required')
    if evaluation['gate']['all_requests_retained'] is not True:
        raise ValueError('all-request denominator was not retained')

    quality = evaluation['quality']
    secondary = evaluation['secondary']
    evidence.setdefault('quality_pairs', {})['r2_192_confirmation'] = {
        'requests': evaluation['requests'],
        'reference_correct': quality['reference_correct'],
        'candidate_correct': quality['candidate_correct'],
        'paired_delta_pp': quality['paired_delta_pp'],
        'ci95_pp': quality['paired_delta_bootstrap_ci95_pp'],
        'noninferiority_passed': quality['noninferiority_passed'],
        'gate_passed': evaluation['gate']['passed'],
        'exact_tokens': secondary['exact_greedy_token_matches'],
        'stop_reason_matches': audit['generation']['stop_reason_matches'],
        'eos_reference_rate': secondary['eos']['reference_rate'],
        'eos_candidate_rate': secondary['eos']['candidate_rate'],
        'eos_delta_pp': secondary['eos']['delta_pp'],
    }
    evidence['r2_192_confirmation'] = {
        'status': 'complete',
        'run_root': 'artifacts/hyperpim_sc_r2_quality_gate_20261029/serving',
        'registered_requests': 192,
        'preregistration': 'artifacts/hyperpim_sc_r2_quality_gate_20261029/preregistration.json',
        'paired_audit': str(args.audit),
        'evaluation': str(args.evaluation),
        'paired_audit_sha256': digest(args.audit),
        'evaluation_sha256': digest(args.evaluation),
        'quality': quality,
        'per_user_quality': evaluation['per_user_quality'],
        'gate': evaluation['gate'],
        'secondary': secondary,
        'exact_generation_transparency': evaluation['exact_generation_transparency'],
        'quality_boundary': evaluation['boundary'],
        'publication_ready': False,
    }
    projection = evidence.setdefault('projection', {})
    projection['cycle_timeline'] = {
        'artifact': str(args.timeline),
        'sha256': digest(args.timeline),
        'decode_sequence': timeline['decode_sequence'],
        'calibration': timeline['calibration'],
        'selected_arm': timeline['selected_arm'],
        'token_timelines': {
            name: timeline['token_timelines'][name]['totals']
            for name in ('r0', 'r2', 'r4')
        },
        'scenarios': timeline['scenarios'],
        'cycle_sensitivity': timeline['cycle_sensitivity'],
        'stage_token_accounting': timeline['stage_token_accounting'],
        'cold_start': timeline['cold_start'],
        'boundary': timeline['boundary'],
    }
    evidence['as_of'] = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')
    evidence['quality_gate_status'] = (
        'passed' if evaluation['gate']['passed'] else 'failed')
    evidence['boundary'] = (
        'development evidence only; no physical PIM or phone measurement; quality is a '
        'same-family local 8B Judge on three development users; cycle timing is a target '
        'LPDDR model calibrated to GeniePIM, not RTL or silicon')
    args.output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({
        'quality_gate_status': evidence['quality_gate_status'],
        'paired_delta_pp': quality['paired_delta_pp'],
        'ci95_pp': quality['paired_delta_bootstrap_ci95_pp'],
        'gate': evaluation['gate'],
        'cycle_pim_linear_ms_per_token': timeline['selected_arm']['pim_linear_ms_per_token'],
        'cycle_speedup_vs_host_lower_bound': {
            key: timeline['scenarios']['cycle_target_speedup'][key]
            for key in ('service_vs_host_control_lower_bound',
                        'ttft_vs_host_control_lower_bound',
                        'service_p95_vs_host_control_lower_bound',
                        'ttft_p95_vs_host_control_lower_bound')},
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
