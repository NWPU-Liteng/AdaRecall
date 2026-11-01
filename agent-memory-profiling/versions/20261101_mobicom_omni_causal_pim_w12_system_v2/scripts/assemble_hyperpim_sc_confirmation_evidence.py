#!/usr/bin/env python3
"""Assemble development and sealed-holdout evidence without rewriting raw results."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            result.update(block)
    return result.hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--development-evidence', type=Path, required=True)
    parser.add_argument('--holdout-evaluation', type=Path, required=True)
    parser.add_argument('--holdout-audit', type=Path, required=True)
    parser.add_argument('--holdout-preregistration', type=Path, required=True)
    parser.add_argument('--holdout-root', type=Path, required=True)
    parser.add_argument('--aggregate-output', type=Path, required=True)
    parser.add_argument('--manifest-output', type=Path, required=True)
    parser.add_argument('--sha256-output', type=Path, required=True)
    args = parser.parse_args()

    development = load(args.development_evidence)
    holdout = load(args.holdout_evaluation)
    audit = load(args.holdout_audit)
    preregistration = load(args.holdout_preregistration)

    if development.get('quality_gate_status') != 'passed':
        raise ValueError('the historical development evidence must remain a passed 192-request run')
    if development.get('r2_192_confirmation', {}).get('registered_requests') != 192:
        raise ValueError('development evidence is not the registered 192-request confirmation')
    development_quality = development.get('quality_pairs', {}).get('r2_192_confirmation', {})
    if not development_quality.get('noninferiority_passed'):
        raise ValueError('development quality pair did not pass')

    quality = deepcopy(holdout.get('quality', {}))
    per_user_quality = deepcopy(holdout.get('per_user_quality', {}))
    stale_scope = ('paired requests resampled within each observed user; conditional on '
                   'these three development users, not a cross-user population guarantee')
    corrected_scope = ('paired requests resampled within each observed user; conditional on '
                       'the observed users, not a cross-user population guarantee')
    for summary in [quality, *per_user_quality.values()]:
        scope = summary.get('bootstrap', {}).get('scope')
        if scope == stale_scope:
            summary['bootstrap']['scope'] = corrected_scope
    gate = holdout.get('gate', {})
    pair = holdout.get('serving_pair', {})
    if holdout.get('complete') is not True or holdout.get('requests') != 200:
        raise ValueError('complete 200-request holdout evaluation required')
    if quality.get('noninferiority_passed') is not False or gate.get('passed') is not False:
        raise ValueError('holdout must retain its failed preregistered gate')
    if pair.get('single_factor') is not True or pair.get('upstream_identical') is not True:
        raise ValueError('holdout serving pair is not a clean single-factor comparison')
    if pair.get('candidate_residual_bits') != 2 or pair.get('candidate_fallback_scope') != 'none':
        raise ValueError('holdout candidate is not the frozen r2/none arm')
    if audit.get('complete') is not True or audit.get('requests') != 200:
        raise ValueError('complete 200-request paired audit required')

    now = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')
    root = args.holdout_root
    referenced = {
        'cohort.json': root / 'cohort.json',
        'protocol.json': root / 'protocol.json',
        'preregistration.json': root / 'preregistration.json',
        'serving/serving_manifest.json': root / 'serving' / 'serving_manifest.json',
        'serving/bf16/cohort.json': root / 'serving' / 'bf16' / 'cohort.json',
        'serving/w8a16_r2/cohort.json': root / 'serving' / 'w8a16_r2' / 'cohort.json',
        'paired_quality_r2.json': root / 'paired_quality_r2.json',
        'evaluation/summary.json': root / 'evaluation' / 'summary.json',
        'evaluation/bf16.requests.json': root / 'evaluation' / 'bf16.requests.json',
        'evaluation/bf16.judge.json': root / 'evaluation' / 'bf16.judge.json',
        'evaluation/w8a16.requests.json': root / 'evaluation' / 'w8a16.requests.json',
        'evaluation/w8a16.judge.json': root / 'evaluation' / 'w8a16.judge.json',
        'evaluation_attempt_1_rejected.json': root / 'evaluation_attempt_1_rejected.json',
        'failed_attempt_1_pre_generation/serving_manifest.json': (
            root / 'failed_attempt_1_pre_generation' / 'serving_manifest.json'),
        'amendments/amendment_2.json': root / 'amendments' / 'amendment_2.json',
        'metadata_correction.json': root / 'metadata_correction.json',
    }
    missing = [str(path) for path in referenced.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError('missing evidence files: ' + ', '.join(missing))

    hashes = {str(path): digest(path) for path in referenced.values()}
    aggregate = deepcopy(development)
    aggregate['schema'] = 'hyperpim-sc-confirmation-evidence-v2'
    aggregate['as_of'] = now
    aggregate['supersedes'] = {
        'path': str(args.development_evidence),
        'sha256': digest(args.development_evidence),
        'scope': 'historical 192-request development evidence; original file is left unchanged',
    }
    aggregate['quality_pairs']['r2_holdout200_confirmation'] = {
        'requests': holdout['requests'],
        'reference_correct': quality['reference_correct'],
        'candidate_correct': quality['candidate_correct'],
        'paired_delta_pp': quality['paired_delta_pp'],
        'ci95_pp': quality['paired_delta_bootstrap_ci95_pp'],
        'noninferiority_margin_pp': quality['noninferiority_margin_pp'],
        'noninferiority_passed': quality['noninferiority_passed'],
        'gate_passed': gate['passed'],
        'exact_tokens': holdout['secondary']['exact_greedy_token_matches'],
        'eos_delta_pp': holdout['secondary']['eos']['delta_pp'],
    }
    aggregate['r2_holdout200_confirmation'] = {
        'status': 'complete_gate_failed',
        'registered_requests': 200,
        'users': preregistration['cohort']['users'],
        'requests_per_user': preregistration['cohort']['per_user'],
        'cohort_sha256': preregistration['cohort']['sha256'],
        'preregistration': str(args.holdout_preregistration),
        'preregistration_sha256': digest(args.holdout_preregistration),
        'paired_audit': str(args.holdout_audit),
        'paired_audit_sha256': digest(args.holdout_audit),
        'evaluation': str(args.holdout_evaluation),
        'evaluation_sha256': digest(args.holdout_evaluation),
        'quality': quality,
        'per_user_quality': per_user_quality,
        'gate': gate,
        'secondary': holdout['secondary'],
        'failed_attempts': {
            'serving_pre_generation': 'failed_attempt_1_pre_generation/',
            'evaluation_pre_score': 'evaluation_attempt_1_rejected.json',
        },
        'post_failure_policy': (
            'do not tune residual bits, group size, margin, seed or resampling on users '
            '6,7,10,11; reserve untouched users 12-17 for a future frozen confirmation'),
        'quality_boundary': holdout['boundary'],
    }
    aggregate['overall_quality_gate_status'] = 'development_passed_holdout_failed'
    aggregate['r2_overall_quality_qualified'] = False
    aggregate['performance_claim_status'] = (
        'mechanism_projection_only_pending_quality_recovery')
    aggregate['quality_gate_status'] = 'development_passed_holdout_failed'
    aggregate['boundary'] = (
        'development evidence passed on three users; sealed four-user holdout failed the '
        'preregistered noninferiority gate; no physical PIM or phone measurement; cycle '
        'timing is a target LPDDR model calibrated to GeniePIM, not RTL or silicon')
    args.aggregate_output.parent.mkdir(parents=True, exist_ok=True)
    args.aggregate_output.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2) + '\n')

    manifest = {
        'schema': 'hyperpim-sc-holdout200-manifest-v1',
        'as_of': now,
        'cohort': preregistration['cohort'],
        'arms': preregistration['arms'],
        'serving': {
            'complete': True,
            'requests': 200,
            'duplicates': 0,
            'evaluation_label_files_read_during_serving': [],
        },
        'audit': {
            'complete': True,
            'requests': audit['requests'],
            'single_factor_reader_quantization': pair['single_factor'],
            'upstream_identical': pair['upstream_identical'],
            'candidate_quantization': pair['candidate_quantization'],
            'candidate_fallback_scope': pair['candidate_fallback_scope'],
            'candidate_residual_bits': pair['candidate_residual_bits'],
        },
        'evaluation': {
            'complete': True,
            'requests': holdout['requests'],
            'quality': quality,
            'gate': gate,
            'secondary': holdout['secondary'],
            'noninferiority_passed': False,
            'point_estimate_interpretation': (
                'failed the preregistered margin; the interval includes zero, so this is '
                'not proof of harm'),
        },
        'derived_evidence': {
            'aggregate': str(args.aggregate_output),
            'aggregate_sha256': digest(args.aggregate_output),
        },
        'files': {
            str(path.relative_to(root)): {
                'bytes': path.stat().st_size,
                'sha256': hashes[str(path)],
            }
            for path in referenced.values()
        },
        'boundary': (
            'sealed four-user quality holdout; same-family local 8B Judge; not population '
            'noninferiority; no physical PIM or phone measurement'),
    }
    args.manifest_output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    lines = [f'{manifest["files"][name]["sha256"]}  {name}' for name in sorted(manifest['files'])]
    args.sha256_output.write_text('\n'.join(lines) + '\n')
    print(json.dumps({
        'overall_quality_gate_status': aggregate['overall_quality_gate_status'],
        'development_delta_pp': development_quality['paired_delta_pp'],
        'holdout_delta_pp': quality['paired_delta_pp'],
        'holdout_ci95_pp': quality['paired_delta_bootstrap_ci95_pp'],
        'holdout_gate_passed': gate['passed'],
        'aggregate': str(args.aggregate_output),
        'manifest': str(args.manifest_output),
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
