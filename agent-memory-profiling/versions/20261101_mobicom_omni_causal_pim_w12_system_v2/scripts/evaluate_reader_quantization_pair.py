"""Full-denominator semantic gate for a frozen BF16/W8A16 serving pair."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from build_omni_smoke_judge_requests import build_requests
from score_mobilemem import load_pinned_protocol


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            result.update(block)
    return result.hexdigest()


def load_arm(root, cohort, prepared_audit_sha256, *, expected_quantization,
             expected_scope=None, expected_residual_bits=None,
             expected_weight_bits=None, expected_group_size=None):
    rows = []
    provenance = {}
    for user, expected_ids in sorted(cohort['question_ids_by_user'].items(), key=lambda item: int(item[0])):
        path = Path(root) / f'baseline_user_{user}.json'
        run = json.loads(path.read_text())
        if not run.get('complete') or run.get('evaluation_label_files_read') != []:
            raise ValueError('complete label-isolated serving arm required')
        if run.get('prepared_audit_sha256') != prepared_audit_sha256:
            raise ValueError('serving arm prepared audit differs')
        if run.get('reader_quantization') != expected_quantization:
            raise ValueError('serving arm quantization differs from registration')
        observed_scope = run.get('reader_w8a16_fallback_scope')
        if expected_quantization == 'bf16':
            if observed_scope not in (None, 'none'):
                raise ValueError('BF16 reference arm declares a W8A16 fallback scope')
        elif expected_scope is not None and observed_scope != expected_scope:
            raise ValueError('serving arm W8A16 fallback scope differs from registration')
        if (expected_residual_bits is not None
                and run.get('reader_w8a16_residual_bits', 0) != expected_residual_bits):
            raise ValueError('serving arm W8A16 residual bits differ from registration')
        if (expected_weight_bits is not None
                and run.get('reader_weight_bits') != expected_weight_bits):
            raise ValueError('serving arm weight bits differ from registration')
        if (expected_group_size is not None
                and run.get('reader_w12a16_group_size') != expected_group_size):
            raise ValueError('serving arm W12A16 group size differs from registration')
        if [row['question_id'] for row in run['rows']] != expected_ids:
            raise ValueError('serving arm order differs from registered cohort')
        rows.extend(run['rows'])
        provenance[str(path)] = digest(path)
    identities = [row['question_id'] for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError('duplicate serving identity')
    return rows, provenance


def paired_summary(rows, margin_pp, bootstrap_samples, seed):
    count = len(rows)
    counts = Counter()
    for row in rows:
        reference, candidate = row['reference_correct'], row['candidate_correct']
        counts['both_correct' if reference and candidate else 'reference_only' if reference
               else 'candidate_only' if candidate else 'both_wrong'] += 1
    discordant = counts['reference_only'] + counts['candidate_only']
    smaller = min(counts['reference_only'], counts['candidate_only'])
    mcnemar = (min(1.0, 2 * sum(math.comb(discordant, index) for index in range(smaller + 1))
                   / 2 ** discordant) if discordant else 1.0)
    generator = np.random.default_rng(seed)
    samples = np.zeros(bootstrap_samples)
    for person_id in sorted({row['person_id'] for row in rows}, key=int):
        differences = np.asarray([
            int(row['candidate_correct']) - int(row['reference_correct'])
            for row in rows if row['person_id'] == person_id])
        samples += generator.choice(
            differences, size=(bootstrap_samples, len(differences))).sum(axis=1)
    samples /= count
    lower, upper = np.quantile(samples, [0.025, 0.975]).tolist()
    delta = (counts['candidate_only'] - counts['reference_only']) / count
    return {
        'requests': count,
        'partition': dict(counts),
        'reference_correct': counts['both_correct'] + counts['reference_only'],
        'candidate_correct': counts['both_correct'] + counts['candidate_only'],
        'paired_accuracy_delta': delta,
        'paired_delta_pp': 100 * delta,
        'paired_delta_bootstrap_ci95_pp': [100 * lower, 100 * upper],
        'exact_mcnemar_two_sided_p': mcnemar,
        'noninferiority_margin_pp': -margin_pp,
        'noninferiority_passed': 100 * lower >= -margin_pp,
        'bootstrap': {
            'samples': bootstrap_samples, 'seed': seed,
            'scope': ('paired requests resampled within each observed user; conditional on the '
                      'observed users, not a cross-user population guarantee'),
        },
    }


def rate(rows, key):
    return sum(bool(row[key]) for row in rows) / len(rows)


def candidate_arm_label(quantization, scope, residual_bits):
    label = quantization
    if residual_bits:
        label += f'_r{residual_bits}'
    if scope not in (None, 'none'):
        label += f'_{scope}'
    return label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pair-audit', type=Path, required=True)
    parser.add_argument('--bf16-root', type=Path, required=True)
    parser.add_argument('--candidate-root', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--allowed-splits', nargs='+',
                        choices=('development', 'validation', 'holdout'),
                        default=['development'])
    parser.add_argument('--preregistration', type=Path,
                        help='required for holdout evaluation authorization')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--margin-pp', type=float, default=5.0)
    parser.add_argument('--bootstrap-samples', type=int, default=20000)
    parser.add_argument('--seed', type=int, default=20261028)
    parser.add_argument('--judge-model', type=Path, default=Path(
        '/data2/user25263093/.cache/modelscope/hub/models/Qwen/Qwen3-VL-8B-Instruct'))
    parser.add_argument('--judge-device', default='cuda:0')
    parser.add_argument('--candidate-quantization', choices=('w8a8', 'w8a16', 'w12a16'),
                        default='w8a16')
    parser.add_argument('--candidate-label', default='w8a16')
    parser.add_argument('--candidate-scope',
                        choices=('none', 'lm_head', 'output_sensitive'),
                        default='none')
    parser.add_argument('--candidate-residual-bits', type=int, choices=(0, 2, 4),
                        default=0)
    parser.add_argument('--candidate-weight-bits', type=int, choices=(8, 12), default=None)
    parser.add_argument('--candidate-w12a16-group-size', type=int,
                        choices=(8, 16, 32, 64, 128, 256), default=128)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError('fresh evaluation output directory required')
    if not math.isfinite(args.margin_pp) or args.margin_pp <= 0:
        raise ValueError('positive finite noninferiority margin required')
    pair = json.loads(args.pair_audit.read_text())
    if (not pair.get('complete') or pair.get('performance_comparable') is not False
            or pair.get('actual_physical_pim_executed') is not False):
        raise ValueError('complete label-free serving pair required')
    expected_candidate = candidate_arm_label(
        args.candidate_quantization, args.candidate_scope,
        args.candidate_residual_bits)
    if pair['arms'] != {'reference': 'bf16', 'candidate': expected_candidate}:
        raise ValueError('BF16/W8A16 serving pair differs from registration')
    if pair['integrity']['semantic_pre_reader_execution_identical'] is not True:
        raise ValueError('pre-Reader execution must be identical before semantic evaluation')

    cohort_path = args.bf16_root / 'cohort.json'
    cohort = json.loads(cohort_path.read_text())
    if (cohort.get('evaluation_label_files_read') != []
            or cohort.get('runtime_labels_read', False) is not False):
        raise ValueError('cohort must remain serving-label isolated')
    allowed_splits = set(args.allowed_splits)
    if set(cohort.get('serving_splits') or ['development']) != allowed_splits:
        raise ValueError('evaluation splits differ from registered cohort')
    evaluation_authorization = {'allowed_splits': sorted(allowed_splits)}
    if 'holdout' in allowed_splits:
        if args.preregistration is None:
            raise ValueError('holdout evaluation requires preregistration')
        registration = json.loads(args.preregistration.read_text())
        if registration.get('status') != 'registered_before_generation':
            raise ValueError('holdout preregistration is not frozen')
        if registration['cohort']['sha256'] != digest(cohort_path):
            raise ValueError('holdout preregistration cohort differs')
        evidence = Path(registration['quality_dependency']['evidence'])
        if digest(evidence) != registration['quality_dependency']['evidence_sha256']:
            raise ValueError('holdout quality dependency changed')
        if registration['quality_dependency']['quality_gate_status'] != 'passed':
            raise ValueError('holdout quality dependency did not pass')
        evaluation_authorization.update({
            'preregistration': str(args.preregistration),
            'preregistration_sha256': digest(args.preregistration),
            'registered_at': registration.get('registered_at'),
        })
    prepared_audit_sha256 = cohort['prepared_audit_sha256']
    audit_path = args.prepared / 'audit.json'
    if digest(audit_path) != prepared_audit_sha256:
        raise ValueError('prepared audit changed')
    source_paths = [args.prepared / name for name in ('labels.jsonl', 'manifests.jsonl')]
    audit = json.loads(audit_path.read_text())
    for path in source_paths:
        if digest(path) != audit['artifacts_sha256'][path.name]:
            raise ValueError('prepared evaluation input changed')
    labels = {row['question_id']: row for row in (
        json.loads(line) for line in (args.prepared / 'labels.jsonl').read_text().splitlines())}
    manifests = {row['question_id']: row for row in (
        json.loads(line) for line in (args.prepared / 'manifests.jsonl').read_text().splitlines())}
    _, _, template, protocol = load_pinned_protocol()

    arms = {}
    for name, root, quantization, scope, residual_bits in (
            ('bf16', args.bf16_root, 'bf16', None, None),
            ('w8a16', args.candidate_root, args.candidate_quantization,
             args.candidate_scope, args.candidate_residual_bits)):
        rows, provenance = load_arm(
            root, cohort, prepared_audit_sha256,
            expected_quantization=quantization, expected_scope=scope,
            expected_residual_bits=residual_bits,
            expected_weight_bits=(args.candidate_weight_bits
                                 if name == 'w8a16' else None),
            expected_group_size=(args.candidate_w12a16_group_size
                                 if name == 'w8a16' and args.candidate_quantization == 'w12a16'
                                 else None))
        arms[name] = {'rows': rows, 'provenance': provenance}
    if [row['question_id'] for row in arms['bf16']['rows']] != [
            row['question_id'] for row in arms['w8a16']['rows']]:
        raise ValueError('serving arms differ in cohort order')

    args.output_dir.mkdir(parents=True, exist_ok=False)
    request_paths = {}
    provenance = {str(args.pair_audit): digest(args.pair_audit),
                  str(cohort_path): digest(cohort_path), str(audit_path): digest(audit_path),
                  **{str(path): digest(path) for path in source_paths}}
    for name in ('bf16', 'w8a16'):
        requests = build_requests({'complete': True, 'rows': arms[name]['rows']},
                                  labels, manifests, template,
                                  allowed_splits=tuple(args.allowed_splits))
        path = args.output_dir / f'{name}.requests.json'
        path.write_text(json.dumps({
            'judge_requests': requests,
            'inputs_sha256': {**provenance, **arms[name]['provenance']},
            'evaluator_commit': protocol['commit'],
            'evaluation_only': True,
            'evaluation_label_files_read': [str(path) for path in source_paths],
            'boundary': 'text-only answer correctness; labels read only after serving completed',
        }, ensure_ascii=False, indent=2) + '\n')
        request_paths[name] = path

    judge_paths = {}
    for name in ('bf16', 'w8a16'):
        judge_path = args.output_dir / f'{name}.judge.json'
        subprocess.run([
            sys.executable, str(ROOT / 'scripts' / 'judge_mobilemem_local.py'),
            '--scores', str(request_paths[name]), '--output', str(judge_path),
            '--model', str(args.judge_model), '--device', args.judge_device,
        ], check=False)
        if not judge_path.exists() or not json.loads(judge_path.read_text()).get('complete'):
            raise ValueError('Judge did not produce a complete full-denominator result')
        judge_paths[name] = judge_path
        provenance[str(request_paths[name])] = digest(request_paths[name])
        provenance[str(judge_path)] = digest(judge_path)

    judged = {}
    for name in ('bf16', 'w8a16'):
        judge = json.loads(judge_paths[name].read_text())
        if not judge.get('complete'):
            raise ValueError('complete judge required')
        requests = json.loads(request_paths[name].read_text())['judge_requests']
        if [row['question_id'] for row in judge['rows']] != [row['question_id'] for row in requests]:
            raise ValueError('judge order differs from serving cohort')
        judged[name] = {row['question_id']: row for row in judge['rows']}

    paired_rows = []
    for reference, candidate in zip(arms['bf16']['rows'], arms['w8a16']['rows']):
        question_id = reference['question_id']
        reference_judge, candidate_judge = judged['bf16'][question_id], judged['w8a16'][question_id]
        paired_rows.append({
            'person_id': str(reference['person_id']), 'question_id': question_id,
            'reference_correct': bool(reference_judge['parse_valid']
                                      and reference_judge['value'] == 1.
                                      and reference['stop_reason'] == 'eos'),
            'candidate_correct': bool(candidate_judge['parse_valid']
                                      and candidate_judge['value'] == 1.
                                      and candidate['stop_reason'] == 'eos'),
            'reference_parse_valid': bool(reference_judge['parse_valid']),
            'candidate_parse_valid': bool(candidate_judge['parse_valid']),
            'reference_eos': reference['stop_reason'] == 'eos',
            'candidate_eos': candidate['stop_reason'] == 'eos',
            'reference_nonempty': bool(reference['answer'].strip()),
            'candidate_nonempty': bool(candidate['answer'].strip()),
            'reference_tokens': len(reference['answer_token_ids']),
            'candidate_tokens': len(candidate['answer_token_ids']),
            'exact_greedy_tokens_equal': (reference['answer_token_ids']
                                          == candidate['answer_token_ids']),
            'answer_text_equal': reference['answer'] == candidate['answer'],
        })
    quality = paired_summary(paired_rows, args.margin_pp, args.bootstrap_samples, args.seed)
    per_user = {person_id: paired_summary(
        [row for row in paired_rows if row['person_id'] == person_id],
        args.margin_pp, args.bootstrap_samples, args.seed)
        for person_id in sorted(cohort['question_ids_by_user'], key=int)}
    secondary = {}
    for key in ('parse_valid', 'eos', 'nonempty'):
        reference_rate = rate(paired_rows, f'reference_{key}')
        candidate_rate = rate(paired_rows, f'candidate_{key}')
        secondary[key] = {
            'reference_rate': reference_rate, 'candidate_rate': candidate_rate,
            'delta_pp': 100 * (candidate_rate - reference_rate),
        }
    secondary['mean_answer_tokens'] = {
        'reference': sum(row['reference_tokens'] for row in paired_rows) / len(paired_rows),
        'candidate': sum(row['candidate_tokens'] for row in paired_rows) / len(paired_rows),
    }
    secondary['exact_greedy_token_matches'] = sum(
        row['exact_greedy_tokens_equal'] for row in paired_rows)
    secondary['answer_text_matches'] = sum(row['answer_text_equal'] for row in paired_rows)
    format_ok = (secondary['parse_valid']['delta_pp'] >= -args.margin_pp
                 and secondary['eos']['delta_pp'] >= -args.margin_pp
                 and secondary['nonempty']['delta_pp'] >= -args.margin_pp)
    result = {
        'schema_version': 'reader-quantization-semantic-gate-v1',
        'complete': True,
        'requests': len(paired_rows),
        'serving_pair': {'reference': 'bf16', 'candidate': args.candidate_label,
                         'candidate_quantization': args.candidate_quantization,
                         'candidate_fallback_scope': args.candidate_scope,
                         'candidate_residual_bits': args.candidate_residual_bits,
                         'single_factor': pair['integrity']['single_factor_reader_quantization'],
                         'fallback_scope_verified': pair['integrity'][
                             'fallback_scope_verified'],
                         'residual_bits_verified': pair['integrity'][
                             'residual_bits_verified'],
                         'upstream_identical': pair['integrity'][
                             'semantic_pre_reader_execution_identical']},
        'preregistered_gate': {
            'primary_endpoint': ('strict local proxy success: valid Judge Yes and EOS, '
                                 'with malformed judgments retained as failures'),
            'noninferiority_margin_pp': -args.margin_pp,
            'confidence': 0.95,
            'cluster_unit': 'user for paired bootstrap; requests paired within user',
            'format_controls': ('parse-valid, EOS and nonempty-answer deltas each must be '
                                'no worse than the same margin'),
            'exact_token_gate': ('reported as a transparency endpoint, not required because '
                                 '8-bit weight quantization need not preserve a greedy path'),
        },
        'quality': quality,
        'per_user_quality': per_user,
        'secondary': secondary,
        'exact_generation_transparency': pair['generation'],
        'gate': {
            'all_requests_retained': len(paired_rows) == len(arms['bf16']['rows']),
            'semantic_noninferiority_passed': quality['noninferiority_passed'],
            'format_noninferiority_passed': format_ok,
            'passed': quality['noninferiority_passed'] and format_ok,
            'strict_exact_generation_passed': pair['gate']['passed'],
        },
        'inputs_sha256': provenance,
        'source_files_sha256': {
            str(path): digest(path) for path in (
                Path(__file__), ROOT / 'scripts' / 'judge_mobilemem_local.py',
                ROOT / 'scripts' / 'build_omni_smoke_judge_requests.py')},
        'evaluation_authorization': evaluation_authorization,
        'serving_evaluation_label_files_read': [],
        'evaluator_label_files_read': [str(path) for path in source_paths],
        'rows': paired_rows,
        'boundary': (f"{'/'.join(args.allowed_splits)} semantic gate on "
                     f"{len(set(row['person_id'] for row in paired_rows))} frozen users; "
                     'conditional user-cluster interval, not population noninferiority; '
                     'same-family local 8B Judge; concurrent screening timings are not '
                     'latency evidence; no physical PIM execution'),
        'performance_comparable': False,
        'actual_physical_pim_executed': False,
        'publication_ready': False,
    }
    with (args.output_dir / 'summary.json').open('x') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print(json.dumps({
        'requests': result['requests'], 'quality': quality,
        'secondary': secondary, 'gate': result['gate'],
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
