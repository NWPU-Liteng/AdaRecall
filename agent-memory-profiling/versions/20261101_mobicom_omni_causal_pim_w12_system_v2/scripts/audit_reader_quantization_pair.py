"""Strict label-free BF16 versus W8A8 Reader pairing for the quality gate."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PROTOCOL_FIELDS = (
    'prepared_audit_sha256', 'archive_sha256', 'reader_config_sha256',
    'embedder_config_sha256', 'system_prompt_sha256', 'evidence_policy',
    'candidate_sessions', 'max_images', 'max_new_tokens', 'session_text_tokens',
    'total_text_tokens', 'image_max_pixels', 'reader_patch_projection',
    'relevance_modality', 'joint_patch_projection', 'image_selection_policy',
)
IDENTITY_FIELDS = (
    'source_files_sha256', 'embedder_identity_files_sha256',
    'embedding_execution_identity', 'visual_state_identity_files_sha256',
)
QUANTIZATION_SOURCE_ALLOWLIST = frozenset({
    'pim_backend/functional_decode.py',
    'scripts/run_omni_raw_image_smoke.py',
})
UPSTREAM_FIELDS = (
    'question_sha256', 'effective_timestamp', 'visibility_protocol',
    'input_tokens', 'visual_patches', 'retrieved_session_ids', 'images',
    'evidence_selection', 'hypergraph_evidence_plan', 'embedding_producers',
    'processor_inputs_sha256',
)
VOLATILE_METRIC_KEYS = frozenset(('lookup_seconds', 'encode_seconds', 'write_seconds'))


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            result.update(block)
    return result.hexdigest()


def load_json(path):
    path = Path(path)
    return path, json.loads(path.read_text())


def first_divergence(reference, candidate):
    common = 0
    for left, right in zip(reference, candidate):
        if left != right:
            break
        common += 1
    return {
        'common_prefix_tokens': common,
        'reference_token_at_divergence': reference[common] if common < len(reference) else None,
        'candidate_token_at_divergence': candidate[common] if common < len(candidate) else None,
        'reference_tokens': len(reference), 'candidate_tokens': len(candidate),
    }


def semantic(value):
    if isinstance(value, dict):
        return {key: semantic(item) for key, item in value.items()
                if key not in VOLATILE_METRIC_KEYS}
    if isinstance(value, list):
        return [semantic(item) for item in value]
    return value


def validate_run(run, expected_quantization, expected_ids, cohort_sha256,
                 expected_scope=None, expected_residual_bits=None,
                 expected_weight_bits=None, expected_group_size=None):
    if not run.get('complete') or not run.get('rows'):
        raise ValueError('complete nonempty run required')
    if run.get('evaluation_label_files_read') != []:
        raise ValueError('quality pair must remain label-isolated')
    if run.get('reader_quantization') != expected_quantization:
        raise ValueError('Reader quantization arm differs from registration')
    if expected_scope is not None:
        if expected_quantization != 'w8a16':
            raise ValueError('W8A16 fallback scope requires the W8A16 arm')
        if run.get('reader_w8a16_fallback_scope') != expected_scope:
            raise ValueError('Reader W8A16 fallback scope differs from registration')
    if expected_residual_bits is not None:
        if expected_residual_bits and expected_quantization != 'w8a16':
            raise ValueError('W8A16 residual correction requires the W8A16 arm')
        if run.get('reader_w8a16_residual_bits', 0) != expected_residual_bits:
            raise ValueError('Reader residual bits differ from registration')
    if expected_weight_bits is not None and run.get('reader_weight_bits') != expected_weight_bits:
        raise ValueError('Reader weight bits differ from registration')
    if expected_group_size is not None:
        if expected_quantization != 'w12a16':
            raise ValueError('W12A16 group size requires the W12A16 arm')
        if run.get('reader_w12a16_group_size') != expected_group_size:
            raise ValueError('Reader W12A16 group size differs from registration')
    if run.get('prepared_audit_sha256') != cohort_sha256:
        raise ValueError('run and registered prepared audit differ')
    observed = [row.get('question_id') for row in run['rows']]
    if observed != expected_ids:
        raise ValueError('run order differs from registered cohort')
    if len(observed) != len(set(observed)):
        raise ValueError('duplicate request identity')
    for row in run['rows']:
        if not isinstance(row.get('answer_token_ids'), list) or not row['answer_token_ids']:
            raise ValueError('nonempty greedy token sequence required')
        if row.get('stop_reason') not in {'eos', 'max_new_tokens'}:
            raise ValueError('known stop reason required')


def validate_registration(run, run_root, cohort_path):
    user = run['rows'][0]['person_id']
    registration_path = Path(run_root) / f'baseline_user_{user}.registration.json'
    _, registration = load_json(registration_path)
    if registration.get('question_ids') != [row['question_id'] for row in run['rows']]:
        raise ValueError('registration question order differs')
    cohort_copy = Path(run_root) / 'cohort.json'
    if digest(cohort_copy) != digest(cohort_path):
        raise ValueError('runtime cohort copy differs from registered cohort')


def quantization_source_differences(reference, candidate):
    reference = reference or {}
    candidate = candidate or {}
    if set(reference) != set(candidate):
        raise ValueError('source identity inventory differs')
    differences = sorted(key for key in reference
                         if reference[key] != candidate[key])
    unexpected = sorted(set(differences) - QUANTIZATION_SOURCE_ALLOWLIST)
    if unexpected:
        raise ValueError('non-quantization source identity differs: '
                         + ','.join(unexpected))
    return differences


def audit(bf16_root, candidate_root, cohort_path, candidate_quantization='w8a8',
          candidate_scope=None, candidate_residual_bits=None,
          candidate_weight_bits=None, candidate_group_size=None):
    cohort_path, cohort = load_json(cohort_path)
    if (cohort.get('evaluation_label_files_read') != []
            or cohort.get('runtime_labels_read', False) is not False):
        raise ValueError('registered cohort must be label-isolated')
    prepared_sha256 = cohort.get('prepared_audit_sha256')
    if not prepared_sha256:
        raise ValueError('registered prepared audit required')
    runs = {'bf16': {}, 'candidate': {}}
    for arm, root, quantization, scope, residual_bits in (
            ('bf16', bf16_root, 'bf16', None, None),
            ('candidate', candidate_root, candidate_quantization, candidate_scope,
             candidate_residual_bits)):
        root = Path(root)
        for user, expected_ids in sorted(cohort['question_ids_by_user'].items()):
            path, run = load_json(root / f'baseline_user_{user}.json')
            validate_run(run, quantization, expected_ids, prepared_sha256, scope,
                         residual_bits,
                         candidate_weight_bits if arm == 'candidate' else None,
                         candidate_group_size if arm == 'candidate' else None)
            validate_registration(run, root, cohort_path)
            runs[arm][user] = {'path': path, 'run': run}
    source_differences = {}
    for user in runs['bf16']:
        reference = runs['bf16'][user]['run']
        candidate = runs['candidate'][user]['run']
        for field in PROTOCOL_FIELDS:
            if reference.get(field) != candidate.get(field):
                raise ValueError(f'non-quantization protocol field differs: {field}')
        for field in IDENTITY_FIELDS:
            if field == 'source_files_sha256':
                source_differences[user] = quantization_source_differences(
                    reference.get(field), candidate.get(field))
                continue
            if reference.get(field) != candidate.get(field):
                raise ValueError(f'non-quantization model or source identity differs: {field}')
        for left, right in zip(reference['rows'], candidate['rows']):
            differing = [field for field in UPSTREAM_FIELDS
                         if semantic(left.get(field)) != semantic(right.get(field))]
            if differing:
                raise ValueError('pre-Reader execution differs: ' + ','.join(differing))

    rows, token_matches = [], 0
    stop_matches = answer_matches = 0
    for user in sorted(runs['bf16'], key=int):
        reference = runs['bf16'][user]['run']['rows']
        candidate = runs['candidate'][user]['run']['rows']
        for left, right in zip(reference, candidate):
            tokens_equal = left['answer_token_ids'] == right['answer_token_ids']
            stop_equal = left['stop_reason'] == right['stop_reason']
            answer_equal = left['answer'] == right['answer']
            token_matches += tokens_equal
            stop_matches += stop_equal
            answer_matches += answer_equal
            rows.append({
                'person_id': user, 'question_id': left['question_id'],
                'complete_greedy_tokens_equal': tokens_equal,
                'stop_reason_equal': stop_equal, 'answer_text_equal': answer_equal,
                'reference_stop_reason': left['stop_reason'],
                'candidate_stop_reason': right['stop_reason'],
                'token_alignment': first_divergence(
                    left['answer_token_ids'], right['answer_token_ids']),
                'upstream_identical': True,
            })
    count = len(rows)
    candidate_arm = candidate_quantization
    if candidate_residual_bits:
        candidate_arm += f'_r{candidate_residual_bits}'
    if candidate_scope not in (None, 'none'):
        candidate_arm += f'_{candidate_scope}'
    result = {
        'schema_version': 'reader-quantization-paired-audit-v1',
        'complete': True,
        'requests': count,
        'arms': {'reference': 'bf16', 'candidate': candidate_arm},
        'candidate_fallback_scope': candidate_scope,
        'candidate_residual_bits': candidate_residual_bits,
        'registration': {
            'cohort': str(cohort_path), 'cohort_sha256': digest(cohort_path),
            'prepared_audit_sha256': prepared_sha256,
            'evaluation_label_files_read': [],
        },
        'integrity': {
            'complete_request_coverage': True,
            'semantic_pre_reader_execution_identical': True,
            'single_factor_reader_quantization': True,
            'fallback_scope_verified': candidate_scope is not None,
            'residual_bits_verified': candidate_residual_bits is not None,
            'source_and_model_identity_identical_except_quantization': True,
            'quantization_source_allowlist': sorted(QUANTIZATION_SOURCE_ALLOWLIST),
            'quantization_source_differences_by_user': source_differences,
            'excluded_volatile_metric_keys': sorted(VOLATILE_METRIC_KEYS),
        },
        'generation': {
            'complete_greedy_token_matches': token_matches,
            'stop_reason_matches': stop_matches,
            'answer_text_matches': answer_matches,
        },
        'rows': rows,
        'gate': {
            'strict_exact_generation_required': True,
            'passed': token_matches == count and stop_matches == count and answer_matches == count,
        },
        'inputs_sha256': {
            str(runs[arm][user]['path']): digest(runs[arm][user]['path'])
            for arm in ('bf16', 'candidate') for user in runs[arm]
        },
        'boundary': 'label-free same-cohort quality audit; exact pre-Reader operands and complete '
                    'greedy outputs must match; no Judge score substitutes for token equality; '
                    'concurrent screening timings are not performance evidence',
        'performance_comparable': False,
        'actual_physical_pim_executed': False,
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bf16-root', type=Path, required=True)
    parser.add_argument('--candidate-root', type=Path)
    parser.add_argument('--w8a8-root', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--candidate-quantization', choices=('w8a8', 'w8a16', 'w12a16'), default='w8a8')
    parser.add_argument('--candidate-scope',
                        choices=('none', 'lm_head', 'output_sensitive'),
                        default=None,
                        help='verify the registered W8A16 fallback scope')
    parser.add_argument('--candidate-residual-bits', type=int, choices=(0, 2, 4),
                        default=None,
                        help='verify the registered W8A16 residual correction')
    parser.add_argument('--candidate-weight-bits', type=int, choices=(8, 12), default=None)
    parser.add_argument('--candidate-w12a16-group-size', type=int,
                        choices=(8, 16, 32, 64, 128, 256), default=None)
    parser.add_argument('--cohort', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('fresh paired audit output required')
    candidate_root = args.candidate_root or args.w8a8_root
    if candidate_root is None:
        parser.error('candidate root required')
    result = audit(args.bf16_root, candidate_root, args.cohort,
                   args.candidate_quantization, args.candidate_scope,
                   args.candidate_residual_bits, args.candidate_weight_bits,
                   args.candidate_w12a16_group_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print(json.dumps({'requests': result['requests'], **result['generation'],
                      'gate': result['gate']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
