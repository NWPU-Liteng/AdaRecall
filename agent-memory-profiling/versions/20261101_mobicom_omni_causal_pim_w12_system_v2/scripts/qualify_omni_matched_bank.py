"""Fixed-prefix matched arithmetic qualification, never a performance benchmark."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qualify_omni_functional_pim import checked, make_inputs, verify_registration
from audit_omni_smoke_closure import digest

SOURCES = ('scripts/qualify_omni_matched_bank.py', 'pim_backend/matched_decode.py',
           'tests/test_matched_decode.py')


def verify_sources(registration):
    for name, expected in registration['source_sha256'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('registered matched source changed')


def worker(path, index):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from pim_backend.matched_decode import MatchedDecodeDispatcher
    from pim_backend.online_residency import OnlineResidency
    from omni_execution_input_contract import execution_input_contract

    registration = json.loads(path.read_text())
    verify_sources(registration)
    base = checked(Path(registration['base_registration']), registration['base_registration_sha256'])
    verify_registration(base)
    plan = checked(Path(base['execution_plan']), base['execution_plan_sha256'])
    routes = checked(Path(base['routes']), base['routes_sha256'])
    profile = checked(Path(base['profile']), routes['profile_sha256'])
    job = plan['jobs'][index]
    smoke = checked(Path(job['smoke']), job['smoke_sha256'])
    checkpoint = Path(plan['reader'])
    for name, expected in {**plan['checkpoint_sha256'], **plan['processor_files_sha256']}.items():
        if digest(checkpoint / name) != expected:
            raise ValueError('Reader checkpoint or processor differs')
    manifest_path = Path(plan['prepared']) / 'manifests.jsonl'
    if digest(manifest_path) != plan['manifests_sha256'] or digest(Path(plan['archive'])) != plan['archive_sha256']:
        raise ValueError('canonical data changed')
    manifests = {row['question_id']: row for row in map(json.loads, manifest_path.read_text().splitlines())}
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    processor = AutoProcessor.from_pretrained(str(checkpoint), local_files_only=True)
    processor.image_processor.size['longest_edge'] = smoke['image_max_pixels']
    reader = AutoModelForImageTextToText.from_pretrained(str(checkpoint), local_files_only=True,
        dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda:0').eval()
    eos = reader.config.text_config.eos_token_id
    stops = set(eos if isinstance(eos, list) else [eos])
    output_root = Path(registration['output_root']) / ('user_' + job['person_id'])
    output_root.mkdir(exist_ok=False)
    results, host_references = [], {}
    with zipfile.ZipFile(plan['archive']) as archive, torch.inference_mode():
        for mode in ('matched_host', 'matched_replace'):
            if mode == 'matched_replace' and (len(host_references) != registration['requests_per_user']
                    or not all(row['all_selected_matrices_observed'] and all(
                        record['numerics']['forward_error_bound_pass'] for record in row['full_matrix_checks'])
                        for row in results)):
                raise ValueError('complete matched-host matrix qualification required')
            residency = OnlineResidency(routes['costs'], routes['placement'],
                capacity_bytes=routes['capacity_bytes'], reserved_bytes=routes['shared_reserved_bytes'])
            with MatchedDecodeDispatcher(reader, residency,
                    mode='shadow' if mode == 'matched_host' else 'replace',
                    banks=profile['config']['channels'] * profile['config']['banks_per_channel']) as dispatcher:
                for original in smoke['rows'][:registration['requests_per_user']]:
                    manifest = manifests[original['question_id']]
                    if (manifest['split'] != 'development' or manifest['visibility_protocol'] != 'history_end'
                            or manifest['person_id'] != job['person_id']):
                        raise ValueError('fixed development cohort required')
                    inputs = make_inputs(processor, original, manifest, archive)
                    contract = execution_input_contract(inputs, original)
                    before = len(dispatcher.records)
                    output = reader(**inputs.to('cuda:0'), use_cache=True, logits_to_keep=1)
                    tokens, reason, invocation = [], 'max_new_tokens', 0
                    for position in range(smoke['max_new_tokens']):
                        token = output.logits[:, -1, :].argmax(-1, keepdim=True)
                        tokens.append(int(token.item()))
                        if tokens[-1] in stops:
                            reason = 'eos'
                            break
                        if position + 1 == smoke['max_new_tokens']:
                            break
                        invocation += 1
                        with dispatcher.decode(observe=invocation == 2):
                            output = reader(input_ids=token, past_key_values=output.past_key_values,
                                            use_cache=True, logits_to_keep=1)
                    records = dispatcher.records[before:]
                    answer = processor.decode(tokens, skip_special_tokens=True)
                    row = {'mode': mode, 'question_id': original['question_id'], 'person_id': job['person_id'],
                        'question_sha256': original['question_sha256'], 'input_contract': contract,
                        'answer': answer, 'answer_sha256': hashlib.sha256(answer.encode()).hexdigest(),
                        'answer_token_ids': tokens, 'stop_reason': reason, 'decode_invocations': invocation,
                        'full_matrix_checks': records, 'evaluation_label_files_read': [],
                        'all_selected_matrices_observed': {record['parameter_identity'] for record in records}
                                                         == set(routes['placement']),
                        'tokens_equal_old_reference': tokens == original['answer_token_ids']}
                    if mode == 'matched_host':
                        host_references[original['question_id']] = {'tokens': tokens, 'stop_reason': reason}
                    else:
                        reference = host_references[original['question_id']]
                        row['tokens_equal_matched_host'] = tokens == reference['tokens']
                        row['stop_equal_matched_host'] = reason == reference['stop_reason']
                    results.append(row)
                    (output_root / 'partial.json').write_text(json.dumps({'complete': False, 'rows': results},
                                                                       ensure_ascii=False, indent=2))
                    print(json.dumps({key: value for key, value in row.items() if key in (
                        'mode', 'question_id', 'stop_reason', 'tokens_equal_matched_host',
                        'tokens_equal_old_reference', 'decode_invocations')}), flush=True)
                    del output, inputs
                (output_root / (mode + '_dispatch.json')).write_text(json.dumps({
                    'selected_calls': dispatcher.calls, 'host_calls': dispatcher.host_calls,
                    'modeled_admitted_calls': dispatcher.pim_calls,
                    'bank_function_replacement_calls': dispatcher.replaced_host_calls,
                    'residency_certificate': residency.certificate()}, indent=2))
    verify_sources(registration)
    verify_registration(base)
    with (output_root / 'result.json').open('x') as handle:
        json.dump({'complete': True, 'registration_sha256': digest(path), 'rows': results,
            'evaluation_label_files_read': [], 'end_to_end_speedup': None,
            'actual_physical_pim_executed': False}, handle, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-registration', type=Path)
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--gpus', nargs='+', default=['3', '4', '5'])
    parser.add_argument('--worker-registration', type=Path)
    parser.add_argument('--worker-index', type=int)
    args = parser.parse_args()
    if args.worker_registration:
        worker(args.worker_registration, args.worker_index)
        return
    if not args.base_registration or not args.output_root:
        parser.error('base registration and fresh output root required')
    base = json.loads(args.base_registration.read_text())
    verify_registration(base)
    plan = checked(Path(base['execution_plan']), base['execution_plan_sha256'])
    if len(args.gpus) != len(plan['jobs']) or len(set(args.gpus)) != len(args.gpus):
        raise ValueError('one distinct GPU per frozen user required')
    if base['requests_per_user'] != 2:
        raise ValueError('this qualification must retain the original two-request prefix')
    args.output_root.mkdir(parents=True, exist_ok=False)
    registration = {'schema_version': 'omni-matched-bank-qualification-v1',
        'base_registration': str(args.base_registration.resolve()),
        'base_registration_sha256': digest(args.base_registration),
        'output_root': str(args.output_root.resolve()), 'requests_per_user': 2,
        'source_sha256': {name: digest(ROOT / name) for name in SOURCES},
        'arithmetic_contract': 'Only selected decode Linears: host BF16 I/O with BF16 reduced-precision '
            'reduction disabled; bank BF16 weights/input, FP32 tile accumulation, BF16 output. Prefill '
            'and nonselected work unchanged. Separate cold residency per user/mode. No answer-based fallback.',
        'selection': 'same original first two requests per user; no outcome-dependent selection',
        'quality_claim': False, 'performance_claim': False, 'evaluation_label_files_read': [],
        'execution_isolation': 'one single-request model process per GPU; global precision flag restored '
                               'after each selected host Linear, not qualified for concurrent serving'}
    path = args.output_root / 'registration.json'
    path.write_text(json.dumps(registration, indent=2))
    processes, handles = [], []
    try:
        for index, gpu in enumerate(args.gpus):
            log = (args.output_root / ('worker_' + str(index) + '.log')).open('x')
            handles.append(log)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
            processes.append(subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                '--worker-registration', str(path.resolve()), '--worker-index', str(index)],
                env=env, stdout=log, stderr=subprocess.STDOUT))
        codes = [process.wait() for process in processes]
    finally:
        for log in handles:
            log.close()
    with (args.output_root / 'summary.json').open('x') as handle:
        json.dump({'complete': not any(codes), 'worker_exit_codes': codes,
            'registration_sha256': digest(path), 'quality_evaluated': False,
            'actual_physical_pim_executed': False, 'end_to_end_speedup': None}, handle, indent=2)
    raise SystemExit(int(any(codes)))


if __name__ == '__main__':
    main()
