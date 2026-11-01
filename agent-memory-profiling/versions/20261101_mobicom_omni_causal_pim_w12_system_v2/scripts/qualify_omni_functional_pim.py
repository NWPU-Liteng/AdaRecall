"""Full selected-matrix qualification and real Reader dispatch, without timing claims."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from audit_omni_smoke_closure import digest


SOURCES = ('scripts/qualify_omni_functional_pim.py', 'pim_backend/functional_decode.py',
           'pim_backend/online_residency.py', 'pim_backend/decode_gemv.py',
           'scripts/omni_execution_input_contract.py', 'scripts/omni_visual_feature_store.py',
           'scripts/run_omni_raw_image_smoke.py', 'tests/test_functional_decode.py')


def checked(path, expected):
    if digest(path) != expected:
        raise ValueError('frozen artifact changed: ' + str(path))
    return json.loads(path.read_text())


def verify_registration(registration):
    for name, expected in registration['source_sha256'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('qualification source differs from registration')


def make_inputs(processor, original, manifest, archive):
    from PIL import Image
    from run_omni_raw_image_smoke import SYSTEM

    content = [{'type': 'text', 'text': '原始对话记忆：\n局部对话超边：\n'
        + original['hypergraph_evidence_plan']['text']
        + f"\n\n查询时间协议：完整历史末端（{manifest['effective_timestamp']}）"
        + f"\n问题：{manifest['question']}\n请依据文字和随后附上的原始图片作答。"}]
    images = []
    for image in original['images']:
        raw = archive.read(image['archive_member'])
        if hashlib.sha256(raw).hexdigest() != image['sha256']:
            raise ValueError('original image bytes differ')
        decoded = Image.open(io.BytesIO(raw)).convert('RGB')
        images.append(decoded)
        content.extend([{'type': 'text', 'text':
            f"以下原图来自会话 {image['source_session_id']} 的对话轮 {image['source_atom_id']}："},
            {'type': 'image', 'image': decoded}])
    text = processor.apply_chat_template([{'role': 'system', 'content': SYSTEM},
        {'role': 'user', 'content': content}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    return processor(text=[text], images=images, return_tensors='pt')


def worker(registration_path, worker_index):
    import torch
    import transformers
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from omni_execution_input_contract import execution_input_contract
    from run_omni_raw_image_smoke import SYSTEM
    from pim_backend.functional_decode import DecodeBankDispatcher
    from pim_backend.online_residency import OnlineResidency

    registration = json.loads(registration_path.read_text())
    verify_registration(registration)
    plan = checked(Path(registration['execution_plan']), registration['execution_plan_sha256'])
    routes = checked(Path(registration['routes']), registration['routes_sha256'])
    profile = checked(Path(registration['profile']), routes['profile_sha256'])
    job = plan['jobs'][worker_index]
    smoke = checked(Path(job['smoke']), job['smoke_sha256'])
    checkpoint = Path(plan['reader'])
    for name, expected in plan['checkpoint_sha256'].items():
        if digest(checkpoint / name) != expected:
            raise ValueError('full Reader checkpoint changed')
    for name, expected in plan['processor_files_sha256'].items():
        if digest(checkpoint / name) != expected:
            raise ValueError('processor configuration changed')
    if digest(Path(plan['archive'])) != smoke['archive_sha256']:
        raise ValueError('image archive differs')
    if hashlib.sha256(SYSTEM.encode()).hexdigest() != smoke['system_prompt_sha256']:
        raise ValueError('Reader system prompt differs')
    manifest_path = Path(plan['prepared']) / 'manifests.jsonl'
    checked_hash = digest(manifest_path)
    if checked_hash != plan['manifests_sha256']:
        raise ValueError('serving manifests changed')
    manifests = {row['question_id']: row for row in map(json.loads, manifest_path.read_text().splitlines())}
    processor = AutoProcessor.from_pretrained(str(checkpoint), local_files_only=True)
    processor.image_processor.size['longest_edge'] = smoke['image_max_pixels']
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    reader = AutoModelForImageTextToText.from_pretrained(str(checkpoint), local_files_only=True,
        dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda:0').eval()
    stop = reader.config.text_config.eos_token_id
    stops = set(stop if isinstance(stop, list) else [stop])
    output_root = Path(registration['output_root']) / ('user_' + job['person_id'])
    output_root.mkdir(exist_ok=False)
    results = []
    with zipfile.ZipFile(plan['archive']) as archive, torch.inference_mode():
        for mode in registration['modes']:
            if mode == 'replace' and not all(row['all_selected_matrices_observed']
                    and row['complete_greedy_tokens_equal_frozen'] and row['stop_reason_equal_frozen']
                    and all(record['numerics']['forward_error_bound_pass']
                            and record['numerics']['host_bf16_forward_error_bound_pass']
                            for record in row['full_matrix_checks']) for row in results):
                raise ValueError('shadow qualification failed; replacement must not proceed')
            residency = OnlineResidency(routes['costs'], routes['placement'],
                capacity_bytes=routes['capacity_bytes'], reserved_bytes=routes['shared_reserved_bytes'])
            with DecodeBankDispatcher(reader, residency, mode=mode,
                    banks=profile['config']['channels'] * profile['config']['banks_per_channel']) as dispatcher:
                for original in smoke['rows'][:registration['requests_per_user']]:
                    question_id = original['question_id']
                    manifest = manifests[question_id]
                    if (manifest['split'] != 'development' or manifest['visibility_protocol'] != 'history_end'
                            or manifest['person_id'] != job['person_id']
                            or hashlib.sha256(manifest['question'].encode()).hexdigest() != original['question_sha256']):
                        raise ValueError('frozen development request differs')
                    inputs = make_inputs(processor, original, manifest, archive)
                    contract = execution_input_contract(inputs, original)
                    inputs = inputs.to('cuda:0')
                    before = len(dispatcher.records)
                    started = time.perf_counter()
                    output = reader(**inputs, use_cache=True, logits_to_keep=1)
                    generated, reason, invocation = [], 'max_new_tokens', 0
                    for position in range(smoke['max_new_tokens']):
                        token = output.logits[:, -1, :].argmax(-1, keepdim=True)
                        generated.append(int(token.item()))
                        if generated[-1] in stops:
                            reason = 'eos'
                            break
                        if position + 1 == smoke['max_new_tokens']:
                            break
                        invocation += 1
                        with dispatcher.decode(observe=invocation == 2):
                            output = reader(input_ids=token, past_key_values=output.past_key_values,
                                            use_cache=True, logits_to_keep=1)
                    torch.cuda.synchronize()
                    records = dispatcher.records[before:]
                    row = {'mode': mode, 'question_id': question_id, 'input_contract': contract,
                        'answer_token_ids': generated, 'answer': processor.decode(generated, skip_special_tokens=True),
                        'stop_reason': reason,
                        'complete_greedy_tokens_equal_frozen': generated == original['answer_token_ids'],
                        'stop_reason_equal_frozen': reason == original['stop_reason'],
                        'decode_invocations': invocation, 'full_matrix_checks': records,
                        'all_selected_matrices_observed': {record['parameter_identity'] for record in records}
                                                        == set(routes['placement']),
                        'diagnostic_seconds_not_service_latency': time.perf_counter() - started,
                        'evaluation_label_files_read': []}
                    results.append(row)
                    (output_root / 'partial.json').write_text(json.dumps({'complete': False, 'rows': results},
                                                                        ensure_ascii=False, indent=2))
                    print(json.dumps({key: row[key] for key in ('mode', 'question_id',
                        'complete_greedy_tokens_equal_frozen', 'all_selected_matrices_observed',
                        'diagnostic_seconds_not_service_latency')}), flush=True)
                    del output, inputs
            (output_root / (mode + '_dispatch.json')).write_text(json.dumps({
                'selected_calls': dispatcher.calls, 'host_calls': dispatcher.host_calls,
                'modeled_admitted_calls': dispatcher.pim_calls,
                'bank_function_replacement_calls': dispatcher.replaced_host_calls,
                'residency_certificate': residency.certificate()}, indent=2))
    verify_registration(registration)
    result = {'complete': True, 'rows': results, 'registration_sha256': digest(registration_path),
        'person_id': job['person_id'], 'torch': torch.__version__, 'transformers': transformers.__version__,
        'reader_dispatch_integration_executed': 'replace' in registration['modes'],
        'actual_physical_pim_executed': False, 'performance_comparable': False,
        'end_to_end_speedup': None, 'evaluation_label_files_read': [], 'new_quality_evaluated': False,
        'boundary': 'full selected matrices at actual decode activations; GPU executes FP32 tiled bank-layout function with BF16 I/O; not physical PIM or performance simulation; one second-invocation numerical sample per request, not all-token per-op certification'}
    with (output_root / 'result.json').open('x') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--execution-root', type=Path)
    parser.add_argument('--route-root', type=Path)
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--requests-per-user', type=int, default=1)
    parser.add_argument('--modes', nargs='+', choices=('shadow', 'replace'), default=['shadow'])
    parser.add_argument('--gpus', nargs='+', default=['3', '4', '5'])
    parser.add_argument('--worker-registration', type=Path)
    parser.add_argument('--worker-index', type=int)
    args = parser.parse_args()
    if args.worker_registration:
        worker(args.worker_registration, args.worker_index)
        return
    if not args.execution_root or not args.route_root or not args.output_root or args.requests_per_user < 1:
        parser.error('frozen execution root, route root, fresh output and positive prefix length required')
    from audit_omni_pim_routes import audit_routes

    audit = audit_routes(args.route_root, args.execution_root / 'cohort_audit.json', args.execution_root)
    if not audit['complete']:
        raise ValueError('independent frozen route reconstruction must pass')
    plan_path = args.execution_root / 'execution_plan.json'
    plan = json.loads(plan_path.read_text())
    routes_path = args.route_root / 'routes.json'
    routes = json.loads(routes_path.read_text())
    expected_audit = routes['producer_audit_sha256']
    if digest(args.execution_root / 'cohort_audit.json') != expected_audit:
        raise ValueError('route must originate from requested canonical execution')
    if any(args.requests_per_user > len(job['question_ids']) for job in plan['jobs']):
        raise ValueError('prefix exceeds frozen cohort')
    if 'replace' in args.modes and args.modes != ['shadow', 'replace']:
        raise ValueError('replacement requires a preceding shadow qualification gate')
    args.output_root.mkdir(parents=True, exist_ok=False)
    registration = {'schema_version': 'omni-full-matrix-functional-qualification-v1',
        'execution_plan': str(plan_path.resolve()), 'execution_plan_sha256': digest(plan_path),
        'routes': str(routes_path.resolve()), 'routes_sha256': digest(routes_path),
        'profile': str((args.route_root / 'profile.json').resolve()),
        'output_root': str(args.output_root.resolve()), 'requests_per_user': args.requests_per_user,
        'selection': 'first registered requests per user in original order, independent of outcomes',
        'modes': args.modes, 'gpus': args.gpus,
        'source_sha256': {name: digest(ROOT / name) for name in SOURCES},
        'quality_claim': False, 'performance_claim': False, 'evaluation_label_files_read': []}
    registration_path = args.output_root / 'registration.json'
    registration_path.write_text(json.dumps(registration, ensure_ascii=False, indent=2))
    active, pending, statuses = {}, list(range(len(plan['jobs']))), []
    while pending or active:
        for index in tuple(pending):
            gpu = args.gpus[index % len(args.gpus)]
            if gpu in active:
                continue
            log = (args.output_root / f'worker_{index}.log').open('x')
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS='2',
                               MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                '--worker-registration', str(registration_path.resolve()), '--worker-index', str(index)],
                env=environment, stdout=log, stderr=subprocess.STDOUT)
            active[gpu] = (index, process, log)
            pending.remove(index)
        for gpu, (index, process, log) in tuple(active.items()):
            code = process.poll()
            if code is not None:
                log.close()
                statuses.append({'worker_index': index, 'exit_code': code})
                del active[gpu]
        if active:
            time.sleep(2)
    complete = all(status['exit_code'] == 0 for status in statuses)
    summary = {'complete': complete, 'workers': statuses, 'registration_sha256': digest(registration_path),
               'end_to_end_speedup': None, 'performance_comparable': False}
    if complete:
        results = [json.loads((args.output_root / ('user_' + job['person_id']) / 'result.json').read_text())
                   for job in plan['jobs']]
        rows = [row for result in results for row in result['rows']]
        checks = [record for row in rows for record in row['full_matrix_checks']]
        summary.update({'request_executions': len(rows), 'full_matrix_checks': len(checks),
            'all_complete_tokens_equal_frozen': all(row['complete_greedy_tokens_equal_frozen'] for row in rows),
            'all_selected_matrices_observed': all(row['all_selected_matrices_observed'] for row in rows),
            'all_fp32_bounds_pass': all(record['numerics']['forward_error_bound_pass'] for record in checks),
            'all_host_bf16_bounds_pass': all(record['numerics']['host_bf16_forward_error_bound_pass'] for record in checks),
            'all_host_bf16_bitwise_equal': all(record['numerics']['host_bf16_bitwise_equal'] for record in checks),
            'actual_physical_pim_executed': False, 'publication_ready': False})
    (args.output_root / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    if not complete:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
