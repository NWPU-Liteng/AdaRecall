"""Fixed-cohort raw-image replay with complete greedy output and first-step DAG."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

from audit_omni_smoke_closure import digest, verify_source_snapshot
from omni_prefix_opportunities import PrefixOpportunityIndex, multimodal_prefix_symbols

ROOT = Path(__file__).resolve().parents[1]
SOURCE_NAMES = ('scripts/run_omni_execution_cohort.py', 'scripts/omni_execution_graph.py',
                'scripts/omni_prefix_opportunities.py', 'scripts/audit_omni_smoke_closure.py',
                'scripts/run_omni_raw_image_smoke.py', 'scripts/omni_visual_feature_store.py',
                'scripts/audit_omni_execution_cohort.py', 'tests/test_omni_opaque_regions.py',
                'scripts/omni_execution_input_contract.py')


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def check_sources(plan):
    if any(digest(ROOT / name) != expected for name, expected in plan['source_sha256'].items()):
        raise ValueError('execution source differs from pre-registered snapshot')


def register(args):
    cohort = json.loads(args.cohort_plan.read_text())
    anchor = json.loads(args.identity_smoke.read_text())
    anchor_registration = json.loads(args.identity_smoke.with_suffix('.registration.json').read_text())
    if (not anchor.get('complete') or not anchor.get('verify_generation')
            or not verify_source_snapshot(anchor, anchor_registration)
            or not anchor.get('visual_state_identity_files_sha256')):
        raise ValueError('independently verified checkpoint identity anchor required')
    checkpoint = Path(anchor['reader'])
    for name, expected in anchor['visual_state_identity_files_sha256'].items():
        if Path(name).name != name or digest(checkpoint / name) != expected:
            raise ValueError('immutable Reader checkpoint differs')
    if digest(args.image_archive) != anchor['archive_sha256']:
        raise ValueError('original image archive differs')
    audit_path = args.prepared / 'audit.json'
    prepared_audit = json.loads(audit_path.read_text())
    manifests_path = args.prepared / 'manifests.jsonl'
    if digest(manifests_path) != prepared_audit['artifacts_sha256']['manifests.jsonl']:
        raise ValueError('serving manifests differ')
    manifests = {row['question_id']: row for row in map(json.loads, manifests_path.read_text().splitlines())}
    jobs, users = [], set()
    for index, original in enumerate(cohort['jobs']):
        smoke_path = Path(original['output'])
        smoke = json.loads(smoke_path.read_text())
        registration = json.loads(smoke_path.with_suffix('.registration.json').read_text())
        if (original['person_id'] in users or not smoke.get('complete')
                or smoke.get('evaluation_label_files_read') != []
                or smoke['prepared_audit_sha256'] != digest(audit_path)
                or smoke['reader'] != anchor['reader']
                or smoke['reader_config_sha256'] != anchor['reader_config_sha256']
                or smoke['visual_state_identity_files_sha256'] != anchor['visual_state_identity_files_sha256']
                or smoke['archive_sha256'] != anchor['archive_sha256']
                or registration['source_files_sha256'] != smoke['source_files_sha256']
                or not verify_source_snapshot(smoke, registration)
                or original['question_ids'] != [row['question_id'] for row in smoke['rows']]
                or registration['question_ids'] != original['question_ids']):
            raise ValueError('complete frozen cohort with immutable source and order required')
        for row in smoke['rows']:
            manifest = manifests[row['question_id']]
            if (manifest['split'] != 'development' or manifest['person_id'] != original['person_id']
                    or manifest['visibility_protocol'] != 'history_end'
                    or manifest['effective_timestamp'] != row['effective_timestamp']
                    or hashlib.sha256(manifest['question'].encode()).hexdigest() != row['question_sha256']
                    or not row['hypergraph_evidence_plan'] or not row['answer_token_ids']):
                raise ValueError('frozen development input identity differs')
        users.add(original['person_id'])
        jobs.append({'person_id': original['person_id'], 'question_ids': original['question_ids'],
                     'smoke': str(smoke_path), 'smoke_sha256': digest(smoke_path),
                     'gpu': args.gpus[index % len(args.gpus)]})
    if not jobs:
        raise ValueError('nonempty whole frozen cohort required')
    args.output_root.mkdir(parents=True, exist_ok=False)
    snapshot = args.output_root / 'sources'
    sources = {name: digest(ROOT / name) for name in SOURCE_NAMES}
    for name, expected in sources.items():
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
        if digest(target) != expected:
            raise ValueError('execution source changed during snapshot')
    reader_files = {name: digest(path) for path in checkpoint.iterdir()
                    if path.is_file() and path.suffix in {'.json', '.txt', '.jinja'}
                    for name in [path.name]}
    plan = {'schema_version': 'omni-execution-cohort-plan-v1', 'jobs': jobs,
        'cohort_plan': str(args.cohort_plan), 'cohort_plan_sha256': digest(args.cohort_plan),
        'identity_smoke_sha256': digest(args.identity_smoke), 'reader': anchor['reader'],
        'checkpoint_sha256': anchor['visual_state_identity_files_sha256'],
        'processor_files_sha256': reader_files, 'archive': str(args.image_archive),
        'archive_sha256': anchor['archive_sha256'], 'prepared': str(args.prepared),
        'prepared_audit_sha256': digest(audit_path), 'manifests_sha256': digest(manifests_path),
        'output_root': str(args.output_root), 'source_sha256': sources,
        'evaluation_label_files_read': [],
        'execution_graph_protocol': 'explicit recurrent/causal-conv callable completion plus child ATen events; durations unknown, never additive',
        'latency_contract': {'replay_scope': 'frozen evidence and raw images through complete greedy answer; excludes evidence retrieval; includes graph observation overhead',
            'full_service_source': 'original frozen serving artifacts, retrieval plus image preparation plus prefill and decode',
            'end_to_end_speedup': None, 'reason': 'diagnostic replay is not a paired optimized full-service execution'},
        'protocol': 'all requests from frozen cohort in original per-user order; no selection by outcomes; original evidence and raw images; full greedy replay; trace first actual decode invocation of each request only',
        'non_goals': ['capacity configuration optimization', 'new evidence policy',
                      'new quality claim', 'PIM speedup claim', 'cross-request KV reuse claim']}
    plan_path = args.output_root / 'execution_plan.json'
    with plan_path.open('x') as handle:
        json.dump(plan, handle, ensure_ascii=False, indent=2)
    return plan_path, plan


def worker(plan_path, worker_index):
    import torch
    import transformers
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from omni_execution_graph import ExecutionGraph, tensors_in
    from omni_visual_feature_store import VisualFeatureStore
    from omni_execution_input_contract import execution_input_contract
    from run_omni_raw_image_smoke import SYSTEM

    plan = json.loads(plan_path.read_text())
    check_sources(plan)
    job = plan['jobs'][worker_index]
    root = Path(plan['output_root']) / ('user_' + job['person_id'])
    root.mkdir(exist_ok=False)
    smoke_path = Path(job['smoke'])
    if digest(smoke_path) != job['smoke_sha256']:
        raise ValueError('frozen serving result changed')
    smoke = json.loads(smoke_path.read_text())
    if hashlib.sha256(SYSTEM.encode()).hexdigest() != smoke['system_prompt_sha256']:
        raise ValueError('frozen system prompt differs')
    checkpoint = Path(plan['reader'])
    pinned_stats = {str(path): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in [Path(plan['archive']), *[checkpoint / name for name in plan['checkpoint_sha256']]]}
    for name, expected in plan['processor_files_sha256'].items():
        if digest(checkpoint / name) != expected:
            raise ValueError('processor identity differs')
    manifest_path = Path(plan['prepared']) / 'manifests.jsonl'
    if digest(manifest_path) != plan['manifests_sha256']:
        raise ValueError('serving input manifests changed')
    manifests = {row['question_id']: row for row in map(json.loads, manifest_path.read_text().splitlines())}
    processor = AutoProcessor.from_pretrained(str(checkpoint), local_files_only=True)
    processor.image_processor.size['longest_edge'] = smoke['image_max_pixels']
    processor_identity = identity(processor.image_processor.to_dict())
    device = 'cuda:0'
    reader = AutoModelForImageTextToText.from_pretrained(str(checkpoint), local_files_only=True,
        dtype=torch.bfloat16, attn_implementation='sdpa').to(device).eval()
    stop = reader.config.text_config.eos_token_id
    stops = set(stop if isinstance(stop, list) else [stop])
    prefix_index = PrefixOpportunityIndex()
    rows = []
    with zipfile.ZipFile(plan['archive']) as archive:
        for original in smoke['rows']:
            request_started = time.perf_counter()
            question_id = original['question_id']
            manifest = manifests[question_id]
            content = [{'type': 'text', 'text': '原始对话记忆：\n局部对话超边：\n'
                + original['hypergraph_evidence_plan']['text']
                + f"\n\n查询时间协议：完整历史末端（{manifest['effective_timestamp']}）"
                + f"\n问题：{manifest['question']}\n请依据文字和随后附上的原始图片作答。"}]
            images = []
            for image in original['images']:
                raw = archive.read(image['archive_member'])
                if hashlib.sha256(raw).hexdigest() != image['sha256']:
                    raise ValueError('frozen original-image bytes differ')
                decoded = Image.open(io.BytesIO(raw)).convert('RGB')
                images.append(decoded)
                content.extend([{'type': 'text', 'text':
                    f"以下原图来自会话 {image['source_session_id']} 的对话轮 {image['source_atom_id']}："},
                    {'type': 'image', 'image': decoded}])
            messages = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': content}]
            text = processor.apply_chat_template(messages, tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
            inputs = processor(text=[text], images=images, return_tensors='pt')
            if (inputs.input_ids.shape[1] != original['input_tokens']
                    or inputs.pixel_values.shape[0] != original['visual_patches']):
                raise ValueError('recompiled actual prompt shape differs from frozen execution')
            input_contract = execution_input_contract(inputs, original)
            token_ids = inputs.input_ids[0].tolist()
            grids = inputs.image_grid_thw.tolist()
            batch_identity = identity({'images': original['images'], 'grids': grids,
                'pixels_sha256': VisualFeatureStore.tensor_digest(inputs.pixel_values),
                'processor': processor_identity, 'checkpoint': plan['checkpoint_sha256']})
            symbols = multimodal_prefix_symbols(token_ids, reader.config.image_token_id,
                [grid[0] * grid[1] * grid[2] // reader.model.visual.spatial_merge_size**2 for grid in grids],
                batch_identity)
            prefix = prefix_index.observe(question_id, symbols)
            context_identity = identity({'person_id': job['person_id'], 'prompt': token_ids,
                'batch': batch_identity, 'model': plan['checkpoint_sha256']})
            graph = ExecutionGraph(context_identity=context_identity)
            with (root / (question_id + '.input.json')).open('x') as handle:
                json.dump({'question_id': question_id, 'input_token_ids': token_ids,
                    'ordered_batch_identity': batch_identity, 'image_grid_thw': grids,
                    'input_identity': context_identity, 'past_only_prefix_opportunity': prefix,
                    'accepted_atom_ids': original['hypergraph_evidence_plan']['accepted_atom_ids'],
                    'selected_hyperedges': original['hypergraph_evidence_plan']['selected_hyperedges'],
                    'frozen_serving_input_contract': input_contract,
                    'evaluation_label_files_read': []}, handle, ensure_ascii=False)
            inputs = inputs.to(device)
            started = time.perf_counter()
            with torch.inference_mode():
                output = reader(**inputs, use_cache=True, logits_to_keep=1)
                torch.cuda.synchronize(device)
                prefill_seconds = time.perf_counter() - started
                decode_started = time.perf_counter()
                generated, traced = [], False
                reason = 'max_new_tokens'
                for position in range(smoke['max_new_tokens']):
                    token = output.logits[:, -1, :].argmax(-1, keepdim=True)
                    generated.append(int(token.item()))
                    if generated[-1] in stops:
                        reason = 'eos'
                        break
                    if position + 1 == smoke['max_new_tokens']:
                        break
                    if not traced:
                        graph.register_external(token, 'first_generated_input_token')
                        for name, parameter in reader.named_parameters(remove_duplicate=False):
                            graph.register_external(parameter, 'parameter:' + name)
                        for index, layer in enumerate(output.past_key_values.layers):
                            for name, value in vars(layer).items():
                                for tensor in tensors_in(value):
                                    graph.register_external(tensor, f'cache.layer_{index}.{name}')
                        graph.attach_linears(reader)
                        graph.attach_opaque_regions(reader)
                        try:
                            with graph:
                                output = reader(input_ids=token, past_key_values=output.past_key_values,
                                                use_cache=True, logits_to_keep=1)
                        finally:
                            graph.close()
                        traced = True
                    else:
                        output = reader(input_ids=token, past_key_values=output.past_key_values,
                                        use_cache=True, logits_to_keep=1)
            torch.cuda.synchronize(device)
            decode_seconds = time.perf_counter() - decode_started
            request_seconds = time.perf_counter() - request_started
            graph_result = graph.summary()
            graph_path = root / (question_id + '.graph.json')
            with graph_path.open('x') as handle:
                json.dump(graph_result, handle, ensure_ascii=False)
            row = {'question_id': question_id, 'graph': str(graph_path), 'graph_sha256': digest(graph_path),
                'input': str(root / (question_id + '.input.json')),
                'complete_greedy_token_ids_equal_frozen': generated == original['answer_token_ids'],
                'stop_reason_equal_frozen': reason == original['stop_reason'],
                'answer_token_ids': generated, 'stop_reason': reason,
                'decode_step_traced': traced, 'events': len(graph_result['events']),
                'linear_calls_observed': len(graph_result['linear_events']),
                'shared_linear_input_groups': len(graph_result['shared_linear_input_values']),
                'past_only_prefix_opportunity': prefix,
                'opaque_callable_regions': len(graph_result['opaque_regions']),
                'diagnostic_latency': {'frozen_evidence_request_seconds': request_seconds,
                    'input_preparation_seconds': started - request_started,
                    'prefill_seconds': prefill_seconds, 'decode_with_graph_probe_seconds': decode_seconds,
                    'evidence_retrieval_included': False, 'graph_observation_included': True,
                    'performance_comparable': False},
                'diagnostic_replay_seconds_not_service_latency': time.perf_counter() - started}
            row['input_sha256'] = digest(Path(row['input']))
            rows.append(row)
            (root / 'partial.json').write_text(json.dumps({'complete': False, 'rows': rows}, indent=2))
            print(json.dumps({key: row[key] for key in ('question_id', 'complete_greedy_token_ids_equal_frozen',
                'events', 'linear_calls_observed', 'shared_linear_input_groups')}, ensure_ascii=False), flush=True)
            del output, inputs, graph, graph_result
            torch.cuda.empty_cache()
    check_sources(plan)
    for name, expected in pinned_stats.items():
        path = Path(name)
        if (path.stat().st_size, path.stat().st_mtime_ns) != expected:
            raise ValueError('pinned model or archive changed during execution')
    for name, expected in plan['processor_files_sha256'].items():
        if digest(checkpoint / name) != expected:
            raise ValueError('processor changed during execution')
    result = {'complete': True, 'person_id': job['person_id'], 'rows': rows,
        'execution_plan_sha256': digest(plan_path), 'producer_smoke_sha256': job['smoke_sha256'],
        'torch': torch.__version__, 'transformers': transformers.__version__,
        'evaluation_label_files_read': [], 'performance_comparable': False,
        'new_quality_evaluated': False, 'end_to_end_speedup': None,
        'boundary': 'frozen-evidence execution replay; complete greedy output checked against prior independent serving artifact; ATen-visible first-decode-step graph only, opaque/custom kernels may require extra instrumentation; no full-service timing or PIM execution claim'}
    with (root / 'result.json').open('x') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cohort-plan', type=Path)
    parser.add_argument('--identity-smoke', type=Path)
    parser.add_argument('--prepared', type=Path)
    parser.add_argument('--image-archive', type=Path)
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--gpus', nargs='+', default=['0'])
    parser.add_argument('--worker-plan', type=Path)
    parser.add_argument('--worker-index', type=int)
    args = parser.parse_args()
    if args.worker_plan is not None:
        worker(args.worker_plan, args.worker_index)
        return
    if any(getattr(args, name) is None for name in
           ('cohort_plan', 'identity_smoke', 'prepared', 'image_archive', 'output_root')):
        parser.error('cohort, identity anchor, serving inputs, archive and fresh output root required')
    args.output_root = args.output_root.resolve()
    plan_path, plan = register(args)
    active, pending, completed = {}, list(range(len(plan['jobs']))), []
    while active or pending:
        for index in tuple(pending):
            gpu = plan['jobs'][index]['gpu']
            if gpu in active:
                continue
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS='2',
                               MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
            log = (args.output_root / f'worker_{index}.log').open('x')
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                '--worker-plan', str(plan_path), '--worker-index', str(index)],
                env=environment, stdout=log, stderr=subprocess.STDOUT)
            active[gpu] = (index, process, log)
            pending.remove(index)
        for gpu, (index, process, log) in tuple(active.items()):
            code = process.poll()
            if code is not None:
                log.close()
                completed.append({'worker_index': index, 'exit_code': code})
                del active[gpu]
        if active:
            time.sleep(2)
    status = {'all_workers_completed': True, 'workers': completed,
              'all_workers_succeeded': all(item['exit_code'] == 0 for item in completed),
              'execution_plan_sha256': digest(plan_path)}
    with (args.output_root / 'execution_status.json').open('x') as handle:
        json.dump(status, handle, indent=2)
    print(json.dumps(status), flush=True)
    if not status['all_workers_succeeded']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
